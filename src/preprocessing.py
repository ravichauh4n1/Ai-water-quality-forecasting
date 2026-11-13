"""
Preprocessing & Time-Series Feature Engineering Module.
Implements:
1. Time-series missing value imputation (forward/backward fill & rolling median).
2. Domain physical boundary clamping.
3. Cyclical calendar harmonics (sin/cos of day of year and month).
4. Past-only lag features (lag_1, lag_7, lag_14, lag_30).
5. Past-only rolling window features (7-day, 30-day rolling means and volatility).
6. Strict chronological train-test splitting (zero lookahead leakage).
7. Leak-free feature scaling with StandardScaler.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "cleaned_ganga_water_quality.csv"

# Physical and scientific domain boundaries for water parameters
PHYSICAL_BOUNDS = {
    "ph": (0.0, 14.0),
    "dissolved_oxygen": (0.0, 25.0),
    "temperature": (0.0, 45.0),
    "bod": (0.0, 100.0),
    "cod": (0.0, 300.0),
    "turbidity": (0.0, 2000.0),
    "conductivity": (0.0, 5000.0),
    "fecal_coliform": (0.0, 1000000.0)
}

WATER_QUALITY_PARAMS = [
    "dissolved_oxygen", "bod", "ph", "turbidity", "temperature", "conductivity", "cod"
]


def handle_missing_and_outliers(df: pd.DataFrame, station_col: str = "station_code") -> pd.DataFrame:
    """
    Cleans missing values and applies physical boundary constraints per station.
    Uses forward fill followed by backward fill to maintain time-series continuity.
    """
    df = df.copy()
    
    # Sort chronologically
    sort_cols = [station_col, "date"] if station_col in df.columns else ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Process per station
    if station_col in df.columns:
        grouped = []
        for _, group in df.groupby(station_col):
            grp = group.copy()
            for param in WATER_QUALITY_PARAMS:
                if param in grp.columns:
                    # Fill missing values with forward fill, then backward fill
                    grp[param] = grp[param].ffill().bfill()
                    # If still NaN (e.g. whole column empty for station), fill with median
                    if grp[param].isna().any():
                        grp[param] = grp[param].fillna(grp[param].median())
                    
                    # Apply physical domain bounds
                    if param in PHYSICAL_BOUNDS:
                        low, high = PHYSICAL_BOUNDS[param]
                        grp[param] = grp[param].clip(lower=low, upper=high)
            grouped.append(grp)
        df = pd.concat(grouped, ignore_index=True)
    else:
        for param in WATER_QUALITY_PARAMS:
            if param in df.columns:
                df[param] = df[param].ffill().bfill()
                if param in PHYSICAL_BOUNDS:
                    low, high = PHYSICAL_BOUNDS[param]
                    df[param] = df[param].clip(lower=low, upper=high)
                    
    return df


def create_time_series_features(
    df: pd.DataFrame,
    target_param: str = "dissolved_oxygen",
    lags: List[int] = [1, 7, 14, 30],
    rolling_windows: List[int] = [7, 30],
    station_col: str = "station_code"
) -> pd.DataFrame:
    """
    Constructs time-series features:
    - Calendar harmonics (sin/cos day of year, sin/cos month)
    - Past lag observations (lag_1, lag_7, lag_14, lag_30)
    - Past rolling statistics (rolling_mean_7, rolling_std_7, rolling_mean_30)
    
    To strictly avoid data leakage, rolling calculations use shift(1) so the current
    day's observation is NEVER included in the rolling input.
    """
    df = df.copy()
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"])
        
    sort_cols = [station_col, "date"] if station_col in df.columns else ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # 1. Calendar & Cyclical Harmonics
    day_of_year = df["date"].dt.dayofyear
    month = df["date"].dt.month
    
    df["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    df["month"] = month
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # 2. Station-grouped Lag and Rolling Features
    feature_dfs = []
    has_station = station_col in df.columns
    groups = df.groupby(station_col) if has_station else [("all", df)]
    
    for _, group in groups:
        grp = group.copy()
        
        # Target lag features
        if target_param in grp.columns:
            for lag in lags:
                grp[f"{target_param}_lag_{lag}"] = grp[target_param].shift(lag)
                
            # Rolling statistics (using shift(1) to avoid lookahead leakage)
            shifted = grp[target_param].shift(1)
            for window in rolling_windows:
                grp[f"{target_param}_roll_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
                grp[f"{target_param}_roll_std_{window}"] = shifted.rolling(window=window, min_periods=1).std().fillna(0)
                
        # Auxiliary parameter lags (e.g. Temperature and Turbidity influence DO & BOD)
        aux_params = [p for p in ["temperature", "ph", "turbidity"] if p in grp.columns and p != target_param]
        for aux in aux_params:
            grp[f"{aux}_lag_1"] = grp[aux].shift(1)
            grp[f"{aux}_lag_7"] = grp[aux].shift(7)
            
        feature_dfs.append(grp)
        
    result_df = pd.concat(feature_dfs, ignore_index=True)
    
    # Drop rows that have NaN from initial max lag
    max_lag = max(lags) if lags else 1
    result_df = result_df.dropna().reset_index(drop=True)
    return result_df


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "dissolved_oxygen"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Chronological train-test split for time-series forecasting.
    Avoids random splitting to ensure realistic historical evaluation.
    Returns: X_train, X_test, y_train, y_test, feature_names.
    """
    df = df.sort_values("date").reset_index(drop=True)
    
    # Identify feature columns if not explicitly passed
    if feature_cols is None:
        exclude_cols = [
            "date", "station_code", "station_name", "river", "state",
            "latitude", "longitude", "is_demonstration_data", target_col
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        
    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test, feature_cols


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fits StandardScaler strictly on training set and transforms both train and test.
    Prevents mean/variance data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def run_preprocessing_pipeline(
    raw_df: pd.DataFrame,
    save_processed: bool = True
) -> pd.DataFrame:
    """
    Runs end-to-end data cleaning and stores cleaned dataset in dataset/processed.
    """
    print("Running preprocessing pipeline...")
    cleaned_df = handle_missing_and_outliers(raw_df)
    
    if save_processed:
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Cleaned dataset saved to: {PROCESSED_DATA_PATH} ({len(cleaned_df):,} rows)")
        
    return cleaned_df


if __name__ == "__main__":
    from data_loader import load_raw_data
    
    print("Testing preprocessing.py...")
    raw = load_raw_data()
    clean = run_preprocessing_pipeline(raw)
    
    # Test feature engineering for Dissolved Oxygen
    feat_df = create_time_series_features(clean, target_param="dissolved_oxygen")
    print(f"Engineered features dataframe shape: {feat_df.shape}")
    
    X_tr, X_te, y_tr, y_te, feats = chronological_train_test_split(feat_df, target_col="dissolved_oxygen")
    print(f"Chronological split: Train {len(X_tr)} rows, Test {len(X_te)} rows. Features: {len(feats)}")
    
    X_tr_sc, X_te_sc, scaler = scale_features(X_tr, X_te)
    print("Feature scaling completed successfully!")
    print("Preprocessing Module verified!")
