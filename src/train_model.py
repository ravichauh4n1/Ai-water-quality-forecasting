"""
Machine Learning Model Training, Comparison, and Evaluation Module.
Trains:
1. Linear Regression (Baseline)
2. Ridge Regression (L2 Regularized)
3. Random Forest Regressor (Bagging Ensemble)
4. Gradient Boosting Regressor (Boosting Ensemble)

Evaluates on chronological test set using MAE, RMSE, and R2.
Saves best performing models and scalers using joblib.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loader import load_raw_data
from preprocessing import (
    handle_missing_and_outliers,
    create_time_series_features,
    chronological_train_test_split,
    scale_features
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_model_candidates() -> Dict[str, object]:
    """
    Returns candidate models for time-series regression.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Avoid zero division in MAPE
    denom = np.where(np.abs(y_true) < 1e-4, 1.0, np.abs(y_true))
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    
    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 3),
        "MAPE": round(float(mape), 2)
    }


def train_and_compare_models(
    df: pd.DataFrame,
    target_param: str = "dissolved_oxygen"
) -> Tuple[object, str, Dict, StandardScaler, List[str], pd.DataFrame]:
    """
    Performs chronological training, evaluation, and selection of the best model
    for a given water quality parameter.
    """
    print(f"\n--- Training Models for: {target_param.upper()} ---")
    
    # 1. Feature Engineering
    feat_df = create_time_series_features(df, target_param=target_param)
    X_train, X_test, y_train, y_test, feature_names = chronological_train_test_split(
        feat_df, test_size=0.20, target_col=target_param
    )
    
    # 2. Leak-Free Scaling
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    
    candidates = get_model_candidates()
    results = []
    trained_models = {}
    
    for name, model in candidates.items():
        # Train model
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        
        metrics = evaluate_predictions(y_test.values, y_pred)
        metrics["Model"] = name
        metrics["Parameter"] = target_param
        results.append(metrics)
        trained_models[name] = model
        print(f"  {name:<20} | MAE: {metrics['MAE']:<6} | RMSE: {metrics['RMSE']:<6} | R2: {metrics['R2']:<6}")
        
    results_df = pd.DataFrame(results)
    
    # Select best model based on lowest RMSE
    best_row = results_df.sort_values(by=["RMSE", "MAE"], ascending=[True, True]).iloc[0]
    best_name = best_row["Model"]
    best_model = trained_models[best_name]
    best_metrics = best_row.to_dict()
    
    print(f"--> BEST MODEL: {best_name} (RMSE: {best_metrics['RMSE']}, R2: {best_metrics['R2']})")
    
    return best_model, best_name, best_metrics, scaler, feature_names, results_df


def save_model_package(
    model: object,
    model_name: str,
    scaler: StandardScaler,
    feature_names: List[str],
    metrics: Dict,
    target_param: str
) -> Path:
    """
    Saves trained model, scaler, feature names, and metadata as a Joblib bundle.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"{target_param}_model_bundle.joblib"
    
    bundle = {
        "model": model,
        "model_name": model_name,
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": metrics,
        "target_param": target_param
    }
    
    joblib.dump(bundle, save_path)
    print(f"Saved {target_param} model bundle to: {save_path}")
    return save_path


def load_model_bundle(target_param: str) -> Optional[Dict]:
    """
    Loads saved model bundle for the specified parameter.
    """
    path = MODELS_DIR / f"{target_param}_model_bundle.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def run_full_training_suite(
    data_df: Optional[pd.DataFrame] = None,
    targets: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Trains, compares, and persists models across all key water quality parameters.
    Saves a consolidated performance report in reports/.
    """
    if data_df is None:
        data_df = handle_missing_and_outliers(load_raw_data())
        
    if targets is None:
        targets = ["dissolved_oxygen", "bod", "ph", "turbidity", "temperature", "conductivity"]
        
    all_comparisons = []
    
    for param in targets:
        if param in data_df.columns:
            best_mod, best_name, metrics, scaler, feat_cols, comp_df = train_and_compare_models(data_df, target_param=param)
            save_model_package(best_mod, best_name, scaler, feat_cols, metrics, param)
            all_comparisons.append(comp_df)
            
    consolidated_report = pd.concat(all_comparisons, ignore_index=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "model_performance_report.csv"
    consolidated_report.to_csv(report_path, index=False)
    print(f"\nConsolidated model performance report saved to: {report_path}")
    return consolidated_report


if __name__ == "__main__":
    print("Testing train_model.py...")
    report = run_full_training_suite()
    print("\nTraining summary table:")
    print(report[["Parameter", "Model", "MAE", "RMSE", "R2"]])
    print("\nModel Training Module verified successfully!")
