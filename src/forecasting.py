"""
Time-Series Multi-Horizon Forecasting Engine.
Supports:
1. 1-Day, 7-Day, and 30-Day ahead recursive multivariate forecasts.
2. Incorporates historical lags and calendar harmonics.
3. Uncertainty quantification (95% confidence intervals).
4. Interactive Plotly visualization for Actual vs Predicted and Future Trajectory.
5. Multi-parameter simultaneous forecasting for Decision Support System (DSS).
"""

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from data_loader import load_raw_data
from preprocessing import handle_missing_and_outliers
from train_model import load_model_bundle, MODELS_DIR

COLOR_PALETTE = {
    "historical": "#1E88E5",
    "forecast": "#00BCD4",
    "ci_fill": "rgba(0, 188, 212, 0.2)",
    "actual": "#43A047",
    "alert_red": "#E53935",
    "grid_color": "#E2E8F0"
}


def build_feature_vector_for_step(
    current_date: pd.Timestamp,
    recent_history: Dict[str, List[float]],
    feature_names: List[str],
    target_param: str
) -> np.ndarray:
    """
    Constructs the exact 1D feature row required by the model for a future timestamp.
    Calculates cyclical harmonics and extracts correct lag/rolling values from history.
    """
    day_of_year = current_date.dayofyear
    month = current_date.month
    
    vec = {}
    
    # 1. Calendar Harmonics
    vec["sin_day"] = np.sin(2 * np.pi * day_of_year / 365.25)
    vec["cos_day"] = np.cos(2 * np.pi * day_of_year / 365.25)
    vec["sin_month"] = np.sin(2 * np.pi * month / 12.0)
    vec["cos_month"] = np.cos(2 * np.pi * month / 12.0)
    vec["month"] = month
    vec["quarter"] = current_date.quarter
    vec["day_of_week"] = current_date.dayofweek
    
    # 2. Target Lags and Rolling
    target_hist = recent_history.get(target_param, [])
    if len(target_hist) > 0:
        for lag in [1, 7, 14, 30]:
            idx = -lag
            vec[f"{target_param}_lag_{lag}"] = target_hist[idx] if len(target_hist) >= lag else target_hist[-1]
            
        # Rolling stats
        for window in [7, 30]:
            win_vals = target_hist[-window:] if len(target_hist) >= window else target_hist
            vec[f"{target_param}_roll_mean_{window}"] = np.mean(win_vals)
            vec[f"{target_param}_roll_std_{window}"] = np.std(win_vals) if len(win_vals) > 1 else 0.0
            
    # 3. Auxiliary parameter lags
    for aux in ["temperature", "ph", "turbidity"]:
        if aux != target_param and aux in recent_history:
            aux_hist = recent_history[aux]
            vec[f"{aux}_lag_1"] = aux_hist[-1] if len(aux_hist) >= 1 else 20.0
            vec[f"{aux}_lag_7"] = aux_hist[-7] if len(aux_hist) >= 7 else aux_hist[-1]
            
    # Assemble feature DataFrame with proper column names
    row = {f: vec.get(f, 0.0) for f in feature_names}
    return pd.DataFrame([row], columns=feature_names)


def generate_forecast(
    station_df: pd.DataFrame,
    target_param: str = "dissolved_oxygen",
    horizon_days: int = 7
) -> pd.DataFrame:
    """
    Generates multi-step recursive forecasts for 1, 7, or 30 days ahead.
    Returns DataFrame with columns: date, forecast, lower_ci_95, upper_ci_95, parameter.
    """
    bundle = load_model_bundle(target_param)
    if bundle is None:
        raise FileNotFoundError(f"Model bundle for parameter '{target_param}' not found in {MODELS_DIR}. Train models first.")
        
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    metrics = bundle.get("metrics", {})
    rmse = metrics.get("RMSE", 0.5)
    
    # Prepare historical context (last 60 days)
    df_sorted = station_df.sort_values("date").reset_index(drop=True)
    last_date = pd.to_datetime(df_sorted["date"].iloc[-1])
    
    # Extract recent series for target and auxiliaries
    recent_history = {}
    for col in [target_param, "temperature", "ph", "turbidity"]:
        if col in df_sorted.columns:
            recent_history[col] = df_sorted[col].tolist()
            
    forecast_dates = []
    forecast_values = []
    lower_bounds = []
    upper_bounds = []
    
    current_date = last_date
    for step in range(1, horizon_days + 1):
        current_date += timedelta(days=1)
        
        # Build feature row
        X_step = build_feature_vector_for_step(current_date, recent_history, feature_names, target_param)
        X_step_scaled = scaler.transform(X_step)
        
        # Predict
        pred_val = float(model.predict(X_step_scaled)[0])
        
        # Apply physical minimum clamp
        if target_param in ["dissolved_oxygen", "bod", "turbidity", "conductivity"]:
            pred_val = max(0.1, pred_val)
        elif target_param == "ph":
            pred_val = np.clip(pred_val, 1.0, 14.0)
            
        pred_val = round(pred_val, 2)
        
        # Uncertainty grows with sqrt of horizon step
        step_sigma = rmse * np.sqrt(1.0 + 0.15 * (step - 1))
        ci_lower = round(max(0.0, pred_val - 1.96 * step_sigma), 2)
        ci_upper = round(pred_val + 1.96 * step_sigma, 2)
        
        forecast_dates.append(current_date)
        forecast_values.append(pred_val)
        lower_bounds.append(ci_lower)
        upper_bounds.append(ci_upper)
        
        # Append prediction to history to recursively update lag features
        recent_history[target_param].append(pred_val)
        
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast_values,
        "lower_ci_95": lower_bounds,
        "upper_ci_95": upper_bounds,
        "parameter": target_param
    })
    return forecast_df


def generate_multivariate_forecast(
    station_df: pd.DataFrame,
    params: Optional[List[str]] = None,
    horizon_days: int = 7
) -> pd.DataFrame:
    """
    Generates multi-step forecasts for all core water quality parameters for a station.
    Used by Decision Support System (DSS) to forecast upcoming Water Quality Index.
    """
    if params is None:
        params = ["dissolved_oxygen", "bod", "ph", "turbidity", "temperature", "conductivity"]
        
    combined_forecasts = []
    for p in params:
        try:
            fc = generate_forecast(station_df, target_param=p, horizon_days=horizon_days)
            combined_forecasts.append(fc)
        except Exception as e:
            print(f"Warning: Could not forecast {p}: {e}")
            
    if not combined_forecasts:
        return pd.DataFrame()
        
    # Pivot to wide format: date, dissolved_oxygen, bod, ph, ...
    all_fc = pd.concat(combined_forecasts, ignore_index=True)
    pivoted = all_fc.pivot(index="date", columns="parameter", values="forecast").reset_index()
    return pivoted


def plot_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    parameter: str = "dissolved_oxygen",
    station_name: str = "River Station",
    history_days_to_show: int = 60
) -> go.Figure:
    """
    Generates an interactive Plotly chart combining recent historical data,
    future predictions, and shaded 95% confidence intervals.
    """
    fig = go.Figure()
    
    # 1. Historical Segment
    recent_hist = historical_df.sort_values("date").tail(history_days_to_show)
    fig.add_trace(go.Scatter(
        x=recent_hist["date"],
        y=recent_hist[parameter],
        mode="lines",
        name="Historical Observation",
        line=dict(color=COLOR_PALETTE["historical"], width=2.5),
        hovertemplate="<b>Observed Date</b>: %{x|%Y-%m-%d}<br><b>Actual</b>: %{y:.2f}<extra></extra>"
    ))
    
    # Bridge point: connect last historical point with first forecast
    last_hist_date = pd.to_datetime(recent_hist["date"].iloc[-1])
    last_hist_val = float(recent_hist[parameter].iloc[-1])
    
    fc_dates = [last_hist_date] + forecast_df["date"].tolist()
    fc_vals = [last_hist_val] + forecast_df["forecast"].tolist()
    ci_low = [last_hist_val] + forecast_df["lower_ci_95"].tolist()
    ci_up = [last_hist_val] + forecast_df["upper_ci_95"].tolist()
    
    # 2. Confidence Interval Shaded Band
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=ci_up + ci_low[::-1],
        fill="toself",
        fillcolor=COLOR_PALETTE["ci_fill"],
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval"
    ))
    
    # 3. Forecast Line
    fig.add_trace(go.Scatter(
        x=fc_dates,
        y=fc_vals,
        mode="lines+markers",
        name="AI Forecast",
        line=dict(color=COLOR_PALETTE["forecast"], width=3, dash="dash"),
        marker=dict(size=6, color=COLOR_PALETTE["forecast"]),
        hovertemplate="<b>Forecast Date</b>: %{x|%Y-%m-%d}<br><b>Prediction</b>: %{y:.2f}<extra></extra>"
    ))
    
    # 4. Standard CPCB Water Quality Benchmarks
    if parameter == "dissolved_oxygen":
        fig.add_hline(
            y=5.0, line_dash="dot", line_color=COLOR_PALETTE["alert_red"],
            annotation_text="CPCB Bathing Min DO (5.0 mg/L)", annotation_position="bottom right"
        )
    elif parameter == "bod":
        fig.add_hline(
            y=3.0, line_dash="dot", line_color=COLOR_PALETTE["alert_red"],
            annotation_text="CPCB Bathing Max BOD (3.0 mg/L)", annotation_position="top right"
        )
    elif parameter == "ph":
        fig.add_hrect(
            y0=6.5, y1=8.5, line_width=0, fillcolor="rgba(76, 175, 80, 0.1)",
            annotation_text="Safe pH Range (6.5 - 8.5)", annotation_position="top left"
        )
        
    param_title = parameter.replace("_", " ").title()
    fig.update_layout(
        title=dict(
            text=f"AI Forecast Trajectory ({len(forecast_df)} Days): {param_title} at {station_name}",
            font=dict(size=16, color="#0A2540")
        ),
        xaxis=dict(title="Date", gridcolor=COLOR_PALETTE["grid_color"]),
        yaxis=dict(title=param_title, gridcolor=COLOR_PALETTE["grid_color"]),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


if __name__ == "__main__":
    print("Testing forecasting.py...")
    raw = handle_missing_and_outliers(load_raw_data())
    kanpur_df = raw[raw["station_name"] == "Kanpur (Jajmau)"].copy()
    
    # 1. Test 7-Day Forecast for Dissolved Oxygen
    fc_7d = generate_forecast(kanpur_df, target_param="dissolved_oxygen", horizon_days=7)
    print("7-Day Forecast for DO at Kanpur:")
    print(fc_7d)
    
    # 2. Test 30-Day Forecast for BOD
    fc_30d = generate_forecast(kanpur_df, target_param="bod", horizon_days=30)
    print(f"\n30-Day Forecast for BOD at Kanpur generated ({len(fc_30d)} steps).")
    
    # 3. Test Multivariate Forecast for DSS
    multi_fc = generate_multivariate_forecast(kanpur_df, horizon_days=7)
    print("\nMultivariate 7-Day Forecast for DSS:")
    print(multi_fc.head())
    
    # 4. Test Plotly Visual Generator
    fig = plot_forecast_chart(kanpur_df, fc_7d, parameter="dissolved_oxygen", station_name="Kanpur (Jajmau)")
    print("Plotly forecast figure built successfully!")
    print("Forecasting Module verified successfully!")
