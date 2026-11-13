"""
Exploratory Data Analysis (EDA) & Interactive Visualization Module.
Provides statistical analysis and high-performance Plotly visual figures
styled with the project's river and water blue theme.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Water and River UI Theme Colors
COLOR_PALETTE = {
    "deep_blue": "#0A2540",
    "primary_blue": "#1E88E5",
    "light_blue": "#00BCD4",
    "teal": "#009688",
    "monsoon_amber": "#FF9800",
    "alert_red": "#E53935",
    "safe_green": "#43A047",
    "card_bg": "#F8FAFC",
    "grid_color": "#E2E8F0"
}

SEASONS_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon"
}


def compute_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive parametric and non-parametric statistics for water parameters.
    """
    numeric_cols = [c for c in ["dissolved_oxygen", "bod", "cod", "ph", "turbidity", "temperature", "conductivity", "fecal_coliform"] if c in df.columns]
    stats_list = []
    
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        stats_list.append({
            "Parameter": col.replace("_", " ").title(),
            "Count": int(series.count()),
            "Mean": round(float(series.mean()), 2),
            "Std Dev": round(float(series.std()), 2),
            "Median": round(float(series.median()), 2),
            "IQR": round(float(q75 - q25), 2),
            "Min": round(float(series.min()), 2),
            "Max": round(float(series.max()), 2),
            "Skewness": round(float(series.skew()), 2)
        })
        
    return pd.DataFrame(stats_list)


def compute_station_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates water quality parameters by monitoring station to assess water quality gradients.
    """
    if "station_name" not in df.columns:
        return pd.DataFrame()
        
    numeric_cols = [c for c in ["dissolved_oxygen", "bod", "ph", "turbidity", "temperature", "conductivity"] if c in df.columns]
    agg_dict = {col: "mean" for col in numeric_cols}
    station_summary = df.groupby(["station_code", "station_name"] if "station_code" in df.columns else "station_name").agg(agg_dict).round(2).reset_index()
    return station_summary


# ==========================================
# INTERACTIVE PLOTLY VISUALIZATIONS
# ==========================================

def plot_parameter_timeseries(
    df: pd.DataFrame,
    station_name: Optional[str] = None,
    parameter: str = "dissolved_oxygen"
) -> go.Figure:
    """
    Interactive time-series line chart with range slider and threshold alert bands.
    """
    filtered = df.copy()
    if station_name and "station_name" in filtered.columns:
        filtered = filtered[filtered["station_name"] == station_name]
        
    filtered = filtered.sort_values("date")
    
    title_str = f"Historical Trend: {parameter.replace('_', ' ').title()}"
    if station_name:
        title_str += f" at {station_name}"
        
    fig = go.Figure()
    
    # Historical observation line
    fig.add_trace(go.Scatter(
        x=filtered["date"],
        y=filtered[parameter],
        mode="lines",
        name=parameter.replace("_", " ").title(),
        line=dict(color=COLOR_PALETTE["primary_blue"], width=2),
        hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Value</b>: %{y:.2f}<extra></extra>"
    ))
    
    # 30-day moving average trend line
    rolling_series = filtered[parameter].rolling(window=30, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=filtered["date"],
        y=rolling_series,
        mode="lines",
        name="30-Day Trend (Smoothed)",
        line=dict(color=COLOR_PALETTE["monsoon_amber"], width=2.5, dash="dash")
    ))
    
    # CPCB Minimum DO benchmark line (5.0 mg/L)
    if parameter == "dissolved_oxygen":
        fig.add_hline(
            y=5.0, line_dash="dot", line_color=COLOR_PALETTE["alert_red"],
            annotation_text="CPCB Bathing Standard Min DO (5.0 mg/L)", annotation_position="bottom right"
        )
    elif parameter == "bod":
        fig.add_hline(
            y=3.0, line_dash="dot", line_color=COLOR_PALETTE["alert_red"],
            annotation_text="CPCB Bathing Standard Max BOD (3.0 mg/L)", annotation_position="top right"
        )
        
    fig.update_layout(
        title=dict(text=title_str, font=dict(size=16, color=COLOR_PALETTE["deep_blue"])),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            gridcolor=COLOR_PALETTE["grid_color"]
        ),
        yaxis=dict(
            title=f"{parameter.replace('_', ' ').title()}",
            gridcolor=COLOR_PALETTE["grid_color"]
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_station_comparison_bar(
    df: pd.DataFrame,
    parameter: str = "dissolved_oxygen"
) -> go.Figure:
    """
    Bar chart comparing mean parameter values across all monitoring stations.
    """
    if "station_name" not in df.columns:
        return go.Figure()
        
    summary = df.groupby("station_name")[parameter].mean().round(2).reset_index()
    # Sort upstream to downstream if station_code exists
    if "station_code" in df.columns:
        st_order = df[["station_code", "station_name"]].drop_duplicates().sort_values("station_code")
        summary = summary.merge(st_order, on="station_name").sort_values("station_code")
        
    fig = px.bar(
        summary,
        x="station_name",
        y=parameter,
        color=parameter,
        color_continuous_scale="Blues_r" if parameter in ["bod", "turbidity", "conductivity"] else "Blues",
        labels={parameter: parameter.replace("_", " ").title(), "station_name": "Monitoring Station"},
        title=f"Station-wise Average: {parameter.replace('_', ' ').title()} (Upstream to Downstream)"
    )
    
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=80),
        xaxis_tickangle=-35,
        coloraxis_showscale=False
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Correlation heatmap showing relationships between physicochemical parameters.
    """
    numeric_cols = [c for c in ["dissolved_oxygen", "bod", "cod", "ph", "turbidity", "temperature", "conductivity"] if c in df.columns]
    corr = df[numeric_cols].corr().round(2)
    
    labels = [c.replace("_", " ").title() for c in numeric_cols]
    
    fig = px.imshow(
        corr,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1.0,
        zmax=1.0,
        title="Parameter Correlation Matrix (Spearman / Pearson)"
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_seasonal_distribution(
    df: pd.DataFrame,
    parameter: str = "dissolved_oxygen"
) -> go.Figure:
    """
    Seasonal distribution boxplots capturing Winter, Summer, Monsoon, and Post-Monsoon shifts.
    """
    temp_df = df.copy()
    if not np.issubdtype(temp_df["date"].dtype, np.datetime64):
        temp_df["date"] = pd.to_datetime(temp_df["date"])
        
    temp_df["season"] = temp_df["date"].dt.month.map(SEASONS_MAP)
    order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    
    fig = px.box(
        temp_df,
        x="season",
        y=parameter,
        color="season",
        category_orders={"season": order},
        color_discrete_map={
            "Winter": "#2196F3",
            "Summer": "#FF9800",
            "Monsoon": "#4CAF50",
            "Post-Monsoon": "#009688"
        },
        title=f"Seasonal Variation & Spread: {parameter.replace('_', ' ').title()}"
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        xaxis_title="Season",
        yaxis_title=parameter.replace("_", " ").title(),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def plot_radar_profile(
    df: pd.DataFrame,
    station_name: str
) -> go.Figure:
    """
    Radar chart comparing the station's normalized parameter levels against ideal CPCB Class A levels.
    """
    st_data = df[df["station_name"] == station_name] if "station_name" in df.columns else df
    
    params = ["dissolved_oxygen", "bod", "ph", "turbidity", "conductivity"]
    existing_params = [p for p in params if p in df.columns]
    
    # Normalized scores on 0 to 100 scale (100 = best quality)
    station_means = st_data[existing_params].mean()
    
    scores = []
    ideal_scores = [90, 85, 95, 90, 90]
    
    for p in existing_params:
        val = station_means[p]
        if p == "dissolved_oxygen":
            # Ideal is >= 8.5
            sc = min(100.0, (val / 9.0) * 100.0)
        elif p == "bod":
            # Lower is better (ideal <= 2.0)
            sc = max(10.0, 100.0 - (val / 10.0) * 100.0)
        elif p == "ph":
            # Ideal is 7.5
            diff = abs(val - 7.5)
            sc = max(20.0, 100.0 - diff * 40.0)
        elif p == "turbidity":
            # Lower is better
            sc = max(10.0, 100.0 - (val / 300.0) * 100.0)
        elif p == "conductivity":
            # Lower is better
            sc = max(10.0, 100.0 - (val / 800.0) * 100.0)
        else:
            sc = 70.0
        scores.append(round(sc, 1))
        
    categories = [p.replace("_", " ").upper() for p in existing_params]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name=station_name,
        line=dict(color=COLOR_PALETTE["primary_blue"])
    ))
    fig.add_trace(go.Scatterpolar(
        r=ideal_scores[:len(existing_params)] + [ideal_scores[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="CPCB Class A Benchmark",
        line=dict(color=COLOR_PALETTE["safe_green"], dash="dash")
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        title=f"Water Quality Profile Radar: {station_name}",
        template="plotly_white",
        margin=dict(l=60, r=60, t=60, b=40)
    )
    return fig


if __name__ == "__main__":
    from data_loader import load_raw_data
    
    print("Testing eda.py...")
    data = load_raw_data()
    stats = compute_statistical_summary(data)
    print("Statistical summary computed for", len(stats), "parameters.")
    rankings = compute_station_rankings(data)
    print("Station rankings computed:")
    print(rankings[["station_name", "dissolved_oxygen", "bod"]].head())
    
    # Test chart generators
    f1 = plot_parameter_timeseries(data, "Kanpur (Jajmau)", "dissolved_oxygen")
    f2 = plot_station_comparison_bar(data, "bod")
    f3 = plot_correlation_heatmap(data)
    f4 = plot_seasonal_distribution(data, "temperature")
    f5 = plot_radar_profile(data, "Rishikesh (Triveni Ghat)")
    print("All 5 interactive Plotly figures generated successfully!")
    print("EDA Module verified!")
