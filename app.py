"""
AI Water Quality Forecasting System - Streamlit Web Dashboard
Project ID: P18 | Group: OJ18
SIH Problem Statement: SIH1694 - Real-time Ganga River Water Quality Forecasting
using AI-enabled DSS, satellite data, IoT, and dynamic models.

Author: Antigravity AI Engineering
"""

import sys
from pathlib import Path

# Add project root and src directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import core system modules
from data_loader import (
    load_raw_data,
    load_stations_metadata,
    validate_uploaded_file,
    get_dataset_summary
)
from preprocessing import (
    handle_missing_and_outliers,
    create_time_series_features,
    chronological_train_test_split,
    scale_features
)
from eda import (
    compute_statistical_summary,
    compute_station_rankings,
    plot_parameter_timeseries,
    plot_station_comparison_bar,
    plot_correlation_heatmap,
    plot_seasonal_distribution,
    plot_radar_profile,
    COLOR_PALETTE
)
from train_model import (
    load_model_bundle,
    train_and_compare_models,
    save_model_package,
    REPORTS_DIR
)
from forecasting import (
    generate_forecast,
    generate_multivariate_forecast,
    plot_forecast_chart
)
from risk_detection import (
    calculate_water_quality_index,
    classify_risk,
    detect_parameter_anomalies,
    generate_dss_recommendations,
    plot_wqi_gauge,
    DISCLAIMER_TEXT
)

# ==========================================
# STREAMLIT PAGE CONFIG & RIVER-THEMED CSS
# ==========================================
st.set_page_config(
    page_title="AI Water Quality Forecasting System | SIH1694",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom River & Water Blue CSS Theme
st.markdown("""
<style>
    /* Primary Colors & Typography */
    :root {
        --primary-blue: #0A2540;
        --river-blue: #1E88E5;
        --cyan-accent: #00BCD4;
        --bg-light: #F8FAFC;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0A2540 0%, #1E88E5 50%, #00BCD4 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(10, 37, 64, 0.15);
    }
    .main-header h1 {
        color: white !important;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        padding-bottom: 6px;
    }
    .main-header p {
        color: #E0F2FE;
        font-size: 14px;
        margin: 0;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background-color: white;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid #1E88E5;
        margin-bottom: 15px;
    }
    .kpi-title {
        color: #64748B;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #0A2540;
        font-size: 24px;
        font-weight: 700;
    }
    .kpi-sub {
        color: #94A3B8;
        font-size: 12px;
        margin-top: 4px;
    }
    
    /* Disclaimer Card */
    .disclaimer-card {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 13px;
        color: #92400E;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Alert Badge Cards */
    .alert-card-critical {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        padding: 12px 16px;
        border-radius: 6px;
        color: #991B1B;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .alert-card-warning {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        color: #92400E;
        margin-bottom: 10px;
        font-size: 14px;
    }
    .alert-card-success {
        background-color: #DCFCE7;
        border-left: 5px solid #10B981;
        padding: 12px 16px;
        border-radius: 6px;
        color: #065F46;
        margin-bottom: 10px;
        font-size: 14px;
    }
    /* Table Container Styling */
    .table-container {
        width: 100%;
        max-height: 420px;
        overflow-y: auto;
        overflow-x: auto;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin: 12px 0;
        border: 1px solid #E2E8F0;
    }
    .river-table {
        width: 100%;
        border-collapse: collapse;
        font-family: inherit;
        font-size: 13px;
        text-align: left;
    }
    .river-table th {
        background-color: #0A2540;
        color: white;
        font-weight: 600;
        padding: 10px 14px;
        position: sticky;
        top: 0;
        z-index: 1;
    }
    .river-table td {
        padding: 8px 14px;
        border-bottom: 1px solid #E2E8F0;
        color: #334155;
    }
    .river-table tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    .river-table tr:hover {
        background-color: #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


def display_table(df: pd.DataFrame, max_rows: int = 50):
    """
    Renders clean, styled, scrollable tables without requiring pyarrow.
    """
    if df is None or df.empty:
        st.info("No records to display.")
        return
    html = df.head(max_rows).to_html(classes="river-table", index=False, border=0, escape=False)
    st.markdown(f'<div class="table-container">{html}</div>', unsafe_allow_html=True)


# ==========================================
# SESSION STATE & DATA INITIALIZATION
# ==========================================
@st.cache_data
def get_base_data():
    df = handle_missing_and_outliers(load_raw_data())
    stations = load_stations_metadata()
    return df, stations

if "water_data" not in st.session_state:
    base_df, base_stations = get_base_data()
    st.session_state["water_data"] = base_df
    st.session_state["stations_meta"] = base_stations
    st.session_state["is_custom_data"] = False

current_df = st.session_state["water_data"]
stations_df = st.session_state["stations_meta"]

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("### 🌊 Ganga DSS Portal")
    st.markdown("**Project ID**: P18 | **Group**: OJ18")
    st.markdown("**SIH PS**: SIH1694")
    st.markdown("---")
    
    page = st.radio(
        "Navigation Menu",
        [
            "🏠 Home Dashboard",
            "📊 Water Quality Monitoring",
            "🤖 AI Forecasting",
            "⚠️ Pollution Risk Analysis",
            "📈 Historical Data Analysis",
            "📁 Dataset Upload",
            "ℹ️ About Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### River Monitoring Scope")
    st.markdown("• **River Basin**: Ganga River Basin")
    st.markdown("• **Standard**: CPCB Best Use Criteria")
    st.markdown(f"• **Active Stations**: {current_df['station_name'].nunique() if 'station_name' in current_df.columns else 1}")
    if st.session_state.get("is_custom_data"):
        st.info("ℹ️ Using User-Uploaded Custom Dataset")
        if st.button("🔄 Reset to Default CPCB Dataset"):
            base_df, base_stations = get_base_data()
            st.session_state["water_data"] = base_df
            st.session_state["stations_meta"] = base_stations
            st.session_state["is_custom_data"] = False
            st.rerun()


# ==========================================
# PAGE 1: HOME DASHBOARD
# ==========================================
if page == "🏠 Home Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🌊 AI Water Quality Forecasting System</h1>
        <p>Real-Time Decision Support System for Ganga River Water Quality Monitoring, AI-Powered Forecasting, and Pollution Mitigation (SIH1694 | Group OJ18)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate Basin-wide latest metrics
    latest_date = current_df["date"].max()
    latest_slice = current_df[current_df["date"] == latest_date]
    if latest_slice.empty:
        latest_slice = current_df.tail(8)
        
    avg_do = latest_slice["dissolved_oxygen"].mean() if "dissolved_oxygen" in latest_slice.columns else 0.0
    avg_bod = latest_slice["bod"].mean() if "bod" in latest_slice.columns else 0.0
    avg_ph = latest_slice["ph"].mean() if "ph" in latest_slice.columns else 7.0
    avg_turb = latest_slice["turbidity"].mean() if "turbidity" in latest_slice.columns else 0.0
    
    # Compute overall basin WQI
    latest_mean_dict = latest_slice[["dissolved_oxygen", "bod", "ph", "turbidity", "conductivity", "temperature"]].mean().to_dict()
    basin_wqi, _ = calculate_water_quality_index(latest_mean_dict)
    basin_risk = classify_risk(basin_wqi)
    
    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {COLOR_PALETTE['primary_blue']};">
            <div class="kpi-title">Avg Dissolved Oxygen</div>
            <div class="kpi-value">{avg_do:.2f} <span style="font-size:14px;color:#64748B;">mg/L</span></div>
            <div class="kpi-sub">CPCB Target: ≥ 5.0 mg/L</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {'#EF4444' if avg_bod > 3.0 else '#10B981'};">
            <div class="kpi-title">Avg Biochemical Oxygen</div>
            <div class="kpi-value">{avg_bod:.2f} <span style="font-size:14px;color:#64748B;">mg/L</span></div>
            <div class="kpi-sub">CPCB Target: ≤ 3.0 mg/L</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {'#10B981' if 6.5 <= avg_ph <= 8.5 else '#F59E0B'};">
            <div class="kpi-title">Average Water pH</div>
            <div class="kpi-value">{avg_ph:.2f}</div>
            <div class="kpi-sub">Tolerance: 6.5 – 8.5</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #00BCD4;">
            <div class="kpi-title">Avg Turbidity</div>
            <div class="kpi-value">{avg_turb:.1f} <span style="font-size:14px;color:#64748B;">NTU</span></div>
            <div class="kpi-sub">Monsoon Siltation Factor</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: {basin_risk['color']};">
            <div class="kpi-title">Basin Status (WQI)</div>
            <div class="kpi-value" style="color:{basin_risk['color']}">{basin_wqi:.1f}</div>
            <div class="kpi-sub">{basin_risk['badge']}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 🗺️ Ganga River Basin Monitoring Network")
    
    col_map, col_alerts = st.columns([3, 2])
    
    with col_map:
        # Check if coordinates exist
        if "latitude" in stations_df.columns and "longitude" in stations_df.columns:
            map_data = stations_df.copy()
            # Calculate current station WQI
            st_wqis = []
            st_statuses = []
            for _, row in map_data.iterrows():
                st_code = row.get("station_code")
                st_slice = current_df[current_df["station_code"] == st_code] if "station_code" in current_df.columns else current_df
                if not st_slice.empty:
                    m = st_slice[["dissolved_oxygen", "bod", "ph", "turbidity", "conductivity", "temperature"]].tail(1).to_dict(orient="records")[0]
                    w, _ = calculate_water_quality_index(m)
                    r = classify_risk(w)
                    st_wqis.append(w)
                    st_statuses.append(r["category"])
                else:
                    st_wqis.append(50.0)
                    st_statuses.append("Moderate Risk")
                    
            map_data["WQI"] = st_wqis
            map_data["Status"] = st_statuses
            
            # Interactive Map (Compatible with Plotly 7+ scatter_map & legacy scatter_mapbox)
            scatter_fn = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
            fig_map = scatter_fn(
                map_data,
                lat="latitude",
                lon="longitude",
                hover_name="station_name",
                hover_data={"station_code": True, "cpcb_class": True, "WQI": True, "Status": True, "latitude": False, "longitude": False},
                color="Status",
                color_discrete_map={
                    "Low Risk": "#43A047",
                    "Moderate Risk": "#FDD835",
                    "High Risk": "#FB8C00",
                    "Critical Risk": "#E53935"
                },
                zoom=5.5,
                center={"lat": 27.5, "lon": 81.5},
                height=420,
                title="Active CPCB Monitoring Stations along River Ganga"
            )
            if hasattr(fig_map.layout, "map"):
                fig_map.update_layout(map_style="open-street-map")
            elif hasattr(fig_map.layout, "mapbox"):
                fig_map.update_layout(mapbox_style="open-street-map")
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Geographic coordinates not present in dataset.")
            
    with col_alerts:
        st.markdown("#### 🚨 Real-Time Risk & Pollution Alerts")
        # Check Kanpur, Varanasi, Kannauj for anomalies
        found_any = False
        for st_name in ["Kanpur (Jajmau)", "Varanasi (Assi Ghat)", "Kannauj (Upstream)"]:
            st_data = current_df[current_df["station_name"] == st_name] if "station_name" in current_df.columns else current_df
            if not st_data.empty:
                last_row = st_data.iloc[-1]
                anoms = detect_parameter_anomalies(last_row)
                for a in anoms:
                    found_any = True
                    sev_class = "alert-card-critical" if a["severity"] == "Critical" else "alert-card-warning"
                    st.markdown(f"""
                    <div class="{sev_class}">
                        <b>[{a['severity'].upper()}] {st_name}</b><br>
                        {a['message']}
                    </div>
                    """, unsafe_allow_html=True)
                    
        if not found_any:
            st.markdown("""
            <div class="alert-card-success">
                <b>🟢 NORMAL CONDITIONS</b><br>
                All reporting monitoring stations currently operate within baseline thresholds.
            </div>
            """, unsafe_allow_html=True)
            
    # Station Quick Summary Table
    st.markdown("### 📋 Latest Station Observations")
    rankings = compute_station_rankings(current_df)
    display_table(rankings)
    
    st.markdown(f'<div class="disclaimer-card">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 2: WATER QUALITY MONITORING
# ==========================================
elif page == "📊 Water Quality Monitoring":
    st.markdown("## 📊 Station-wise Water Quality Monitoring")
    st.markdown("Inspect live-style sensor readings, historical trends, and parameter compliance.")
    
    stations_list = current_df["station_name"].unique().tolist() if "station_name" in current_df.columns else ["Default Station"]
    
    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_station = st.selectbox("Select Monitoring Station", stations_list, index=0)
    with col_sel2:
        param_options = [
            ("dissolved_oxygen", "Dissolved Oxygen (DO mg/L)"),
            ("bod", "Biochemical Oxygen Demand (BOD mg/L)"),
            ("ph", "pH Level"),
            ("turbidity", "Turbidity (NTU)"),
            ("temperature", "Water Temperature (°C)"),
            ("conductivity", "Electrical Conductivity (µS/cm)")
        ]
        param_keys = [p[0] for p in param_options if p[0] in current_df.columns]
        param_labels = {p[0]: p[1] for p in param_options}
        selected_param = st.selectbox("Select Parameter", param_keys, format_func=lambda x: param_labels.get(x, x))
        
    # Station metadata
    st_df = current_df[current_df["station_name"] == selected_station] if "station_name" in current_df.columns else current_df
    last_rec = st_df.iloc[-1]
    
    # Station metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("DO", f"{last_rec.get('dissolved_oxygen', 0.0):.2f} mg/L", "Target ≥ 5.0")
    with m2:
        st.metric("BOD", f"{last_rec.get('bod', 0.0):.2f} mg/L", "Target ≤ 3.0", delta_color="inverse")
    with m3:
        st.metric("pH", f"{last_rec.get('ph', 7.0):.2f}", "Range 6.5-8.5")
    with m4:
        st.metric("Turbidity", f"{last_rec.get('turbidity', 0.0):.1f} NTU")
    with m5:
        st.metric("Temperature", f"{last_rec.get('temperature', 0.0):.1f} °C")
    with m6:
        st.metric("Conductivity", f"{last_rec.get('conductivity', 0.0):.0f} µS/cm")
        
    st.markdown("---")
    
    # Interactive Time-Series Plot
    st.markdown(f"### 📈 Historical Multi-Year Trend: {param_labels.get(selected_param, selected_param)}")
    fig_ts = plot_parameter_timeseries(current_df, selected_station, selected_param)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Cross-Station Comparative Bar
    st.markdown("### 📊 Cross-Station River Gradient")
    fig_bar = plot_station_comparison_bar(current_df, selected_param)
    st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# PAGE 3: AI WATER QUALITY FORECASTING
# ==========================================
elif page == "🤖 AI Forecasting":
    st.markdown("## 🤖 AI Water Quality Forecasting Engine")
    st.markdown(
        "Multivariate recursive time-series forecasting utilizing historical observations, "
        "cyclical calendar harmonics, and validated ML regressors."
    )
    
    stations_list = current_df["station_name"].unique().tolist() if "station_name" in current_df.columns else ["Default Station"]
    
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        fc_station = st.selectbox("Forecast Monitoring Station", stations_list, index=3 if len(stations_list) > 3 else 0)
    with fcol2:
        fc_params = [c for c in ["dissolved_oxygen", "bod", "ph", "turbidity", "temperature", "conductivity"] if c in current_df.columns]
        param_display = {
            "dissolved_oxygen": "Dissolved Oxygen (DO)",
            "bod": "Biochemical Oxygen Demand (BOD)",
            "ph": "pH Level",
            "turbidity": "Turbidity",
            "temperature": "Temperature",
            "conductivity": "Conductivity"
        }
        fc_param = st.selectbox("Target Parameter", fc_params, format_func=lambda x: param_display.get(x, x))
    with fcol3:
        horizon = st.selectbox("Forecast Horizon", [1, 7, 30], index=1, format_func=lambda x: f"{x} Days Ahead")
        
    station_data = current_df[current_df["station_name"] == fc_station] if "station_name" in current_df.columns else current_df
    
    # Load model info
    bundle = load_model_bundle(fc_param)
    if bundle is None:
        st.warning(f"No pre-trained model bundle found for '{fc_param}'. Please train models in the backend.")
    else:
        best_mod_name = bundle["model_name"]
        metrics = bundle.get("metrics", {})
        
        # Display model stats banner
        st.markdown(f"""
        <div style="background-color:#F1F5F9; border-radius:8px; padding:12px 18px; margin-bottom:18px;">
            <b>Selected Model:</b> <span style="color:#1E88E5;font-weight:600;">{best_mod_name}</span> | 
            <b>Test RMSE:</b> {metrics.get('RMSE', 'N/A')} | 
            <b>MAE:</b> {metrics.get('MAE', 'N/A')} | 
            <b>R² Score:</b> {metrics.get('R2', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
        
    if st.button("🚀 Generate AI Forecast", type="primary"):
        with st.spinner("Generating recursive multi-step time-series forecast..."):
            try:
                forecast_df = generate_forecast(station_data, target_param=fc_param, horizon_days=horizon)
                
                # Plot Forecast Chart
                fig_fc = plot_forecast_chart(station_data, forecast_df, parameter=fc_param, station_name=fc_station)
                st.plotly_chart(fig_fc, use_container_width=True)
                
                # Show forecast table
                st.markdown("#### 📋 Forecasted Values & 95% Confidence Intervals")
                disp_fc = forecast_df.copy()
                disp_fc["date"] = disp_fc["date"].dt.strftime("%Y-%m-%d")
                disp_fc = disp_fc.rename(columns={
                    "date": "Date",
                    "forecast": f"Forecast ({fc_param})",
                    "lower_ci_95": "Lower Bound (95% CI)",
                    "upper_ci_95": "Upper Bound (95% CI)"
                })
                
                tcol1, tcol2 = st.columns([3, 1])
                with tcol1:
                    display_table(disp_fc)
                with tcol2:
                    csv = disp_fc.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv,
                        file_name=f"{fc_station}_{fc_param}_{horizon}d_forecast.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error during forecasting: {str(e)}")
                
    # Model Comparison Leaderboard Table
    st.markdown("---")
    st.markdown("### 🏆 Algorithm Performance Comparison")
    report_file = REPORTS_DIR / "model_performance_report.csv"
    if report_file.exists():
        comp_df = pd.read_csv(report_file)
        param_comp = comp_df[comp_df["Parameter"] == fc_param] if "Parameter" in comp_df.columns else comp_df
        display_table(param_comp)
    else:
        st.info("Run model training suite to generate the comparative performance leaderboard.")


# ==========================================
# PAGE 4: POLLUTION RISK ANALYSIS & DSS
# ==========================================
elif page == "⚠️ Pollution Risk Analysis":
    st.markdown("## ⚠️ Pollution Risk Analysis & Decision Support System (DSS)")
    st.markdown(
        "Automated water quality indexing, hazard anomaly detection, and "
        "interventional action matrix for municipal and pollution control authorities."
    )
    
    stations_list = current_df["station_name"].unique().tolist() if "station_name" in current_df.columns else ["Default Station"]
    risk_station = st.selectbox("Select Station for Risk Assessment", stations_list, index=3 if len(stations_list) > 3 else 0)
    
    st_data = current_df[current_df["station_name"] == risk_station] if "station_name" in current_df.columns else current_df
    last_record = st_data.iloc[-1]
    
    # Calculate WQI and anomalies
    wqi, sub_indices = calculate_water_quality_index(last_record)
    risk_info = classify_risk(wqi)
    anomalies = detect_parameter_anomalies(last_record)
    dss = generate_dss_recommendations(wqi, anomalies, risk_station)
    
    rcol1, rcol2 = st.columns([2, 3])
    
    with rcol1:
        st.plotly_chart(plot_wqi_gauge(wqi, risk_station), use_container_width=True)
        st.markdown(f"""
        <div style="background-color:white; border-radius:8px; padding:16px; border-left:6px solid {risk_info['color']}; box-shadow:0 2px 6px rgba(0,0,0,0.06);">
            <h3 style="color:{risk_info['color']}; margin:0;">{risk_info['badge']}</h3>
            <p style="font-weight:600; color:#334155; margin:6px 0;">CPCB Equivalent: {risk_info['cpcb_class']}</p>
            <p style="color:#64748B; font-size:13px; margin:0;">{risk_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with rcol2:
        st.markdown("#### 🚨 Identified Hazards & Anomalies")
        if not anomalies:
            st.markdown("""
            <div class="alert-card-success">
                <b>🟢 NO CRITICAL HAZARDS DETECTED</b><br>
                All physicochemical indicators comply with designated CPCB bathing and conservation thresholds.
            </div>
            """, unsafe_allow_html=True)
        else:
            for a in anomalies:
                sev_cls = "alert-card-critical" if a["severity"] == "Critical" else "alert-card-warning"
                st.markdown(f"""
                <div class="{sev_cls}">
                    <b>[{a['severity'].upper()}] {a['parameter']}</b><br>
                    {a['message']}
                </div>
                """, unsafe_allow_html=True)
                
        # Sub-Index Contribution
        st.markdown("#### 📊 Parameter Sub-Index Pollution Scores (qᵢ)")
        sub_df = pd.DataFrame([{"Parameter": k.replace("_", " ").title(), "Sub-Index (qᵢ)": v} for k, v in sub_indices.items()])
        display_table(sub_df)
        
    st.markdown("---")
    st.markdown("### 🏛️ Decision Support System (DSS) Action Matrix")
    
    tab_stp, tab_spcb, tab_basin, tab_public = st.tabs([
        "🏭 Municipal STPs",
        "⚖️ Pollution Control Board (SPCB)",
        "🌊 River Basin & Irrigation",
        "👥 Community Advisory"
    ])
    
    with tab_stp:
        st.markdown("##### Recommended Actions for Sewage & Effluent Treatment Plants:")
        for action in dss["stp_actions"]:
            st.markdown(f"• **{action}**")
            
    with tab_spcb:
        st.markdown("##### Regulatory & Enforcement Directives for SPCB / CPCB:")
        for action in dss["spcb_actions"]:
            st.markdown(f"• **{action}**")
            
    with tab_basin:
        st.markdown("##### Environmental Flow & Hydrological Interventions:")
        for action in dss["basin_actions"]:
            st.markdown(f"• **{action}**")
            
    with tab_public:
        st.markdown("##### Public Ghats & Municipal Water Supply Advisories:")
        for action in dss["public_actions"]:
            st.markdown(f"• **{action}**")
            
    # Radar chart comparison
    st.markdown("---")
    st.markdown("### 🎯 Water Quality Profile Benchmark Radar")
    fig_radar = plot_radar_profile(current_df, risk_station)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown(f'<div class="disclaimer-card">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ==========================================
# PAGE 5: HISTORICAL DATA ANALYSIS & EDA
# ==========================================
elif page == "📈 Historical Data Analysis":
    st.markdown("## 📈 Historical Data Analysis & Exploratory Data Science (EDA)")
    st.markdown("Statistical distributions, parameter correlations, and seasonal limnological cycles.")
    
    tab_stats, tab_corr, tab_season, tab_radar = st.tabs([
        "📊 Statistical Summary",
        "🔥 Correlation Matrix",
        "🌦️ Seasonal Dynamics",
        "🌐 Multi-Station Radar"
    ])
    
    with tab_stats:
        st.markdown("#### Comprehensive Descriptive Statistics")
        stats_df = compute_statistical_summary(current_df)
        display_table(stats_df)
        
    with tab_corr:
        st.markdown("#### Parameter Correlation Analysis")
        st.markdown(
            "Observe the fundamental limnological inverse relationship between Water Temperature and Dissolved Oxygen "
            "(Henry's Law of gas solubility), as well as the positive coupling between BOD and COD."
        )
        fig_corr = plot_correlation_heatmap(current_df)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with tab_season:
        st.markdown("#### Seasonal Variation across Winter, Summer, Monsoon, and Post-Monsoon")
        p_season = st.selectbox(
            "Select Parameter to Inspect Seasonal Spread",
            ["temperature", "dissolved_oxygen", "bod", "turbidity", "conductivity", "ph"]
        )
        fig_season = plot_seasonal_distribution(current_df, p_season)
        st.plotly_chart(fig_season, use_container_width=True)
        
    with tab_radar:
        st.markdown("#### Station Profile Comparison against CPCB Class A Benchmark")
        st_choice = st.selectbox("Select Station for Radar Analysis", current_df["station_name"].unique().tolist())
        fig_r = plot_radar_profile(current_df, st_choice)
        st.plotly_chart(fig_r, use_container_width=True)


# ==========================================
# PAGE 6: DATASET MANAGEMENT & UPLOAD
# ==========================================
elif page == "📁 Dataset Upload":
    st.markdown("## 📁 Dataset Management & Custom Upload")
    st.markdown(
        "Upload new water quality data dumps from CPCB, India-WRIS, or regional sensor networks. "
        "The system validates schemas, handles missing values, and dynamically enables forecasting."
    )
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"]
    )
    
    if uploaded_file is not None:
        is_valid, msg, validated_df = validate_uploaded_file(uploaded_file)
        if is_valid and validated_df is not None:
            st.success(f"✅ {msg}")
            
            summary = get_dataset_summary(validated_df)
            ucol1, ucol2, ucol3 = st.columns(3)
            with ucol1:
                st.metric("Total Records", f"{summary['total_records']:,}")
            with ucol2:
                st.metric("Stations Identified", summary['total_stations'])
            with ucol3:
                st.metric("Date Span", f"{summary['date_range'][0]} to {summary['date_range'][1]}")
                
            st.markdown("#### 🔍 Dataset Preview (Cleaned & Standardized)")
            display_table(validated_df.head(10))
            
            if st.button("📥 Apply & Load this Dataset into Active Session", type="primary"):
                cleaned_up = handle_missing_and_outliers(validated_df)
                st.session_state["water_data"] = cleaned_up
                st.session_state["is_custom_data"] = True
                st.success("Custom dataset successfully loaded into active session! Navigate to any dashboard page to analyze.")
                st.rerun()
        else:
            st.error(f"❌ Validation Error: {msg}")
            
    st.markdown("---")
    st.markdown("#### ℹ️ Current Active Dataset Information")
    curr_summary = get_dataset_summary(current_df)
    st.json({
        "Total Observations": curr_summary["total_records"],
        "Stations Count": curr_summary["total_stations"],
        "Stations": curr_summary["stations"],
        "Date Range": curr_summary["date_range"],
        "Is Demonstration Data": curr_summary["is_demonstration_data"]
    })


# ==========================================
# PAGE 7: ABOUT PROJECT
# ==========================================
elif page == "ℹ️ About Project":
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ Project Overview & Documentation</h1>
        <p>AI Water Quality Forecasting System | Smart India Hackathon (SIH1694) | College Group Project</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 📌 Project Metadata
    - **Project Title**: AI Water Quality Forecasting System
    - **Project ID**: P18
    - **Group ID**: OJ18
    - **Mapped Problem Statement**: **SIH1694** – *Real-time Ganga River Water Quality Forecasting using AI-enabled DSS, satellite data, IoT, and dynamic models.*
    - **Target River Basin**: National River Ganga (spanning Uttarakhand, Uttar Pradesh, Bihar, and West Bengal).

    ---

    ### 🎯 Problem Statement & Motivation
    River water quality degrades dynamically due to municipal sewage discharges, industrial effluents, and seasonal runoff. 
    Traditional laboratory sample testing often incurs multi-day turnaround delays, rendering post-hoc interventions ineffective.
    
    This project addresses the critical need for **proactive forecasting** by combining historical Central Pollution Control Board (CPCB) 
    water quality monitoring parameters with **multivariate Machine Learning algorithms** to forecast Dissolved Oxygen (DO), 
    Biochemical Oxygen Demand (BOD), Turbidity, pH, and Conductivity 1-day, 7-days, and 30-days in advance.

    ---

    ### 🔬 Machine Learning Architecture
    1. **Time-Series Feature Engineering**:
       - Chronological cyclical harmonics (sin/cos of day of year and month) to capture natural seasonal limnological variation.
       - Lag observations (lag_1, lag_7, lag_14, lag_30) without lookahead leakage.
       - Past-only rolling window averages and volatility metrics (7-day and 30-day).
    2. **Model Benchmarking**:
       - Linear Regression (Baseline)
       - Ridge Regression (L2 Regularization)
       - Random Forest Regressor (Non-linear Bagging)
       - Gradient Boosting Regressor (Sequential Residual Minimization)
    3. **Evaluation Protocol**:
       - Chronological train-test splitting (no random shuffling).
       - Quantitative metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R²).
    4. **Water Quality Index (WQI)**:
       - Weighted Arithmetic Index method adhering to CPCB / National Sanitation Foundation (NSF) standards.

    ---

    ### 📜 CPCB Designated Best Use Water Quality Criteria
    | Class | Designated Best Use | Criteria / Thresholds |
    | :---: | :--- | :--- |
    | **Class A** | Drinking water source without conventional treatment (disinfection only) | DO ≥ 6.0 mg/L, BOD ≤ 2.0 mg/L, pH 6.5 – 8.5 |
    | **Class B** | Outdoor bathing (organized) | DO ≥ 5.0 mg/L, BOD ≤ 3.0 mg/L, pH 6.5 – 8.5 |
    | **Class C** | Drinking water source after conventional treatment & disinfection | DO ≥ 4.0 mg/L, BOD ≤ 3.0 mg/L, pH 6.0 – 9.0 |
    | **Class D** | Propagation of wildlife and fisheries | DO ≥ 4.0 mg/L, Free Ammonia ≤ 1.2 mg/L |
    | **Class E** | Irrigation, industrial cooling, controlled waste disposal | Electrical Conductivity ≤ 2250 µS/cm |
    """)
    
    st.markdown(f'<div class="disclaimer-card">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)
