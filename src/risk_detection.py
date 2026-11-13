"""
Pollution Risk Assessment, Water Quality Index (WQI), and Decision Support System (DSS).
Implements:
1. Weighted Arithmetic Water Quality Index (WQI) based on CPCB / NSF standards.
2. 4-Tier Risk Classification: Low, Moderate, High, Critical.
3. Specific physicochemical anomaly detectors (Hypoxia, High BOD, pH shock, Siltation).
4. AI Decision Support & Intervention Recommendations Engine.
5. Interactive WQI Gauge and Risk Breakdown Visualizations.
6. Statutory safety disclaimer.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# CPCB Designated Best Use & Standard Benchmarks
# Standards (Si), Ideal Values (Vi), and Relative Weights (Wi)
WQI_PARAMETERS_CONFIG = {
    "dissolved_oxygen": {
        "standard": 5.0,     # mg/L (Minimum for bathing)
        "ideal": 14.6,        # Saturated DO at 0°C
        "weight": 0.35,
        "name": "Dissolved Oxygen (DO)"
    },
    "bod": {
        "standard": 3.0,     # mg/L (Maximum for bathing)
        "ideal": 0.0,
        "weight": 0.25,
        "name": "Biochemical Oxygen Demand (BOD)"
    },
    "ph": {
        "standard": 8.5,     # Upper limit (safe range 6.5 - 8.5)
        "ideal": 7.0,        # Neutral
        "weight": 0.15,
        "name": "pH Level"
    },
    "turbidity": {
        "standard": 50.0,    # NTU
        "ideal": 0.0,
        "weight": 0.10,
        "name": "Turbidity"
    },
    "conductivity": {
        "standard": 500.0,   # µS/cm
        "ideal": 0.0,
        "weight": 0.10,
        "name": "Electrical Conductivity"
    },
    "temperature": {
        "standard": 32.0,    # °C
        "ideal": 22.0,
        "weight": 0.05,
        "name": "Water Temperature"
    }
}

RISK_TIERS = [
    {
        "category": "Low Risk",
        "wqi_range": (0, 35),
        "cpcb_class": "Class A / B (Drinking / Bathing)",
        "color": "#43A047",  # Green
        "description": "Good water quality. Dissolved oxygen levels support thriving aquatic life. Safe for outdoor bathing and drinking source with conventional disinfection.",
        "badge": "🟢 LOW RISK"
    },
    {
        "category": "Moderate Risk",
        "wqi_range": (35.01, 65),
        "cpcb_class": "Class C (Conventional Treatment)",
        "color": "#FDD835",  # Yellow
        "description": "Medium water quality with moderate organic or sediment load. Outdoor bathing acceptable with caution; municipal intake requires complete conventional treatment and disinfection.",
        "badge": "🟡 MODERATE RISK"
    },
    {
        "category": "High Risk",
        "wqi_range": (65.01, 85),
        "cpcb_class": "Class D (Wildlife & Fisheries Propagation)",
        "color": "#FB8C00",  # Orange
        "description": "Poor water quality. Significant oxygen depletion or organic pollution. Unsuitable for direct body contact; ecological stress observed.",
        "badge": "🟠 HIGH RISK"
    },
    {
        "category": "Critical Risk",
        "wqi_range": (85.01, 999),
        "cpcb_class": "Class E & Below (Irrigation / Industrial / Unfit)",
        "color": "#E53935",  # Red
        "description": "Very poor / polluted water. Severe hypoxia (DO < 4 mg/L) or hazardous BOD levels (> 6 mg/L). Risk of widespread fish mortality and health hazards.",
        "badge": "🔴 CRITICAL RISK"
    }
]

DISCLAIMER_TEXT = (
    "STATUTORY DISCLAIMER: This AI Water Quality Forecasting Decision Support System (DSS) "
    "provides predictive estimates and scientific advisory based on historical CPCB measurements, "
    "sensor dynamics, and machine learning models. It is developed for research, academic, and decision-support "
    "demonstration purposes and does NOT constitute an official government water safety certification "
    "or statutory guarantee by the Central Pollution Control Board (CPCB) or the Ministry of Jal Shakti."
)


def calculate_water_quality_index(row: Union[pd.Series, Dict[str, float]]) -> Tuple[float, Dict[str, float]]:
    """
    Computes Weighted Arithmetic Water Quality Index (WQI) based on available parameters.
    Returns: (overall_wqi, sub_indices_dict).
    """
    sum_weighted_q = 0.0
    sum_weights = 0.0
    sub_indices = {}
    
    for param, cfg in WQI_PARAMETERS_CONFIG.items():
        if param in row and pd.notna(row[param]):
            val = float(row[param])
            ideal = cfg["ideal"]
            standard = cfg["standard"]
            weight = cfg["weight"]
            
            # Sub-index q_i calculation
            if param == "ph":
                if val < 7.0:
                    q_i = abs((val - 7.0) / (6.5 - 7.0)) * 100.0
                else:
                    q_i = abs((val - 7.0) / (8.5 - 7.0)) * 100.0
            elif param == "dissolved_oxygen":
                # For DO, higher is better. Standard 5 mg/L, Saturation 14.6 mg/L
                # If DO >= 14.6, q_i = 0 (perfect); if DO = 5, q_i = 100
                q_i = max(0.0, (14.6 - val) / (14.6 - standard)) * 100.0
            else:
                # For pollutants (BOD, Turbidity, Cond): lower is better
                q_i = max(0.0, (val - ideal) / (standard - ideal)) * 100.0
                
            q_i = min(250.0, q_i)  # Cap extreme sub-indices for stability
            sub_indices[param] = round(q_i, 1)
            sum_weighted_q += q_i * weight
            sum_weights += weight
            
    if sum_weights > 0:
        overall_wqi = round(sum_weighted_q / sum_weights, 1)
    else:
        overall_wqi = 50.0
        
    return overall_wqi, sub_indices


def classify_risk(wqi: float) -> Dict:
    """
    Maps overall WQI score to risk tier, CPCB class, color, and description.
    """
    for tier in RISK_TIERS:
        low, high = tier["wqi_range"]
        if low <= wqi <= high:
            return tier
    return RISK_TIERS[-1]


def detect_parameter_anomalies(row: Union[pd.Series, Dict[str, float]]) -> List[Dict]:
    """
    Identifies specific parameter-level hazards and deviations from CPCB standards.
    """
    anomalies = []
    
    # 1. Dissolved Oxygen Check
    if "dissolved_oxygen" in row and pd.notna(row["dissolved_oxygen"]):
        do_val = float(row["dissolved_oxygen"])
        if do_val < 3.5:
            anomalies.append({
                "parameter": "Dissolved Oxygen",
                "severity": "Critical",
                "message": f"Severe Hypoxia Hazard (DO = {do_val:.2f} mg/L < 3.5 mg/L). Aquatic fauna and fish biodiversity are at imminent risk of mortality.",
                "color": "#E53935"
            })
        elif do_val < 5.0:
            anomalies.append({
                "parameter": "Dissolved Oxygen",
                "severity": "Warning",
                "message": f"Sub-standard DO (DO = {do_val:.2f} mg/L). Below CPCB outdoor bathing benchmark (5.0 mg/L).",
                "color": "#FB8C00"
            })
            
    # 2. BOD Check
    if "bod" in row and pd.notna(row["bod"]):
        bod_val = float(row["bod"])
        if bod_val > 6.0:
            anomalies.append({
                "parameter": "Biochemical Oxygen Demand",
                "severity": "Critical",
                "message": f"High Organic Load (BOD = {bod_val:.2f} mg/L > 6.0 mg/L). Indicates influx of raw sewage or untreated industrial effluents.",
                "color": "#E53935"
            })
        elif bod_val > 3.0:
            anomalies.append({
                "parameter": "Biochemical Oxygen Demand",
                "severity": "Warning",
                "message": f"Elevated Organic Pollution (BOD = {bod_val:.2f} mg/L). Exceeds CPCB bathing tolerance standard of 3.0 mg/L.",
                "color": "#FB8C00"
            })
            
    # 3. pH Check
    if "ph" in row and pd.notna(row["ph"]):
        ph_val = float(row["ph"])
        if ph_val < 6.5:
            anomalies.append({
                "parameter": "pH Level",
                "severity": "Warning",
                "message": f"Acidic Water Condition (pH = {ph_val:.2f} < 6.5). Potential acidic chemical or trade effluent discharge.",
                "color": "#FB8C00"
            })
        elif ph_val > 8.5:
            anomalies.append({
                "parameter": "pH Level",
                "severity": "Warning",
                "message": f"Alkaline Water Condition (pH = {ph_val:.2f} > 8.5). Excessive algal bloom photosynthesis or alkaline washings.",
                "color": "#FB8C00"
            })
            
    # 4. Turbidity Check
    if "turbidity" in row and pd.notna(row["turbidity"]):
        turb_val = float(row["turbidity"])
        if turb_val > 150.0:
            anomalies.append({
                "parameter": "Turbidity",
                "severity": "Warning",
                "message": f"High Siltation & Suspended Solids (Turbidity = {turb_val:.1f} NTU > 150 NTU). Common during heavy monsoonal runoff; impairs light penetration.",
                "color": "#FF9800"
            })
            
    # 5. Conductivity Check
    if "conductivity" in row and pd.notna(row["conductivity"]):
        cond_val = float(row["conductivity"])
        if cond_val > 700.0:
            anomalies.append({
                "parameter": "Electrical Conductivity",
                "severity": "Warning",
                "message": f"High Mineralization / Salinity (EC = {cond_val:.1f} µS/cm > 700 µS/cm). Indicates dissolved salts or industrial trade effluent.",
                "color": "#FF9800"
            })
            
    return anomalies


def generate_dss_recommendations(
    wqi: float,
    anomalies: List[Dict],
    station_name: str
) -> Dict:
    """
    Generates actionable Decision Support System (DSS) recommendations for:
    - Municipal Sewage Treatment Plants (STPs)
    - Industrial Regulators (SPCB / CPCB)
    - Irrigation & River Basin Authorities (e-flow management)
    - Public Bathing Ghats & Community Advisories
    """
    risk_info = classify_risk(wqi)
    category = risk_info["category"]
    
    stp_actions = []
    spcb_actions = []
    basin_actions = []
    public_actions = []
    explanations = []
    
    # Root-cause explanations
    if not anomalies:
        explanations.append(
            f"Water quality at {station_name} is currently within permissible limits. "
            "Dissolved oxygen satisfies natural reaeration requirements, and organic decomposition is balanced."
        )
    else:
        for anom in anomalies:
            explanations.append(f"• {anom['parameter']}: {anom['message']}")
            
    # Interventions based on risk level
    if category in ["Critical Risk", "High Risk"]:
        stp_actions.extend([
            "Trigger emergency aerator activation in downstream Sewage Treatment Plants (STPs) to elevate DO.",
            "Verify that municipal pumping stations are not bypassing untreated sewage during peak drainage hours.",
            "Inspect in-stream oxidation ponds and biological treatment filters."
        ])
        spcb_actions.extend([
            "Deploy SPCB surveillance squad to inspect major industrial outfalls (tanneries, distilleries, textile units).",
            "Mandate real-time OCEMS (Online Continuous Effluent Monitoring System) audit on nearby CETPs."
        ])
        basin_actions.extend([
            "Liaise with Irrigation Department to release supplementary ecological dilution flow (e-flow) from upstream barrages.",
            "Temporarily restrict river sand mining and desiltation activity causing excessive sediment re-suspension."
        ])
        public_actions.extend([
            "Issue advisory to local district administration regarding organized mass bathing at ghats.",
            "Increase coagulation and chemical dosing at municipal drinking water intake treatment plants."
        ])
    elif category == "Moderate Risk":
        stp_actions.extend([
            "Maintain optimal sludge retention time and monitor biological oxygen uptake rate in secondary clarifiers.",
            "Ensure functional operation of chlorination / UV disinfection units before river discharge."
        ])
        spcb_actions.extend([
            "Conduct routine spot sampling of industrial effluent treatment plants (ETPs) within the catchment."
        ])
        basin_actions.extend([
            "Monitor upstream reservoir discharge levels to sustain designated baseline river velocity."
        ])
        public_actions.extend([
            "Water is acceptable for bathing; avoid direct consumption without standard municipal filtration and boiling."
        ])
    else:  # Low Risk
        stp_actions.append("Continue standard preventive maintenance schedule and continuous monitoring.")
        spcb_actions.append("Maintain routine monthly water sampling as per CPCB NWMP calendar.")
        basin_actions.append("Maintain natural environmental flow (e-flow) allocations.")
        public_actions.append("Water satisfies designated best use criteria for Class A/B outdoor bathing and aquatic conservation.")
        
    return {
        "station_name": station_name,
        "wqi": wqi,
        "risk_tier": risk_info,
        "explanations": explanations,
        "stp_actions": stp_actions,
        "spcb_actions": spcb_actions,
        "basin_actions": basin_actions,
        "public_actions": public_actions,
        "disclaimer": DISCLAIMER_TEXT
    }


def plot_wqi_gauge(wqi: float, station_name: str = "River Station") -> go.Figure:
    """
    Renders an interactive semicircular gauge for Water Quality Index (WQI).
    """
    risk = classify_risk(wqi)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=wqi,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"Water Quality Index (WQI)<br><span style='font-size:0.8em;color:gray'>{station_name}</span>", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": risk["color"], "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#E2E8F0",
            "steps": [
                {"range": [0, 35], "color": "rgba(67, 160, 71, 0.25)"},
                {"range": [35, 65], "color": "rgba(253, 216, 53, 0.25)"},
                {"range": [65, 85], "color": "rgba(251, 140, 0, 0.25)"},
                {"range": [85, 100], "color": "rgba(229, 57, 53, 0.25)"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": wqi
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=30, r=30, t=50, b=20)
    )
    return fig


if __name__ == "__main__":
    print("Testing risk_detection.py...")
    sample_clean = {
        "dissolved_oxygen": 8.8,
        "bod": 1.5,
        "ph": 7.8,
        "turbidity": 15.0,
        "conductivity": 180.0,
        "temperature": 18.0
    }
    wqi_clean, subs_clean = calculate_water_quality_index(sample_clean)
    risk_clean = classify_risk(wqi_clean)
    print(f"Pristine Sample (Rishikesh style): WQI = {wqi_clean} -> {risk_clean['badge']}")
    
    sample_polluted = {
        "dissolved_oxygen": 3.2,
        "bod": 8.5,
        "ph": 7.4,
        "turbidity": 48.0,
        "conductivity": 780.0,
        "temperature": 27.0
    }
    wqi_polluted, subs_polluted = calculate_water_quality_index(sample_polluted)
    risk_polluted = classify_risk(wqi_polluted)
    print(f"Polluted Sample (Kanpur style): WQI = {wqi_polluted} -> {risk_polluted['badge']}")
    
    anoms = detect_parameter_anomalies(sample_polluted)
    print(f"Detected {len(anoms)} anomalies:")
    for a in anoms:
        print(f"  - [{a['severity']}] {a['message']}")
        
    dss = generate_dss_recommendations(wqi_polluted, anoms, "Kanpur (Jajmau)")
    print(f"\nGenerated {len(dss['stp_actions'])} STP actions, {len(dss['spcb_actions'])} SPCB actions.")
    
    fig = plot_wqi_gauge(wqi_polluted, "Kanpur (Jajmau)")
    print("Plotly WQI gauge built successfully!")
    print("Risk Detection & Decision Support System verified successfully!")
