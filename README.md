# 🌊 AI Water Quality Forecasting System
### Real-Time Decision Support System for Ganga River Water Quality Monitoring & Forecasting

**Project ID**: P18  
**Group ID**: OJ18  
**Mapped Problem Statement**: **SIH1694** – *Real-time Ganga River Water Quality Forecasting using AI-enabled DSS, satellite data, IoT, and dynamic models.*  
**Domain**: AI / Machine Learning, Time-Series Forecasting, Environmental Data Science, Hydrological Decision Support Systems (DSS)

---

## 🌟 Overview

The **AI Water Quality Forecasting System** is a full-stack, multivariate AI-powered platform developed to support proactive environmental interventions across river basins. Rather than waiting for laboratory water sample incubation (which typically requires 3 to 5 days), this system applies machine learning models (Linear Regression, Ridge, Random Forest, and Gradient Boosting) to historical water monitoring data to forecast:
- **Dissolved Oxygen (DO in mg/L)**
- **Biochemical Oxygen Demand (BOD in mg/L)**
- **Water pH Level**
- **Nephelometric Turbidity Units (NTU)**
- **Water Temperature (°C)**
- **Electrical Conductivity (µS/cm)**

The system delivers **1-Day**, **7-Day**, and **30-Day** forecasts with **95% Confidence Intervals**, calculates a standard **Weighted Arithmetic Water Quality Index (WQI)**, categorizes river stretches into 4 risk tiers (Low, Moderate, High, Critical), detects hazards (e.g. Hypoxia, organic discharge spikes), and provides prescriptive action directives for Sewage Treatment Plants (STPs), State Pollution Control Boards (SPCBs), and river basin authorities.

---

## 🎯 Key System Features

1. **Executive Dashboard (`Home Dashboard`)**:
   - Basin-wide live KPI metrics cards (Average DO, BOD, pH, Turbidity, Overall WQI status).
   - Interactive OpenStreetMap view of Ganga monitoring stations (Rishikesh, Haridwar, Kannauj, Kanpur, Prayagraj, Varanasi, Buxar, Patna).
   - Real-time priority pollution alert feed.
2. **Station-Wise Monitoring (`Water Quality Monitoring`)**:
   - Detailed station metadata and physicochemical sensor gauges.
   - Interactive multi-year time-series plots with 30-day smoothed moving trends and CPCB threshold lines.
   - Cross-station upstream-to-downstream gradient bar charts.
3. **AI Forecasting Engine (`AI Forecasting`)**:
   - Horizon selector: 1-Day (immediate), 7-Day (weekly operational), 30-Day (strategic).
   - Interactive Plotly trajectory charts displaying historical observations, forecasted points, and shaded 95% confidence bounds.
   - Model accuracy card displaying test RMSE, MAE, and $R^2$ scores.
   - Downloadable CSV forecast reports.
   - Model leaderboard comparing all 4 algorithms.
4. **Pollution Risk & Decision Support System (`Pollution Risk Analysis`)**:
   - Dynamic WQI speedometer gauge.
   - 4-Tier CPCB-aligned risk classification (Class A/B, Class C, Class D, Class E).
   - Multi-stakeholder DSS Action Matrix with targeted interventions for STPs, regulators, irrigation engineers, and public ghats.
   - Benchmark radar chart comparing station water against ideal Class A criteria.
5. **Historical Data Science & EDA (`Historical Data Analysis`)**:
   - Parametric and non-parametric statistical summaries (Mean, Std, Median, IQR, Skewness).
   - Spearman/Pearson correlation heatmap capturing Henry's Law (inverse DO-Temperature) and organic BOD-COD relationships.
   - Seasonal boxplots comparing Winter, Summer, Monsoon, and Post-Monsoon shifts.
6. **Dataset Management & Custom Upload (`Dataset Upload`)**:
   - Ingestion of custom CSV and Excel (`.xlsx`, `.xls`) files.
   - Automated schema harmonization, date parsing, missing value handling, and physical boundary clamping.
   - Live session retraining and forecasting on newly uploaded river datasets.
7. **Comprehensive Documentation & Viva Prep (`About Project`)**:
   - SIH1694 project mapping, academic problem formulation, limnological foundations, and viva questions.

---

## 🏗️ Project Architecture & Directory Structure

```
AI_Water_Quality_Forecasting/
├── dataset/
│   ├── raw/
│   │   ├── cpcb_ganga_water_quality_demonstration_daily.csv  # 11,688 daily records (2021-2024 across 8 stations)
│   │   └── ganga_monitoring_stations.csv                     # CPCB stations with official coordinates
│   └── processed/
│       └── cleaned_ganga_water_quality.csv                   # Cleaned & boundary-clamped dataset
├── notebooks/
│   └── exploratory_data_analysis.ipynb                      # Jupyter notebook for viva & experimental verification
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Ingestion, schema standardization & upload validation
│   ├── preprocessing.py     # Missing imputation, bounds, cyclical harmonics, lag/rolling features
│   ├── eda.py               # Statistical computations & Plotly visual generators
│   ├── train_model.py       # Chronological split, ML model training & evaluation
│   ├── forecasting.py       # Multi-horizon recursive forecaster with 95% CI
│   ├── risk_detection.py    # Weighted arithmetic WQI, risk tiers, hazard alerts & DSS advice
│   └── generate_dataset.py  # Demonstration dataset generation pipeline
├── models/                  # Serialized Joblib model bundles (.joblib) for all parameters
├── reports/
│   ├── PROJECT_DOCUMENTATION.md # Full college project report & literature review
│   ├── ARCHITECTURE_DIAGRAM.md  # Mermaid system architecture & sequence diagrams
│   └── model_performance_report.csv # Consolidated model benchmark metrics
├── app.py                   # Complete 7-page interactive Streamlit web dashboard
├── requirements.txt         # Pinned tested Python package dependencies
└── README.md                # Project documentation & run guide
```

---

## ⚡ Quick Start & Execution Guide

### 1. Prerequisites
- Python 3.10+ (Tested and verified on Python 3.14.6 64-bit AMD64).
- Windows, macOS, or Linux.

### 2. Installation
Open PowerShell or your preferred terminal in the project directory:

```powershell
cd C:\Users\SPEED\.gemini\antigravity\scratch\AI_Water_Quality_Forecasting

# Install dependencies
python -m pip install -r requirements.txt
```

### 3. Run Machine Learning Training (Optional - Pre-trained models included)
To re-train all models and generate updated evaluation metrics:

```powershell
python src/train_model.py
```

### 4. Launch the Web Application
Launch the interactive Streamlit dashboard:

```powershell
streamlit run app.py
```

The web browser will automatically open at:
```
http://localhost:8501
```

---

## 📊 Model Evaluation Summary

All models were evaluated using chronological train-test splitting (80% historical training, 20% future test) to prevent time-series lookahead data leakage.

| Parameter | Selected Best Model | Test MAE | Test RMSE | Test $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Dissolved Oxygen (DO)** | **Gradient Boosting** | **0.302 mg/L** | **0.388 mg/L** | **0.938** |
| **Biochemical Oxygen Demand (BOD)** | **Random Forest** | **0.372 mg/L** | **0.487 mg/L** | **0.947** |
| **Water Temperature** | **Ridge Regression** | **0.647 °C** | **0.802 °C** | **0.987** |
| **Turbidity** | **Random Forest** | **21.151 NTU** | **41.914 NTU** | **0.878** |
| **Electrical Conductivity** | **Gradient Boosting** | **37.661 µS/cm** | **51.558 µS/cm** | **0.875** |
| **Water pH** | **Linear Regression** | **0.175** | **0.226** | **0.293** |

---

## 🧪 College Viva Voce Preparation Guide

1. **Why not use standard K-Fold cross validation?**  
   Standard K-Fold randomly shuffles data points, allowing models to train on observations from 2024 to predict 2022. This causes lookahead bias (data leakage). Time-series data must be split chronologically.
2. **What is Henry's Law in the context of water quality?**  
   Henry's Law states that the solubility of a gas in a liquid is proportional to its partial pressure above the liquid. Dissolved Oxygen solubility decreases as temperature rises, which is why cold upstream stations (Rishikesh) naturally hold more DO than warm downstream reaches (Patna).
3. **How does the Decision Support System generate recommendations?**  
   The DSS combines forecasted parameters with expert rule matrices based on CPCB designated best use criteria. When hypoxia ($\text{DO} < 4.0\text{ mg/L}$) or high organic load ($\text{BOD} > 5.0\text{ mg/L}$) is forecasted, the DSS automatically generates specific directives for municipal STPs, pollution control boards, and irrigation barrages.

---

## ⚖️ Statutory Disclaimer

> **DISCLAIMER**: This AI Water Quality Forecasting Decision Support System (DSS) provides predictive estimates and scientific advisory based on multivariate sensor and time-series modeling. It is intended for early warning and academic decision-support purposes and does NOT constitute an official government water safety certification or statutory guarantee by the Central Pollution Control Board (CPCB) or the Ministry of Jal Shakti.
