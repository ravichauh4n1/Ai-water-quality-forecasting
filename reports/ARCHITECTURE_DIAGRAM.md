# AI Water Quality Forecasting System - System Architecture
**Project ID**: P18 | **Group ID**: OJ18  
**Mapped SIH Problem Statement**: SIH1694 – Real-time Ganga River Water Quality Forecasting using AI-enabled DSS, satellite data, IoT, and dynamic models.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["1. Data Ingestion & Storage"]
        RawCSV["dataset/raw/<br>CPCB Ganga Monitoring Records (2021-2024)"]
        StationMeta["dataset/raw/<br>Monitoring Stations Registry (GPS Coordinates)"]
        UploadModule["Streamlit File Uploader<br>(CSV / Excel validation)"]
    end

    subgraph Preprocessing["2. Preprocessing & Feature Engineering Engine"]
        Impute["Missing Value Imputation<br>(Time-Series Forward/Backward Fill)"]
        Bounds["Domain Physics Clamping<br>(pH 0-14, DO > 0, Temp bounds)"]
        Cyclical["Calendar Harmonics<br>(sin/cos day-of-year, sin/cos month)"]
        Lags["Past Observation Lags<br>(lag_1, lag_7, lag_14, lag_30)"]
        Rolling["Past Rolling Statistics<br>(7-day & 30-day mean & volatility)"]
        Split["Strict Chronological Split<br>(80% Historical Train / 20% Recent Test)"]
        Scale["Leak-Free StandardScaler<br>(Fitted strictly on training partition)"]
    end

    subgraph Modeling["3. Machine Learning Modeling & Comparison Suite"]
        LR["Linear Regression (Baseline OLS)"]
        RidgeM["Ridge Regression (L2 Regularization)"]
        RF["Random Forest Regressor (Non-linear Bagging)"]
        GB["Gradient Boosting Regressor (Sequential Minimization)"]
        Eval["Model Evaluation (MAE, RMSE, R², MAPE)"]
        ModelStore["models/<br>Serialized Joblib Model Bundles (.joblib)"]
    end

    subgraph ForecastingEngine["4. Multi-Horizon Recursive Forecasting Engine"]
        Forecast1D["1-Day Ahead Horizon (Immediate Operational Dispatch)"]
        Forecast7D["7-Day Ahead Horizon (Weekly STP Planning)"]
        Forecast30D["30-Day Ahead Horizon (Monthly Basin Management)"]
        CI["95% Uncertainty Estimation Bounds (± 1.96 · σ_residual · √h)"]
    end

    subgraph DecisionSupport["5. Pollution Risk Assessment & AI DSS"]
        WQI["Weighted Arithmetic Water Quality Index (CPCB / NSF)"]
        Tiers["4-Tier Risk Categorization<br>(Low, Moderate, High, Critical)"]
        Anomaly["Physicochemical Hazard Detector<br>(Hypoxia, Elevated BOD, pH Shock, Turbidity Siltation)"]
        DSSRec["AI Interventions Engine<br>(STP aeration, SPCB audits, e-flow regulation, public alerts)"]
    end

    subgraph Presentation["6. Interactive Streamlit Web Portal (River-Blue Theme)"]
        P1["🏠 Home Dashboard (KPI Cards, Interactive Basin Map, Real-time Alerts)"]
        P2["📊 Water Quality Monitoring (Station Selector, Gauges, Multi-Year Trends)"]
        P3["🤖 AI Forecasting (Horizon Selector, Prediction Charts, Model Leaderboard)"]
        P4["⚠️ Pollution Risk Analysis (WQI Gauge, Hazard Cards, DSS Action Matrix)"]
        P5["📈 Historical EDA (Statistical Tables, Correlation Matrix, Seasonal Boxplots)"]
        P6["📁 Dataset Upload (Custom CSV/XLSX Ingestion & Schema Validator)"]
        P7["ℹ️ About Project (SIH1694 Mapping, Architecture, CPCB Standards, Viva Guide)"]
    end

    RawCSV --> Impute
    StationMeta --> Impute
    UploadModule --> Impute
    Impute --> Bounds --> Cyclical --> Lags --> Rolling --> Split --> Scale
    Scale --> LR & RidgeM & RF & GB
    LR & RidgeM & RF & GB --> Eval --> ModelStore
    ModelStore --> Forecast1D & Forecast7D & Forecast30D --> CI
    CI --> WQI --> Tiers --> Anomaly --> DSSRec
    DSSRec --> P1 & P2 & P3 & P4 & P5 & P6 & P7
```

---

## 2. Machine Learning Pipeline Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User as Student / Evaluator
    participant UI as Streamlit Web App (app.py)
    participant Loader as Data Loader (data_loader.py)
    participant Prep as Preprocessor (preprocessing.py)
    participant Model as ML Engine (train_model.py)
    participant Forecast as Forecaster (forecasting.py)
    participant DSS as Risk Engine (risk_detection.py)

    User->>UI: Select River Station & Horizon (e.g. Kanpur, 7 Days)
    UI->>Loader: Fetch historical data for station
    Loader->>Prep: Harmonize schema & impute gaps
    Prep->>Forecast: Build rolling lag vectors & cyclical harmonics
    Forecast->>Model: Query trained best model bundle (GradientBoost / RF)
    Model-->>Forecast: Predict t+1 through t+7 iteratively
    Forecast->>DSS: Pass multivariate predictions
    DSS->>DSS: Compute future WQI & detect hypoxia / BOD hazard
    DSS-->>UI: Return forecast values, 95% CI bounds & DSS recommendations
    UI-->>User: Render interactive Plotly charts, alerts & action matrix
```

---

## 3. Water Quality Index (WQI) Mathematical Formulation

The Water Quality Index is calculated using the standard **Weighted Arithmetic Index Method** recognized by CPCB and the National Sanitation Foundation (NSF):

$$\text{WQI} = \frac{\sum_{i=1}^{n} q_i \cdot W_i}{\sum_{i=1}^{n} W_i}$$

Where:
- $q_i$ is the quality rating or sub-index for parameter $i$:
  $$q_i = \left| \frac{V_{\text{actual}} - V_{\text{ideal}}}{S_{\text{standard}} - V_{\text{ideal}}} \right| \times 100$$
- $W_i$ is the unit weight assigned to parameter $i$ (Dissolved Oxygen: 0.35, BOD: 0.25, pH: 0.15, Turbidity: 0.10, Conductivity: 0.10, Temperature: 0.05).
- $S_{\text{standard}}$ is the CPCB permissible threshold.
- $V_{\text{ideal}}$ is the ideal benchmark value (e.g., $14.6\text{ mg/L}$ for DO, $0.0$ for BOD, $7.0$ for pH).

---

## 4. Risk Classification Tiers

| WQI Score Range | Risk Category | CPCB Water Quality Class | Permissible Uses |
| :---: | :---: | :---: | :--- |
| **0 – 35** | 🟢 **Low Risk** | Class A / B | Drinking water source (after disinfection), organized outdoor bathing |
| **36 – 65** | 🟡 **Moderate Risk** | Class C | Drinking source after conventional treatment & disinfection |
| **66 – 85** | 🟠 **High Risk** | Class D | Propagation of wildlife and fisheries only; no direct contact |
| **86 – 100+** | 🔴 **Critical Risk** | Class E & Below | Industrial cooling, controlled irrigation; severe hypoxia/pollution alert |
