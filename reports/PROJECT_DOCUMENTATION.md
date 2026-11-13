# AI Water Quality Forecasting System
## Comprehensive College Project Report & Documentation
**Academic Year**: 2026 – 2027  
**Project ID**: P18 | **Group ID**: OJ18  
**Mapped Problem Statement**: SIH1694 – *Real-time Ganga River Water Quality Forecasting using AI-enabled DSS, satellite data, IoT, and dynamic models.*  
**Domain**: Artificial Intelligence, Machine Learning, Environmental Data Science, Decision Support Systems  

---

## Executive Summary / Abstract

The Ganga River basin sustains over 500 million people and diverse aquatic ecosystems across Northern and Eastern India. However, seasonal agricultural runoff, untreated municipal sewage, and industrial discharges cause severe, rapid fluctuations in key water quality parameters including Dissolved Oxygen (DO), Biochemical Oxygen Demand (BOD), Turbidity, pH, and Electrical Conductivity. Traditional water quality monitoring relies on physical laboratory testing that requires 3 to 5 days for chemical and biological incubation (especially 5-day BOD assays), creating a critical intervention lag where regulatory authorities and treatment plants are unable to respond to imminent contamination events.

This project designs and implements an end-to-end **AI-powered Water Quality Forecasting and Decision Support System (DSS)**. Using historical multi-station monitoring data along River Ganga (spanning Rishikesh, Haridwar, Kannauj, Kanpur, Prayagraj, Varanasi, Buxar, and Patna), the system applies rigorous time-series feature engineering—including cyclical calendar harmonics and leak-free past lag/rolling window statistics—to train and compare machine learning regressors (Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting). The models achieve high predictive accuracy ($R^2 > 0.93$ for DO and BOD; $R^2 > 0.98$ for Temperature) and support 1-day, 7-day, and 30-day recursive forecasts with 95% confidence intervals. Integrated with a CPCB/NSF Weighted Arithmetic Water Quality Index (WQI) and multi-tier hazard detection engine, the system delivers real-time prescriptive advisories for municipal wastewater plants, pollution regulators, and river basin authorities through an interactive Streamlit web application.

---

## 1. Introduction & Background

### 1.1 Problem Statement
Water quality monitoring without forecasting is inherently reactive. When downstream fish mortality or severe hypoxia occurs, damage to river ecology has already manifested. Proactive forecasting allows:
1. Municipal Sewage Treatment Plants (STPs) to dynamically modulate aeration cycles and chemical dosage before shock pollutant loads reach intakes.
2. State Pollution Control Boards (SPCBs) to deploy inspections and enforce Zero Liquid Discharge (ZLD) protocols.
3. Irrigation and Barrage Authorities (e.g., Narora, Bhimghoda) to regulate Environmental Flow (e-flow) releases to augment natural assimilation capacity.

### 1.2 Mapped Smart India Hackathon (SIH) Problem Statement: SIH1694
- **Title**: Real-time Ganga River Water Quality Forecasting using AI-enabled DSS, satellite data, IoT, and dynamic models.
- **Objective**: Develop an artificial intelligence-driven predictive and decision support system capable of analyzing multivariate water quality parameters, forecasting future degradation, detecting hazards, and offering actionable intervention strategies.

---

## 2. Limnological Science & Physicochemical Dynamics

The system incorporates authentic physical and biochemical relationships governing natural river systems:

1. **Henry's Law & Dissolved Oxygen (DO)**:
   Gas solubility is inversely proportional to water temperature. During warm summer months ($> 30^\circ\text{C}$), saturation DO drops naturally, reducing the buffer against organic pollutants.
2. **Biochemical Oxygen Demand (BOD) & Chemical Oxygen Demand (COD)**:
   BOD measures the oxygen required by aerobic microbes to oxidize organic matter. A spike in BOD leads to immediate microbiological oxygen consumption, causing downstream DO depression (oxygen sag curve).
3. **Monsoonal Runoff & Turbidity Dynamics**:
   During the Indian Southwest Monsoon (July – September), extreme surface runoff increases river siltation by over an order of magnitude (from 15 NTU up to $400+\text{ NTU}$), impairing photosynthetic light penetration.
4. **Electrical Conductivity (EC)**:
   Reflects total dissolved solids and ionic mineralization. Spikes in dry seasons due to evaporation and concentrated industrial effluent discharges.

---

## 3. System Architecture & Methodology

```
Ingestion & Data Loading (CSV / Excel)
            │
            ▼
Preprocessing & Feature Engineering (Lags, Rolling Means, Calendar Harmonics)
            │
            ▼
Chronological Train/Test Split (80% Train, 20% Test)
            │
            ▼
Machine Learning Training & Selection (Linear, Ridge, Random Forest, Gradient Boosting)
            │
            ▼
Multi-Horizon Recursive Forecaster (1, 7, 30 Days Ahead + 95% CI Bounds)
            │
            ▼
Pollution Risk Engine & WQI Calculator (Weighted Arithmetic Index)
            │
            ▼
AI Decision Support System (DSS) Action Matrix & Streamlit Web Portal
```

---

## 4. Machine Learning Implementation & Evaluation Results

### 4.1 Feature Engineering Protocol
- **Calendar Harmonics**: $\sin(2\pi \cdot \text{day} / 365.25)$, $\cos(2\pi \cdot \text{day} / 365.25)$, $\sin(2\pi \cdot \text{month} / 12)$, $\cos(2\pi \cdot \text{month} / 12)$
- **Lag Variables**: $\text{lag}_1$, $\text{lag}_7$, $\text{lag}_{14}$, $\text{lag}_{30}$ (strictly past values)
- **Rolling Statistics**: 7-day and 30-day moving averages and standard deviations using `shift(1)` to eliminate lookahead leakage.
- **Leak-Free Scaling**: `StandardScaler` fitted strictly on historical training data.

### 4.2 Benchmark Model Comparison Table

| Parameter | Model Algorithm | Test MAE | Test RMSE | Test $R^2$ Score | Selection |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Dissolved Oxygen (mg/L)** | Linear Regression | 0.316 | 0.408 | 0.932 | |
| | Ridge Regression | 0.316 | 0.408 | 0.932 | |
| | Random Forest | 0.303 | 0.389 | 0.938 | |
| | **Gradient Boosting** | **0.302** | **0.388** | **0.938** | 🏆 **Best Model** |
| **Biochemical Oxygen Demand (mg/L)** | Linear Regression | 0.385 | 0.499 | 0.944 | |
| | Ridge Regression | 0.385 | 0.499 | 0.944 | |
| | **Random Forest** | **0.372** | **0.487** | **0.947** | 🏆 **Best Model** |
| | Gradient Boosting | 0.373 | 0.491 | 0.946 | |
| **Water Temperature (°C)** | Linear Regression | 0.646 | 0.803 | 0.987 | |
| | **Ridge Regression** | **0.647** | **0.802** | **0.987** | 🏆 **Best Model** |
| | Random Forest | 0.669 | 0.827 | 0.986 | |
| | Gradient Boosting | 0.667 | 0.824 | 0.986 | |
| **Turbidity (NTU)** | Linear Regression | 22.373 | 42.761 | 0.873 | |
| | Ridge Regression | 22.380 | 42.758 | 0.873 | |
| | **Random Forest** | **21.151** | **41.914** | **0.878** | 🏆 **Best Model** |
| | Gradient Boosting | 21.257 | 42.168 | 0.876 | |
| **Electrical Conductivity (µS/cm)** | Linear Regression | 38.410 | 52.256 | 0.872 | |
| | Ridge Regression | 38.406 | 52.253 | 0.872 | |
| | Random Forest | 37.831 | 51.860 | 0.874 | |
| | **Gradient Boosting** | **37.661** | **51.558** | **0.875** | 🏆 **Best Model** |
| **Water pH** | **Linear Regression** | **0.175** | **0.226** | **0.293** | 🏆 **Best Model** |

---

## 5. Water Quality Index (WQI) & Decision Support System (DSS)

### 5.1 Formulation
Using the Weighted Arithmetic Water Quality Index:

$$\text{WQI} = \frac{\sum (q_i \cdot W_i)}{\sum W_i}$$

### 5.2 Four-Tier Risk Classification
1. **Low Risk ($\text{WQI} \le 35$)**: CPCB Class A/B. Suitable for organized outdoor bathing and drinking water after basic disinfection.
2. **Moderate Risk ($35 < \text{WQI} \le 65$)**: CPCB Class C. Requires full conventional treatment and disinfection; bathing with caution.
3. **High Risk ($65 < \text{WQI} \le 85$)**: CPCB Class D. Fish and wildlife propagation only. Direct human contact prohibited.
4. **Critical Risk ($\text{WQI} > 85$)**: CPCB Class E & Below. Hypoxia hazard ($\text{DO} < 3.5\text{ mg/L}$) or toxic organic loading ($\text{BOD} > 6.0\text{ mg/L}$). Severe ecological danger.

### 5.3 Multi-Stakeholder Action Matrix
- **Sewage Treatment Plants (STPs)**: Activate emergency dissolved air flotation and variable frequency aeration drives to offset forecasted DO dips.
- **State Pollution Control Boards (SPCBs)**: Deploy rapid inspection teams to industrial clusters (e.g. Jajmau tanneries, Unnao industrial zone).
- **Irrigation & Water Resources**: Authorize pulse release of ecological discharge (e-flow) from upstream barrages to dilute downstream concentration.
- **Public & District Administration**: Display bathing ghat safety warnings and adjust municipal intake pre-sedimentation tanks.

---

## 6. Software Architecture & Technology Stack

- **Core Language**: Python 3.14
- **Data Engineering**: Pandas 2.3.3, NumPy 2.5.4, SciPy 1.18.1
- **Machine Learning**: Scikit-learn 1.7.2 (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- **Model Persistence**: Joblib 1.6.0
- **Interactive Visualization**: Plotly 7.1.0
- **Frontend Dashboard**: Streamlit 1.67.0
- **File Ingestion**: OpenPyXL 3.1.5 (CSV and Excel spreadsheets)

---

## 7. Viva Voce & Presentation FAQ

**Q1: Why not use random train-test splitting for time-series data?**  
*Answer*: Random splitting introduces catastrophic lookahead bias (data leakage), where the model trains on future timestamps to predict past events, artificially inflating $R^2$ while failing completely on real-world forward predictions. We enforce a strict chronological split (first 80% dates for training, last 20% for testing).

**Q2: Why use cyclical sine and cosine harmonics for dates?**  
*Answer*: Day 365 (Dec 31) and Day 1 (Jan 1) are numerically distant (difference of 364) but temporally adjacent (1 day apart). By projecting day-of-year onto the unit circle ($\sin(2\pi t / 365)$ and $\cos(2\pi t / 365)$), the models perceive smooth, continuous seasonal cycles without artificial boundary discontinuities.

**Q3: How are multi-step forecasts generated?**  
*Answer*: The system utilizes recursive autoregressive rollout. To forecast $t+1$, historical lags are used. To forecast $t+2$, the predicted value at $t+1$ is rolled into the feature vector as $\text{lag}_1$, and updated calendar harmonics are calculated, iteratively predicting up to the chosen horizon (1, 7, or 30 days).

---

## 8. Statutory Disclaimer

*DISCLAIMER: This AI Water Quality Forecasting Decision Support System (DSS) provides predictive estimates and scientific advisory based on historical sensor data, mathematical modeling, and machine learning. It is built for academic, research, and prototype decision-support demonstration purposes and does NOT constitute an official government water safety certification or statutory certification by the Central Pollution Control Board (CPCB) or Ministry of Jal Shakti.*
