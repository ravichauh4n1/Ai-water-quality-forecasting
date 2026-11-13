"""
Dataset Generation Script for CPCB Ganga River Water Quality Demonstration Dataset.
Generates 4 years (2021-2024) of daily multivariate measurements across 8 river stations.
Maintains authentic limnological relationships (Henry's Law for DO, seasonal monsoon
runoff for Turbidity, industrial organic loading for BOD/COD, and temperature cycles).

Clearly labelled with 'is_demonstration_data=True' to ensure complete transparency.
"""

import numpy as np
import pandas as pd
from datetime import datetime

def generate_ganga_dataset():
    np.random.seed(42)
    
    # Date range: 2021-01-01 to 2024-12-31 (4 full years = 1,461 days)
    dates = pd.date_range(start="2021-01-01", end="2024-12-31", freq="D")
    n_days = len(dates)
    
    # Station definitions with baseline parameters & coordinates
    stations = [
        {
            "station_code": "CPCB_G01",
            "station_name": "Rishikesh (Triveni Ghat)",
            "river": "Ganga",
            "state": "Uttarakhand",
            "lat": 30.1033, "lon": 78.2948,
            "base_temp": 18.0, "temp_amp": 6.5,
            "base_do": 9.2, "do_amp": 0.9,
            "base_bod": 1.4, "bod_noise": 0.25,
            "base_ph": 7.9, "ph_noise": 0.15,
            "base_turb": 12.0, "monsoon_turb": 95.0,
            "base_cond": 160.0, "cond_noise": 15.0,
            "base_fc": 320.0
        },
        {
            "station_code": "CPCB_G02",
            "station_name": "Haridwar (Har Ki Pauri)",
            "river": "Ganga",
            "state": "Uttarakhand",
            "lat": 29.9560, "lon": 78.1700,
            "base_temp": 19.5, "temp_amp": 7.0,
            "base_do": 8.4, "do_amp": 0.8,
            "base_bod": 2.2, "bod_noise": 0.4,
            "base_ph": 7.8, "ph_noise": 0.18,
            "base_turb": 18.0, "monsoon_turb": 160.0,
            "base_cond": 210.0, "cond_noise": 25.0,
            "base_fc": 950.0
        },
        {
            "station_code": "CPCB_G03",
            "station_name": "Kannauj (Upstream)",
            "river": "Ganga",
            "state": "Uttar Pradesh",
            "lat": 27.0540, "lon": 79.9200,
            "base_temp": 24.0, "temp_amp": 8.5,
            "base_do": 6.3, "do_amp": 0.9,
            "base_bod": 4.6, "bod_noise": 0.7,
            "base_ph": 7.7, "ph_noise": 0.22,
            "base_turb": 28.0, "monsoon_turb": 280.0,
            "base_cond": 380.0, "cond_noise": 45.0,
            "base_fc": 4200.0
        },
        {
            "station_code": "CPCB_G04",
            "station_name": "Kanpur (Jajmau)",
            "river": "Ganga",
            "state": "Uttar Pradesh",
            "lat": 26.4310, "lon": 80.3990,
            "base_temp": 25.5, "temp_amp": 9.0,
            "base_do": 4.5, "do_amp": 1.1,
            "base_bod": 7.8, "bod_noise": 1.4,
            "base_ph": 7.5, "ph_noise": 0.35,
            "base_turb": 42.0, "monsoon_turb": 340.0,
            "base_cond": 640.0, "cond_noise": 85.0,
            "base_fc": 38000.0
        },
        {
            "station_code": "CPCB_G05",
            "station_name": "Prayagraj (Sangam)",
            "river": "Ganga",
            "state": "Uttar Pradesh",
            "lat": 25.4299, "lon": 81.8845,
            "base_temp": 25.0, "temp_amp": 8.8,
            "base_do": 6.9, "do_amp": 0.85,
            "base_bod": 4.0, "bod_noise": 0.65,
            "base_ph": 7.8, "ph_noise": 0.2,
            "base_turb": 32.0, "monsoon_turb": 310.0,
            "base_cond": 420.0, "cond_noise": 50.0,
            "base_fc": 8500.0
        },
        {
            "station_code": "CPCB_G06",
            "station_name": "Varanasi (Assi Ghat)",
            "river": "Ganga",
            "state": "Uttar Pradesh",
            "lat": 25.2885, "lon": 83.0063,
            "base_temp": 25.8, "temp_amp": 8.6,
            "base_do": 6.1, "do_amp": 0.9,
            "base_bod": 5.4, "bod_noise": 0.9,
            "base_ph": 7.7, "ph_noise": 0.22,
            "base_turb": 35.0, "monsoon_turb": 330.0,
            "base_cond": 460.0, "cond_noise": 55.0,
            "base_fc": 24000.0
        },
        {
            "station_code": "CPCB_G07",
            "station_name": "Buxar (Ramrekha Ghat)",
            "river": "Ganga",
            "state": "Bihar",
            "lat": 25.5788, "lon": 83.9814,
            "base_temp": 26.0, "temp_amp": 8.2,
            "base_do": 6.7, "do_amp": 0.8,
            "base_bod": 3.6, "bod_noise": 0.55,
            "base_ph": 7.8, "ph_noise": 0.19,
            "base_turb": 38.0, "monsoon_turb": 360.0,
            "base_cond": 410.0, "cond_noise": 40.0,
            "base_fc": 6200.0
        },
        {
            "station_code": "CPCB_G08",
            "station_name": "Patna (Gandhi Ghat)",
            "river": "Ganga",
            "state": "Bihar",
            "lat": 25.6207, "lon": 85.1720,
            "base_temp": 26.5, "temp_amp": 8.0,
            "base_do": 6.3, "do_amp": 0.85,
            "base_bod": 4.4, "bod_noise": 0.75,
            "base_ph": 7.6, "ph_noise": 0.21,
            "base_turb": 44.0, "monsoon_turb": 410.0,
            "base_cond": 450.0, "cond_noise": 45.0,
            "base_fc": 16500.0
        }
    ]
    
    rows = []
    
    # Day of year array (1 to 365/366)
    day_of_year = dates.dayofyear.values
    month = dates.month.values
    
    for st in stations:
        # Seasonal cycle: phase aligned so minimum temp is mid-Jan (d ~ 15), max temp is end of May (d ~ 150)
        # Cosine with phase shift
        annual_phase = 2 * np.pi * (day_of_year - 150) / 365.25
        
        # 1. Temperature (°C)
        temp_cycle = st["base_temp"] + st["temp_amp"] * np.cos(annual_phase)
        temp_noise = np.random.normal(0, 0.8, n_days)
        temperature = np.round(np.clip(temp_cycle + temp_noise, 10.0, 36.0), 2)
        
        # 2. Dissolved Oxygen (mg/L): Henry's Law inverse relationship with Temperature
        # Higher temp = lower gas solubility; plus biological respiration in warm water
        do_cycle = st["base_do"] - st["do_amp"] * np.cos(annual_phase)
        do_noise = np.random.normal(0, 0.35, n_days)
        # Occasional episodic industrial discharge or pollution slump
        pollution_spikes = (np.random.rand(n_days) < 0.04) * np.random.uniform(0.6, 1.8, n_days)
        dissolved_oxygen = np.round(np.clip(do_cycle + do_noise - pollution_spikes, 1.5, 12.0), 2)
        
        # 3. Biochemical Oxygen Demand (BOD mg/L): increases during summer low-flow & festival bathing
        # In monsoon, dilution can slightly reduce concentration, but urban runoff carries organics
        bod_seasonal = 0.5 * np.cos(annual_phase) # Slightly higher in warm months
        bod_noise = np.random.normal(0, st["bod_noise"], n_days)
        bod_spikes = pollution_spikes * 1.8 # Correlated with DO drops
        bod = np.round(np.clip(st["base_bod"] + bod_seasonal + bod_noise + bod_spikes, 0.5, 20.0), 2)
        
        # 4. Chemical Oxygen Demand (COD mg/L): typically 2.2 to 3.2 times BOD
        cod_ratio = np.random.uniform(2.3, 3.1, n_days)
        cod = np.round(np.clip(bod * cod_ratio + np.random.normal(0, 1.2, n_days), 1.5, 60.0), 2)
        
        # 5. pH: Stable natural buffer ~7.4-8.4, with slight diurnal/seasonal photosynthetic drift
        ph_noise = np.random.normal(0, st["ph_noise"], n_days)
        ph = np.round(np.clip(st["base_ph"] + 0.15 * np.sin(annual_phase) + ph_noise, 6.2, 9.0), 2)
        
        # 6. Turbidity (NTU): Huge peak in Monsoon (July, August, September = months 7, 8, 9)
        # Using a Gaussian bump centered at day 225 (mid-August)
        monsoon_factor = np.exp(-0.5 * ((day_of_year - 225) / 32) ** 2)
        turb_base = st["base_turb"] + np.random.exponential(4.0, n_days)
        turb_monsoon = st["monsoon_turb"] * monsoon_factor * (1.0 + 0.3 * np.random.randn(n_days))
        turbidity = np.round(np.clip(turb_base + turb_monsoon, 2.0, 650.0), 1)
        
        # 7. Electrical Conductivity (µS/cm): Highest in summer pre-monsoon (concentration),
        # lowest during monsoon peak due to high rainwater dilution
        cond_seasonal = -0.22 * st["base_cond"] * monsoon_factor + 0.1 * st["base_cond"] * np.cos(annual_phase)
        cond_noise = np.random.normal(0, st["cond_noise"], n_days)
        conductivity = np.round(np.clip(st["base_cond"] + cond_seasonal + cond_noise, 80.0, 1200.0), 1)
        
        # 8. Fecal Coliform (MPN/100ml): Spikes during monsoon runoff and bathing events
        fc_factor = 1.0 + 2.5 * monsoon_factor + np.random.exponential(0.6, n_days)
        fecal_coliform = np.round(np.clip(st["base_fc"] * fc_factor, 50.0, 150000.0), 0)
        
        for i in range(n_days):
            rows.append({
                "date": dates[i].strftime("%Y-%m-%d"),
                "station_code": st["station_code"],
                "station_name": st["station_name"],
                "river": st["river"],
                "state": st["state"],
                "latitude": st["lat"],
                "longitude": st["lon"],
                "temperature": temperature[i],
                "ph": ph[i],
                "dissolved_oxygen": dissolved_oxygen[i],
                "bod": bod[i],
                "cod": cod[i],
                "turbidity": turbidity[i],
                "conductivity": conductivity[i],
                "fecal_coliform": fecal_coliform[i],
                "is_demonstration_data": True
            })
            
    df = pd.DataFrame(rows)
    output_path = r"C:\Users\SPEED\.gemini\antigravity\scratch\AI_Water_Quality_Forecasting\dataset\raw\cpcb_ganga_water_quality_demonstration_daily.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} records for {len(stations)} stations saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_ganga_dataset()
