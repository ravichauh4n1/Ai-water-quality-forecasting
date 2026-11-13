"""
Data Loader Module for AI Water Quality Forecasting System.
Handles data ingestion, schema harmonization, station metadata, and upload validation.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Default project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "dataset" / "raw" / "cpcb_ganga_water_quality_demonstration_daily.csv"
DEFAULT_STATIONS_PATH = PROJECT_ROOT / "dataset" / "raw" / "ganga_monitoring_stations.csv"

# Standard canonical column mapping
COLUMN_MAPPINGS = {
    # Date
    "date": "date",
    "timestamp": "date",
    "datetime": "date",
    "sampling_date": "date",
    
    # Station
    "station_code": "station_code",
    "station_id": "station_code",
    "code": "station_code",
    "station_name": "station_name",
    "station": "station_name",
    "location": "station_name",
    "monitoring_station": "station_name",
    
    # Geographic
    "river": "river",
    "water_body": "river",
    "state": "state",
    "lat": "latitude",
    "latitude": "latitude",
    "lon": "longitude",
    "longitude": "longitude",
    
    # Physico-Chemical Parameters
    "temperature": "temperature",
    "temp": "temperature",
    "water_temp": "temperature",
    "water_temperature": "temperature",
    
    "ph": "ph",
    "ph_level": "ph",
    "ph_val": "ph",
    
    "dissolved_oxygen": "dissolved_oxygen",
    "do": "dissolved_oxygen",
    "d.o.": "dissolved_oxygen",
    "do_mg_l": "dissolved_oxygen",
    
    "bod": "bod",
    "b.o.d.": "bod",
    "bod_mg_l": "bod",
    "biochemical_oxygen_demand": "bod",
    
    "cod": "cod",
    "c.o.d.": "cod",
    "cod_mg_l": "cod",
    "chemical_oxygen_demand": "cod",
    
    "turbidity": "turbidity",
    "turb": "turbidity",
    "turbidity_ntu": "turbidity",
    
    "conductivity": "conductivity",
    "cond": "conductivity",
    "ec": "conductivity",
    "electrical_conductivity": "conductivity",
    
    "fecal_coliform": "fecal_coliform",
    "fc": "fecal_coliform",
    "coliform": "fecal_coliform",
    "total_coliform": "fecal_coliform",
    
    "is_demonstration_data": "is_demonstration_data"
}

REQUIRED_CORE_PARAMS = ["ph", "dissolved_oxygen", "temperature"]
RECOMMENDED_PARAMS = ["bod", "turbidity", "conductivity"]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column names to canonical system names.
    Trims whitespace, converts to lowercase, replaces punctuation/spaces.
    """
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        clean_col = str(col).strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
        clean_col = "".join(c for c in clean_col if c.isalnum() or c == "_")
        # Direct match or partial key match
        matched = None
        if clean_col in COLUMN_MAPPINGS:
            matched = COLUMN_MAPPINGS[clean_col]
        else:
            for k, target in COLUMN_MAPPINGS.items():
                if k == clean_col or k in clean_col:
                    matched = target
                    break
        if matched:
            rename_map[col] = matched
            
    df = df.rename(columns=rename_map)
    return df


def load_raw_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads raw water quality data from CSV or Excel file.
    Standardizes schema, parses dates, and sorts chronologically.
    """
    path = Path(file_path) if file_path else DEFAULT_RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Water quality dataset not found at: {path}")
        
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
        
    df = standardize_columns(df)
    
    # Date parsing
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values(by=["station_code", "date"] if "station_code" in df.columns else ["date"]).reset_index(drop=True)
    
    # Ensure numeric types on water quality parameters
    numeric_cols = ["temperature", "ph", "dissolved_oxygen", "bod", "cod", "turbidity", "conductivity", "fecal_coliform"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    return df


def load_stations_metadata(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads river monitoring station metadata including GPS coordinates and CPCB designated class.
    """
    path = Path(file_path) if file_path else DEFAULT_STATIONS_PATH
    if not path.exists():
        # Fallback dictionary if file is missing
        return pd.DataFrame([
            {"station_code": "CPCB_G01", "station_name": "Rishikesh (Triveni Ghat)", "river": "Ganga", "state": "Uttarakhand", "latitude": 30.1033, "longitude": 78.2948, "cpcb_class": "A"},
            {"station_code": "CPCB_G02", "station_name": "Haridwar (Har Ki Pauri)", "river": "Ganga", "state": "Uttarakhand", "latitude": 29.9560, "longitude": 78.1700, "cpcb_class": "B"},
            {"station_code": "CPCB_G04", "station_name": "Kanpur (Jajmau)", "river": "Ganga", "state": "Uttar Pradesh", "latitude": 26.4310, "longitude": 80.3990, "cpcb_class": "D"},
            {"station_code": "CPCB_G06", "station_name": "Varanasi (Assi Ghat)", "river": "Ganga", "state": "Uttar Pradesh", "latitude": 25.2885, "longitude": 83.0063, "cpcb_class": "C"},
            {"station_code": "CPCB_G08", "station_name": "Patna (Gandhi Ghat)", "river": "Ganga", "state": "Bihar", "latitude": 25.6207, "longitude": 85.1720, "cpcb_class": "C"}
        ])
    return pd.read_csv(path)


def validate_uploaded_file(file_obj_or_path: Union[str, Path, object]) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Validates uploaded CSV or Excel file for required columns and data integrity.
    Returns (is_valid, message, standardized_df).
    """
    try:
        if hasattr(file_obj_or_path, "name"):
            filename = file_obj_or_path.name.lower()
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(file_obj_or_path)
            else:
                df = pd.read_csv(file_obj_or_path)
        else:
            path = Path(file_obj_or_path)
            if path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
                
        if df.empty:
            return False, "The uploaded file is empty.", None
            
        df = standardize_columns(df)
        
        # Check required columns
        missing_core = [p for p in REQUIRED_CORE_PARAMS if p not in df.columns]
        if missing_core:
            return False, f"Missing required core parameters: {', '.join(missing_core)}. Found columns: {list(df.columns)}", None
            
        if "date" not in df.columns:
            return False, "Dataset must include a 'date' or 'timestamp' column for time-series forecasting.", None
            
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        valid_dates = df["date"].notna().sum()
        if valid_dates < 10:
            return False, f"Only {valid_dates} valid date entries found. At least 10 observations required for time-series.", None
            
        # Ensure default station if not present
        if "station_name" not in df.columns:
            df["station_name"] = "Custom Station"
        if "station_code" not in df.columns:
            df["station_code"] = "CUST_01"
            
        # Convert numeric columns
        for c in REQUIRED_CORE_PARAMS + [col for col in RECOMMENDED_PARAMS if col in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return True, f"Dataset successfully validated! {len(df):,} valid records parsed.", df
        
    except Exception as e:
        return False, f"File reading error: {str(e)}", None


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """
    Extracts summary statistics, date range, station count, and missing counts.
    """
    stations = df["station_name"].unique().tolist() if "station_name" in df.columns else ["Unknown"]
    date_min = df["date"].min().strftime("%Y-%m-%d") if "date" in df.columns and len(df) > 0 else "N/A"
    date_max = df["date"].max().strftime("%Y-%m-%d") if "date" in df.columns and len(df) > 0 else "N/A"
    
    numeric_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "bod", "cod", "turbidity", "conductivity", "fecal_coliform"] if c in df.columns]
    stats = df[numeric_cols].describe().to_dict() if numeric_cols else {}
    null_counts = df.isnull().sum().to_dict()
    
    is_demo = bool(df["is_demonstration_data"].iloc[0]) if "is_demonstration_data" in df.columns else False
    
    return {
        "total_records": len(df),
        "total_stations": len(stations),
        "stations": stations,
        "date_range": (date_min, date_max),
        "columns": list(df.columns),
        "numeric_parameters": numeric_cols,
        "missing_counts": null_counts,
        "is_demonstration_data": is_demo,
        "statistics": stats
    }


if __name__ == "__main__":
    print("Testing data_loader.py...")
    raw_df = load_raw_data()
    print(f"Loaded raw dataset with {len(raw_df):,} rows.")
    stations_df = load_stations_metadata()
    print(f"Loaded {len(stations_df)} stations.")
    summary = get_dataset_summary(raw_df)
    print(f"Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
    print(f"Stations: {summary['stations']}")
    print(f"Is Demo Data: {summary['is_demonstration_data']}")
    print("Data Loader Module verified successfully!")
