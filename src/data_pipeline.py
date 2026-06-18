import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import os

def process_data(input_csv="data/violations.csv", output_csv="data/violations_scored.csv"):
    if not os.path.exists(input_csv):
        print(f"Error: Could not find {input_csv}. Please download the dataset and place it in the 'data' folder.")
        return

    print("Loading data...")
    df = pd.read_csv(input_csv)
    
    # Use standard column names based on the sample data you provided
    if "created_datetime" in df.columns:
        df["created_datetime"] = pd.to_datetime(df["created_datetime"], format='mixed', errors='coerce')
        df["hour"] = df["created_datetime"].dt.hour
    else:
        df["hour"] = 12 # Fallback if time is missing

    if "vehicle_type" in df.columns:
        df["vehicle_type"] = df["vehicle_type"].astype(str).str.upper().str.strip()
    
    # Keep approved + pending records, drop rejected (assuming 'validation_status' exists)
    if "validation_status" in df.columns:
        df = df[df["validation_status"] != "rejected"]

    # ---- 2. Vehicle impact weight ----
    VEHICLE_WEIGHT = {
        "SCOOTER": 1, "MOTORCYCLE": 1,
        "PASSENGER AUTO": 2, "AUTO": 2,
        "CAR": 3,
        "MAXI-CAB": 4, "MAXI CAB": 4,
        "BUS": 5,
        "TANKER": 6, "TRUCK": 6, "LORRY": 6,
    }
    if "vehicle_type" in df.columns:
        df["vehicle_weight"] = df["vehicle_type"].map(VEHICLE_WEIGHT).fillna(2)
    else:
        df["vehicle_weight"] = 2

    # ---- 3. Junction proximity ----
    if "junction_name" in df.columns:
        df["near_junction"] = (df["junction_name"] != "No Junction").astype(int)
    else:
        df["near_junction"] = 0

    # ---- 4. Violation frequency per ~30m grid cell ----
    if "latitude" in df.columns and "longitude" in df.columns:
        df["geo_cell"] = (df["latitude"].round(3).astype(str) + "_" +
                          df["longitude"].round(3).astype(str))
        freq = df.groupby("geo_cell").size().rename("location_violation_count")
        df = df.merge(freq, on="geo_cell", how="left")
    else:
        df["location_violation_count"] = 1

    # ---- 5. Time-of-day factor ----
    def time_factor(hour):
        if hour in [8, 9, 18, 19]:
            return 1.0
        elif hour in [7, 10, 17, 20]:
            return 0.7
        return 0.3
    
    df["time_factor"] = df["hour"].apply(time_factor)

    # ---- 6. Congestion Impact Score (CIS) ----
    def normalize(s):
        if s.max() == s.min():
            return s * 0
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    w1, w2, w3, w4 = 0.35, 0.20, 0.25, 0.20
    df["CIS"] = 100 * (
        w1 * normalize(df["vehicle_weight"]) +
        w2 * df["near_junction"] +
        w3 * normalize(df["location_violation_count"]) +
        w4 * df["time_factor"]
    )
    
    # ---- 7. Priority and Action logic ----
    def priority_and_action(cis):
        if cis <= 30:  return "LOW", "Monitor"
        if cis <= 55:  return "MEDIUM", "Issue Notice"
        if cis <= 80:  return "HIGH", "Immediate Enforcement"
        return "CRITICAL", "Tow Vehicle"

    df[["priority", "action"]] = df["CIS"].apply(
        lambda x: pd.Series(priority_and_action(x))
    )

    # ---- 8. Hotspot Clustering using DBSCAN ----
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = df[["latitude", "longitude"]].dropna().to_numpy()
        if len(coords) > 0:
            print("Clustering hotspots...")
            db = DBSCAN(eps=0.001, min_samples=4).fit(coords)   # eps ≈ 100m in lat/lon degrees
            # Assign labels back, aligning with dropped NA rows if necessary.
            # Simplified for dataset where lat/lon are always present:
            df["hotspot_cluster"] = db.labels_                   # -1 = noise, not a hotspot
        else:
            df["hotspot_cluster"] = -1
    else:
        df["hotspot_cluster"] = -1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Successfully processed {len(df)} records and saved to {output_csv}")

if __name__ == "__main__":
    process_data()
