import pandas as pd
import re

def compute_aoi_time(df, aoi_cols, time_col):
    results = []
    current_aoi = None
    acc_time = 0.0
    prev_t = None

    for _, row in df.iterrows():
        t = row[time_col]
        if pd.isna(t):
            continue

        # AOI detected
        active_aois = [c for c in aoi_cols if row[c] == 1]

        if len(active_aois) != 1:
            prev_t = t
            continue

        # Extract AOI number (e.g. AOI[23]Hit → 23)
        match = re.search(r"\[(\d+)\]", active_aois[0])
        if match is None:
            prev_t = t
            continue

        active_aoi = int(match.group(1))

        # Starting AOI
        if current_aoi is None:
            current_aoi = active_aoi
            acc_time = 0.0
            prev_t = t
            continue

        # Same AOI → accumulate time
        if active_aoi == current_aoi:
            if prev_t is not None:
                acc_time += (t - prev_t)
            prev_t = t
            continue

        # Switching AOI
        results.append({
            "AOI": current_aoi,
            "Time": acc_time
        })

        current_aoi = active_aoi
        acc_time = 0.0
        prev_t = t

    # End AOI
    if current_aoi is not None:
        results.append({
            "AOI": current_aoi,
            "Time": acc_time
        })

    return pd.DataFrame(results)