import pandas as pd
import re

def compute_aoi_time(df, aoi_cols, time_col):
    results = []

    df = df.copy()

    # Convert time column safely
    if df[time_col].dtype == "object":
        # Try datetime conversion first
        time_as_datetime = pd.to_datetime(df[time_col], errors="coerce")

        if time_as_datetime.notna().sum() > 0:
            # Convert to milliseconds relative to first valid timestamp
            first_time = time_as_datetime.dropna().iloc[0]
            df[time_col] = (time_as_datetime - first_time).dt.total_seconds() * 1000
        else:
            # Fallback: try numeric conversion
            df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    current_aoi = None
    acc_time = 0.0
    prev_t = None

    for _, row in df.iterrows():
        t = row[time_col]

        if pd.isna(t):
            continue

        active_aois = [c for c in aoi_cols if row[c] == 1]

        if len(active_aois) != 1:
            prev_t = t
            continue

        match = re.search(r"\[(\d+)\]", active_aois[0])
        if match is None:
            prev_t = t
            continue

        active_aoi = int(match.group(1))

        if current_aoi is None:
            current_aoi = active_aoi
            acc_time = 0.0
            prev_t = t
            continue

        if active_aoi == current_aoi:
            if prev_t is not None:
                acc_time += t - prev_t
        else:
            results.append({
                "AOI": current_aoi,
                "Time": acc_time
            })

            current_aoi = active_aoi
            acc_time = 0.0

        prev_t = t

    if current_aoi is not None:
        results.append({
            "AOI": current_aoi,
            "Time": acc_time
        })

    return pd.DataFrame(results)