import pandas as pd
import re
from pathlib import Path
from src.data_processing.time_processing import compute_aoi_time


# =========================
# Paths
# =========================
DATA_DIR = Path("/Users/lukaboisgibault/Desktop/PhD/Eye_Tracking /Données Excel/GAME123")

OUTPUT_DIR = Path("/Users/lukaboisgibault/Desktop/PhD/Eye_Tracking /Données Excel/Extraction")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# Loop over players
# =========================
skip_box = []

for file_path in DATA_DIR.glob("*.xlsx"):

    if file_path.name.startswith("~$"):
        print(f"Skipping {file_path.name} (issue)")
        skip_box.append(file_path.name)
        continue

    player_name = file_path.stem
    output_file = OUTPUT_DIR / (f"{player_name}_AOI_TIME.xlsx")

    if output_file.exists():
        print(f"Skipping {file_path.name} (already exists)")
        continue

    print(f"\nProcessing {player_name}")

    xls = pd.ExcelFile(file_path, engine="openpyxl")
    game_sheets = [s for s in xls.sheet_names if s.startswith("GAME")]

    output_file = OUTPUT_DIR / f"{player_name}_AOI_TIME.xlsx"

    # =========================
    # One Excel per player
    # =========================
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

        for game in game_sheets:
            print(f"  -> {game}")

            df = pd.read_excel(xls, sheet_name=game)

            # AOI columns (handles .1 and spaces)
            aoi_cols = [
                c for c in df.columns
                if re.fullmatch(r"AOI\[\s*\d+\]Hit(\.1)?", str(c))
            ]

            # Time column
            if "EyeTrackerTimestamp" in df.columns:
                time_col = "EyeTrackerTimestamp"
            else:
                time_col = "LocalTimeStamp"

            # Compute AOI time
            df_result = compute_aoi_time(df, aoi_cols, time_col)

            # Write sheet
            df_result.to_excel(writer, sheet_name=game, index=False)

    print(f"Saved → {output_file}")

print(f"Done for all xlsx files in : {DATA_DIR}")
print(f"There is an issue with : {skip_box}")