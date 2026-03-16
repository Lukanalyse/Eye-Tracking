from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from src.data_processing.time_processing import compute_aoi_time


def get_data_dirs() -> tuple[Path, Path]:
    """Return input/output directories for Excel processing.

    Priority:
    1) env vars EYE_TRACKING_DATA_DIR / EYE_TRACKING_OUTPUT_DIR
    2) historical Desktop folders used in this project
    3) local fallback inside the repository: data/raw and data/processed
    """
    project_root = Path(__file__).resolve().parents[2]

    default_data = project_root / "data" / "raw"
    default_output = project_root / "data" / "processed"

    desktop_base = Path.home() / "Desktop" / "PhD" / "Eye_Tracking" / "Donnees Excel"
    legacy_data = desktop_base / "GAME123"
    legacy_output = desktop_base / "Extraction"

    data_dir = Path(os.getenv("EYE_TRACKING_DATA_DIR", "")) if os.getenv("EYE_TRACKING_DATA_DIR") else legacy_data
    output_dir = Path(os.getenv("EYE_TRACKING_OUTPUT_DIR", "")) if os.getenv("EYE_TRACKING_OUTPUT_DIR") else legacy_output

    if not data_dir.exists():
        data_dir = default_data
    if not output_dir.exists():
        output_dir = default_output

    output_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, output_dir


def process_player_file(file_path: Path, output_dir: Path, overwrite: bool = False) -> Path | None:
    """Process one raw player workbook and save AOI/Time sheets."""
    if file_path.name.startswith("~$"):
        return None

    player_name = file_path.stem
    output_file = output_dir / f"{player_name}_AOI_TIME.xlsx"

    if output_file.exists() and not overwrite:
        return output_file

    xls = pd.ExcelFile(file_path, engine="openpyxl")
    game_sheets = [s for s in xls.sheet_names if str(s).upper().startswith("GAME")]

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for game in game_sheets:
            df = pd.read_excel(xls, sheet_name=game)

            aoi_cols = [
                c for c in df.columns
                if re.fullmatch(r"AOI\[\s*\d+]Hit(\.1)?", str(c))
            ]

            if "EyeTrackerTimestamp" in df.columns:
                time_col = "EyeTrackerTimestamp"
            elif "LocalTimeStamp" in df.columns:
                time_col = "LocalTimeStamp"
            else:
                continue

            df_result = compute_aoi_time(df, aoi_cols, time_col)
            df_result.to_excel(writer, sheet_name=game, index=False)

    return output_file


def process_all_players(overwrite: bool = False) -> list[Path]:
    """Batch process all raw player workbooks in the configured data directory."""
    data_dir, output_dir = get_data_dirs()
    created: list[Path] = []

    for file_path in sorted(data_dir.glob("*.xlsx")):
        result = process_player_file(file_path, output_dir=output_dir, overwrite=overwrite)
        if result is not None:
            created.append(result)
    return created

if __name__ == "__main__":
    data_dir, output_dir = get_data_dirs()
    print(f"Input directory : {data_dir}")
    print(f"Output directory: {output_dir}")
    generated = process_all_players(overwrite=False)
    print(f"Processed files  : {len(generated)}")
