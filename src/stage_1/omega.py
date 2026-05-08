from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.pages.page_1 import (
    discover_player_files,
    is_supported_game_sheet,
    load_workbook_sheets,
)
from src.stage_0.score_0 import Score0Params, normalized_shannon_entropy
from src.stage_1.stage_1 import Stage1Params, stage_1_step
from src.stage_1.score_adapter import (
    stage1_br_count_distribution,
    br_region_weights,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / "resultats_stage_1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIGMA_SCORE_GRID = np.round(np.arange(0.5, 15.5, 0.5), 2)

# Fixed Stage 1 parameters from joint map calibration
SIGMA_COMP_FIXED = 2.0
SIGMA_SPACE_FIXED = 1.5
LAMBDA_STAT = 0.0
LAMBDA_BR = 1.0

PLAYER_FILTER: str | None = None
USE_RELATIVE_OVERLAP = True

PARSIMONY_OBJECTIVE_FRACTION = 0.90

# Sensitivity grid for omega weights
OMEGA_EXPECTED_GRID = [0.40, 0.50, 0.60]
OMEGA_NONZERO_GRID = [0.25, 0.35, 0.45]
OMEGA_ENTROPY_GRID = [0.30, 0.40, 0.50]


# =============================================================================
# DATA LOADING
# =============================================================================

def validate_fixations(fixations: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"AOI", "Time"}
    missing = required_cols - set(fixations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = fixations.copy()
    clean = clean.dropna(subset=["AOI", "Time"]).reset_index(drop=True)
    clean["AOI"] = pd.to_numeric(clean["AOI"], errors="coerce")
    clean["Time"] = pd.to_numeric(clean["Time"], errors="coerce")
    clean = clean.dropna(subset=["AOI", "Time"]).reset_index(drop=True)

    clean["AOI"] = clean["AOI"].astype(int)
    clean["Time"] = clean["Time"].astype(float)

    clean = clean[(clean["AOI"] >= 1) & (clean["AOI"] <= 100)]
    clean = clean[clean["Time"] > 0].reset_index(drop=True)
    return clean


def load_stage1_sequences(player_filter: str | None = None) -> list[tuple[str, str, pd.DataFrame]]:
    project_root = Path(__file__).resolve().parents[2]
    player_files = discover_player_files(str(project_root))

    sequences: list[tuple[str, str, pd.DataFrame]] = []

    for player_name, workbook_path in sorted(player_files.items()):
        if player_filter is not None and player_name != player_filter:
            continue

        try:
            workbook = load_workbook_sheets(workbook_path)
        except Exception as exc:
            print(f"Skipping workbook {player_name}: {exc}")
            continue

        for sheet_name, fixations in workbook.items():
            if not is_supported_game_sheet(sheet_name):
                continue

            try:
                clean = validate_fixations(fixations)
            except Exception as exc:
                print(f"Skipping sheet {player_name} / {sheet_name}: {exc}")
                continue

            if len(clean) < 2:
                continue

            sequences.append((player_name, sheet_name, clean))

    return sequences


# =============================================================================
# SCORE EVALUATION
# =============================================================================

def mean_score_entropy_for_sequence(
    fixations: pd.DataFrame,
    game: int | str,
    sigma_score: float,
    stage1_params: Stage1Params,
) -> float:
    clean = validate_fixations(fixations)
    if len(clean) < 2:
        return np.nan

    y = np.arange(1, 101, dtype=float)
    entropies: list[float] = []

    for t in range(len(clean) - 1):
        x_t = int(clean.iloc[t]["AOI"])

        step = stage_1_step(
            game=game,
            x_t=float(x_t),
            params=stage1_params,
        )

        target = float(step["T_g"])
        w_t = br_region_weights(
            y=y,
            target=target,
            sigma_score=float(sigma_score),
        )
        entropies.append(float(normalized_shannon_entropy(w_t)))

    return float(np.mean(entropies)) if entropies else np.nan


def evaluate_sequence_score1(
    fixations: pd.DataFrame,
    game: int | str,
    stage1_params: Stage1Params,
    sigma_score: float,
    use_relative_overlap: bool,
) -> dict[str, float]:
    score_params = Score0Params(
        sigma_score=float(sigma_score),
        use_relative_overlap=bool(use_relative_overlap),
    )

    result = stage1_br_count_distribution(
        fixations=fixations,
        game=game,
        stage1_params=stage1_params,
        score_params=score_params,
    )

    summary = result["summary"]
    transition_df = result["transition_df"]
    distribution = np.asarray(result["distribution"], dtype=float)

    mean_p_t = float(transition_df["p_t"].mean()) if not transition_df.empty else np.nan
    prob_zero = float(distribution[0]) if len(distribution) > 0 else np.nan
    mean_entropy = mean_score_entropy_for_sequence(
        fixations=fixations,
        game=game,
        sigma_score=float(sigma_score),
        stage1_params=stage1_params,
    )

    return {
        "n_transitions": float(len(transition_df)),
        "expected_count": float(summary["expected_count"]),
        "variance": float(summary["variance"]),
        "mode": float(summary["mode"]),
        "median": float(summary["median"]),
        "prob_zero": prob_zero,
        "mean_p_t": mean_p_t,
        "mean_score_entropy": mean_entropy,
    }


def aggregate_score1_metrics(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_score: float,
) -> dict[str, Any]:
    stage1_params = Stage1Params(
        lambda_stat=LAMBDA_STAT,
        lambda_br=LAMBDA_BR,
        sigma_comp=SIGMA_COMP_FIXED,
        sigma_space=SIGMA_SPACE_FIXED,
    )

    per_sequence_rows: list[dict[str, Any]] = []

    for player_name, sheet_name, fixations in sequences:
        metrics = evaluate_sequence_score1(
            fixations=fixations,
            game=sheet_name,
            stage1_params=stage1_params,
            sigma_score=float(sigma_score),
            use_relative_overlap=USE_RELATIVE_OVERLAP,
        )

        if metrics["n_transitions"] <= 0:
            continue

        per_sequence_rows.append(
            {
                "player": player_name,
                "game": sheet_name,
                "sigma_score": float(sigma_score),
                **metrics,
            }
        )

    per_sequence_df = pd.DataFrame(per_sequence_rows)

    if per_sequence_df.empty:
        summary = {
            "sigma_score": float(sigma_score),
            "n_sequences": 0.0,
            "mean_expected_count": np.nan,
            "mean_variance": np.nan,
            "mean_mode": np.nan,
            "mean_median": np.nan,
            "mean_prob_zero": np.nan,
            "mean_p_t": np.nan,
            "mean_score_entropy": np.nan,
        }
        return {"summary": summary, "per_sequence": per_sequence_df}

    summary = {
        "sigma_score": float(sigma_score),
        "n_sequences": float(len(per_sequence_df)),
        "mean_expected_count": float(per_sequence_df["expected_count"].mean()),
        "mean_variance": float(per_sequence_df["variance"].mean()),
        "mean_mode": float(per_sequence_df["mode"].mean()),
        "mean_median": float(per_sequence_df["median"].mean()),
        "mean_prob_zero": float(per_sequence_df["prob_zero"].mean()),
        "mean_p_t": float(per_sequence_df["mean_p_t"].mean()),
        "mean_score_entropy": float(per_sequence_df["mean_score_entropy"].mean()),
    }

    return {"summary": summary, "per_sequence": per_sequence_df}


# =============================================================================
# GRID SEARCH
# =============================================================================

def run_sigma_score_grid_search(
    sequences: list[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    summary_rows: list[dict[str, Any]] = []

    for sigma_score in SIGMA_SCORE_GRID:
        result = aggregate_score1_metrics(
            sequences=sequences,
            sigma_score=float(sigma_score),
        )
        summary_rows.append(result["summary"])

    return pd.DataFrame(summary_rows).sort_values("sigma_score").reset_index(drop=True)


def add_objective_columns(
    summary_df: pd.DataFrame,
    omega_expected: float,
    omega_nonzero: float,
    omega_entropy: float,
) -> pd.DataFrame:
    df = summary_df.copy()

    expected_count_max = float(df["mean_expected_count"].max()) if len(df) > 0 else 1.0
    if expected_count_max <= 0:
        df["norm_expected_count"] = 0.0
    else:
        df["norm_expected_count"] = df["mean_expected_count"] / expected_count_max

    df["nonzero_score"] = 1.0 - df["mean_prob_zero"]

    df["objective_score"] = (
        float(omega_expected) * df["norm_expected_count"]
        + float(omega_nonzero) * df["nonzero_score"]
        - float(omega_entropy) * df["mean_score_entropy"]
    )

    return df


def compute_best_sigma_score(
    summary_df: pd.DataFrame,
    omega_expected: float,
    omega_nonzero: float,
    omega_entropy: float,
) -> dict[str, float]:
    df = add_objective_columns(summary_df, omega_expected, omega_nonzero, omega_entropy)
    df = df.dropna(subset=["objective_score"]).sort_values("sigma_score")

    if df.empty:
        return {
            "best_sigma_score": np.nan,
            "best_objective_score": np.nan,
        }

    best_idx = int(df["objective_score"].idxmax())
    best_row = df.loc[best_idx]

    return {
        "best_sigma_score": float(best_row["sigma_score"]),
        "best_objective_score": float(best_row["objective_score"]),
        "best_mean_expected_count": float(best_row["mean_expected_count"]),
        "best_mean_prob_zero": float(best_row["mean_prob_zero"]),
        "best_mean_p_t": float(best_row["mean_p_t"]),
        "best_mean_score_entropy": float(best_row["mean_score_entropy"]),
    }


def compute_parsimonious_sigma_score(
    summary_df: pd.DataFrame,
    omega_expected: float,
    omega_nonzero: float,
    omega_entropy: float,
) -> dict[str, float]:
    df = add_objective_columns(summary_df, omega_expected, omega_nonzero, omega_entropy)
    df = df.dropna(subset=["objective_score"]).sort_values("sigma_score")

    if df.empty:
        return {
            "parsimonious_sigma_score": np.nan,
            "objective_max": np.nan,
            "objective_threshold": np.nan,
            "parsimonious_objective_score": np.nan,
            "achieved_fraction": np.nan,
        }

    objective_max = float(df["objective_score"].max())
    threshold = PARSIMONY_OBJECTIVE_FRACTION * objective_max

    eligible = df[df["objective_score"] >= threshold]

    if eligible.empty:
        chosen_row = df.loc[df["objective_score"].idxmax()]
    else:
        chosen_row = eligible.iloc[0]

    achieved_fraction = (
        float(chosen_row["objective_score"]) / objective_max
        if abs(objective_max) > 1e-12
        else np.nan
    )

    return {
        "parsimonious_sigma_score": float(chosen_row["sigma_score"]),
        "objective_max": objective_max,
        "objective_threshold": threshold,
        "parsimonious_objective_score": float(chosen_row["objective_score"]),
        "parsimonious_mean_expected_count": float(chosen_row["mean_expected_count"]),
        "parsimonious_mean_prob_zero": float(chosen_row["mean_prob_zero"]),
        "parsimonious_mean_p_t": float(chosen_row["mean_p_t"]),
        "parsimonious_mean_score_entropy": float(chosen_row["mean_score_entropy"]),
        "achieved_fraction": achieved_fraction,
    }


# =============================================================================
# MAIN SENSITIVITY LOOP
# =============================================================================

def main() -> None:
    sequences = load_stage1_sequences(player_filter=PLAYER_FILTER)

    if not sequences:
        raise RuntimeError("No valid sequences found for Stage 1 omega sensitivity analysis.")

    # Compute sigma_score grid once
    summary_df = run_sigma_score_grid_search(sequences=sequences)

    result_rows: list[dict[str, Any]] = []

    for omega_e in OMEGA_EXPECTED_GRID:
        for omega_nz in OMEGA_NONZERO_GRID:
            for omega_h in OMEGA_ENTROPY_GRID:
                best_info = compute_best_sigma_score(
                    summary_df=summary_df,
                    omega_expected=float(omega_e),
                    omega_nonzero=float(omega_nz),
                    omega_entropy=float(omega_h),
                )

                parsimonious_info = compute_parsimonious_sigma_score(
                    summary_df=summary_df,
                    omega_expected=float(omega_e),
                    omega_nonzero=float(omega_nz),
                    omega_entropy=float(omega_h),
                )

                result_rows.append(
                    {
                        "omega_expected": float(omega_e),
                        "omega_nonzero": float(omega_nz),
                        "omega_entropy": float(omega_h),

                        "best_sigma_score": best_info["best_sigma_score"],
                        "best_objective_score": best_info["best_objective_score"],
                        "best_mean_expected_count": best_info.get("best_mean_expected_count", np.nan),
                        "best_mean_prob_zero": best_info.get("best_mean_prob_zero", np.nan),
                        "best_mean_p_t": best_info.get("best_mean_p_t", np.nan),
                        "best_mean_score_entropy": best_info.get("best_mean_score_entropy", np.nan),

                        "parsimonious_sigma_score": parsimonious_info["parsimonious_sigma_score"],
                        "parsimonious_objective_score": parsimonious_info["parsimonious_objective_score"],
                        "parsimonious_mean_expected_count": parsimonious_info.get("parsimonious_mean_expected_count", np.nan),
                        "parsimonious_mean_prob_zero": parsimonious_info.get("parsimonious_mean_prob_zero", np.nan),
                        "parsimonious_mean_p_t": parsimonious_info.get("parsimonious_mean_p_t", np.nan),
                        "parsimonious_mean_score_entropy": parsimonious_info.get("parsimonious_mean_score_entropy", np.nan),
                        "parsimonious_achieved_fraction": parsimonious_info["achieved_fraction"],
                    }
                )

    results_df = pd.DataFrame(result_rows).sort_values(
        ["omega_expected", "omega_nonzero", "omega_entropy"]
    ).reset_index(drop=True)

    output_csv = RESULTS_DIR / "stage1_sigma_score_omega_sensitivity.csv"
    results_df.to_csv(output_csv, index=False)

    print("=" * 80)
    print("STAGE 1 OMEGA SENSITIVITY")
    print("=" * 80)
    print(f"Player filter               : {PLAYER_FILTER if PLAYER_FILTER is not None else 'ALL PLAYERS'}")
    print(f"Number of sequences         : {len(sequences)}")
    print(f"Tested omega triplets       : {len(results_df)}")
    print()
    print("Unique best_sigma_score values:")
    print(sorted(results_df["best_sigma_score"].dropna().unique().tolist()))
    print()
    print("Unique parsimonious_sigma_score values:")
    print(sorted(results_df["parsimonious_sigma_score"].dropna().unique().tolist()))
    print()
    print(f"Saved CSV                   : {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()