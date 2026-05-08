from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pages.page_1 import (
    discover_player_files,
    is_supported_game_sheet,
    load_workbook_sheets,
)
from src.stage_0.stage_0 import Stage0Params, stage_0_step


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / "resultats_stage_0"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIGMA_COMP_GRID = np.round(np.arange(0.5, 15.5, 0.5), 2)

LAMBDA_STAT = 0.0
LAMBDA_BR = 1.0

PLAYER_FILTER: str | None = None

PARSIMONY_GAIN_FRACTION = 0.90

# Sensitivity grid for penalty weights
LAMBDA_ENTROPY_GRID = [0.10, 0.15, 0.20, 0.25]
LAMBDA_RANK_GRID = [0.03, 0.05, 0.07, 0.10]


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
    clean["AOI"] = clean["AOI"].astype(int)
    clean["Time"] = clean["Time"].astype(float)
    clean = clean[(clean["AOI"] >= 1) & (clean["AOI"] <= 100)]
    clean = clean[clean["Time"] > 0].reset_index(drop=True)
    return clean


def load_stage0_sequences(player_filter: str | None = None) -> list[tuple[str, str, pd.DataFrame]]:
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
# METRICS
# =============================================================================

def normalized_entropy(probabilities: np.ndarray) -> float:
    q = np.asarray(probabilities, dtype=float)
    q = np.clip(q, 1e-12, 1.0)
    entropy = -float(np.sum(q * np.log(q)))
    max_entropy = float(np.log(len(q)))
    return float(entropy / max_entropy)


def rank_of_observed_aoi(probabilities: np.ndarray, observed_aoi: int) -> int:
    q = np.asarray(probabilities, dtype=float)
    order = np.argsort(q)[::-1]
    observed_index = int(observed_aoi) - 1
    matches = np.where(order == observed_index)[0]
    if len(matches) == 0:
        return len(q)
    return int(matches[0]) + 1


def evaluate_sequence_stage0(
    fixations: pd.DataFrame,
    game: int | str,
    params: Stage0Params,
) -> dict[str, float]:
    clean = validate_fixations(fixations)

    if len(clean) < 2:
        return {
            "n_predictions": 0.0,
            "mean_log_likelihood": np.nan,
            "mean_entropy": np.nan,
            "mean_rank": np.nan,
        }

    log_likelihoods: list[float] = []
    entropies: list[float] = []
    ranks: list[int] = []

    for t in range(len(clean) - 1):
        x_t = int(clean.iloc[t]["AOI"])
        x_next = int(clean.iloc[t + 1]["AOI"])

        step = stage_0_step(
            game=game,
            x_t=float(x_t),
            params=params,
        )

        q_t = np.asarray(step["q_t"], dtype=float)
        prob_next = max(float(q_t[x_next - 1]), 1e-12)

        log_likelihoods.append(float(np.log(prob_next)))
        entropies.append(normalized_entropy(q_t))
        ranks.append(rank_of_observed_aoi(q_t, x_next))

    return {
        "n_predictions": float(len(log_likelihoods)),
        "mean_log_likelihood": float(np.mean(log_likelihoods)),
        "mean_entropy": float(np.mean(entropies)),
        "mean_rank": float(np.mean(ranks)),
    }


def aggregate_stage0_metrics(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_comp: float,
    lambda_entropy_penalty: float,
    lambda_rank_penalty: float,
) -> dict[str, Any]:
    params = Stage0Params(
        lambda_stat=LAMBDA_STAT,
        lambda_br=LAMBDA_BR,
        sigma_comp=float(sigma_comp),
    )

    total_n = 0.0
    weighted_ll = 0.0
    weighted_entropy = 0.0
    weighted_rank = 0.0

    for _, sheet_name, fixations in sequences:
        metrics = evaluate_sequence_stage0(
            fixations=fixations,
            game=sheet_name,
            params=params,
        )

        n_pred = float(metrics["n_predictions"])
        if n_pred <= 0:
            continue

        total_n += n_pred
        weighted_ll += n_pred * float(metrics["mean_log_likelihood"])
        weighted_entropy += n_pred * float(metrics["mean_entropy"])
        weighted_rank += n_pred * float(metrics["mean_rank"])

    if total_n <= 0:
        return {
            "sigma_comp": float(sigma_comp),
            "mean_log_likelihood": np.nan,
            "mean_entropy": np.nan,
            "mean_rank": np.nan,
            "normalized_rank": np.nan,
            "objective": np.nan,
        }

    mean_ll = weighted_ll / total_n
    mean_entropy = weighted_entropy / total_n
    mean_rank = weighted_rank / total_n
    normalized_rank = float(np.clip((mean_rank - 1.0) / 99.0, 0.0, 1.0))

    objective = (
        float(mean_ll)
        - float(lambda_entropy_penalty) * float(mean_entropy)
        - float(lambda_rank_penalty) * float(normalized_rank)
    )

    return {
        "sigma_comp": float(sigma_comp),
        "mean_log_likelihood": float(mean_ll),
        "mean_entropy": float(mean_entropy),
        "mean_rank": float(mean_rank),
        "normalized_rank": float(normalized_rank),
        "objective": float(objective),
    }


def run_sigma_grid_for_lambda_pair(
    sequences: list[tuple[str, str, pd.DataFrame]],
    lambda_entropy_penalty: float,
    lambda_rank_penalty: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for sigma_comp in SIGMA_COMP_GRID:
        rows.append(
            aggregate_stage0_metrics(
                sequences=sequences,
                sigma_comp=float(sigma_comp),
                lambda_entropy_penalty=float(lambda_entropy_penalty),
                lambda_rank_penalty=float(lambda_rank_penalty),
            )
        )

    df = pd.DataFrame(rows).sort_values("sigma_comp").reset_index(drop=True)
    return df


def compute_parsimonious_sigma(summary_df: pd.DataFrame) -> dict[str, float]:
    df = summary_df.dropna(subset=["mean_log_likelihood"]).copy().sort_values("sigma_comp")

    ll_min = float(df["mean_log_likelihood"].min())
    ll_max = float(df["mean_log_likelihood"].max())
    total_gain = ll_max - ll_min

    if total_gain <= 0:
        chosen_row = df.iloc[0]
        return {
            "sigma_comp": float(chosen_row["sigma_comp"]),
            "gain_fraction": 0.0,
        }

    threshold_ll = ll_min + PARSIMONY_GAIN_FRACTION * total_gain
    eligible = df[df["mean_log_likelihood"] >= threshold_ll]

    if eligible.empty:
        chosen_row = df.iloc[-1]
    else:
        chosen_row = eligible.iloc[0]

    achieved_fraction = (float(chosen_row["mean_log_likelihood"]) - ll_min) / total_gain

    return {
        "sigma_comp": float(chosen_row["sigma_comp"]),
        "gain_fraction": float(achieved_fraction),
    }


# =============================================================================
# MAIN SENSITIVITY LOOP
# =============================================================================

def main() -> None:
    sequences = load_stage0_sequences(player_filter=PLAYER_FILTER)

    if not sequences:
        raise RuntimeError("No valid sequences found for lambda sensitivity analysis.")

    result_rows: list[dict[str, Any]] = []

    for lambda_h in LAMBDA_ENTROPY_GRID:
        for lambda_r in LAMBDA_RANK_GRID:
            if lambda_h <= lambda_r:
                # impose the hierarchy lambda_H > lambda_R
                continue

            summary_df = run_sigma_grid_for_lambda_pair(
                sequences=sequences,
                lambda_entropy_penalty=float(lambda_h),
                lambda_rank_penalty=float(lambda_r),
            )

            best_idx = int(summary_df["objective"].idxmax())
            best_row = summary_df.loc[best_idx]

            parsimonious_info = compute_parsimonious_sigma(summary_df)
            parsimonious_sigma = float(parsimonious_info["sigma_comp"])
            parsimonious_row = summary_df.loc[summary_df["sigma_comp"] == parsimonious_sigma].iloc[0]

            result_rows.append(
                {
                    "lambda_entropy_penalty": float(lambda_h),
                    "lambda_rank_penalty": float(lambda_r),
                    "best_sigma_comp": float(best_row["sigma_comp"]),
                    "best_objective": float(best_row["objective"]),
                    "best_mean_log_likelihood": float(best_row["mean_log_likelihood"]),
                    "best_mean_entropy": float(best_row["mean_entropy"]),
                    "best_mean_rank": float(best_row["mean_rank"]),
                    "parsimonious_sigma_comp": float(parsimonious_sigma),
                    "parsimonious_objective": float(parsimonious_row["objective"]),
                    "parsimonious_mean_log_likelihood": float(parsimonious_row["mean_log_likelihood"]),
                    "parsimonious_mean_entropy": float(parsimonious_row["mean_entropy"]),
                    "parsimonious_mean_rank": float(parsimonious_row["mean_rank"]),
                    "parsimonious_gain_fraction": float(parsimonious_info["gain_fraction"]),
                }
            )

    results_df = pd.DataFrame(result_rows).sort_values(
        ["lambda_entropy_penalty", "lambda_rank_penalty"]
    ).reset_index(drop=True)

    output_csv = RESULTS_DIR / "stage0_lambda_sensitivity.csv"
    results_df.to_csv(output_csv, index=False)

    print("=" * 80)
    print("STAGE 0 LAMBDA SENSITIVITY")
    print("=" * 80)
    print(f"Player filter       : {PLAYER_FILTER if PLAYER_FILTER is not None else 'ALL PLAYERS'}")
    print(f"Number of sequences : {len(sequences)}")
    print(f"Tested pairs        : {len(results_df)}")
    print()
    print("Unique best_sigma_comp values:")
    print(sorted(results_df["best_sigma_comp"].unique().tolist()))
    print()
    print("Unique parsimonious_sigma_comp values:")
    print(sorted(results_df["parsimonious_sigma_comp"].unique().tolist()))
    print()
    print(f"Saved CSV           : {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()