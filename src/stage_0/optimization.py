## OPTI SIGMA COMPUT


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

# Grid of candidate sigma_comp values for Stage 0
SIGMA_COMP_GRID = np.round(np.arange(0.5, 15.5, 0.5), 2)

# Fixed Stage 0 weights used for sigma_comp identification
# We remove the uniform baseline here so sigma_comp is identified from the BR map itself.
LAMBDA_STAT = 0.0
LAMBDA_BR = 1.0

# Optional player filter
# Set to None to use all players
PLAYER_FILTER: str | None = None

# Compromise objective weights
LAMBDA_ENTROPY_PENALTY = 0.15
LAMBDA_RANK_PENALTY = 0.05

# Parsimonious rule:
# choose the smallest sigma_comp reaching this fraction of the maximal LL gain
PARSIMONY_GAIN_FRACTION = 0.90


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
# STAGE 0 METRICS
# =============================================================================

def normalized_entropy(probabilities: np.ndarray) -> float:
    """
    Entropy normalized to [0, 1].
    0 = concentrated distribution
    1 = perfectly uniform distribution
    """
    q = np.asarray(probabilities, dtype=float)
    q = np.clip(q, 1e-12, 1.0)
    entropy = -float(np.sum(q * np.log(q)))
    max_entropy = float(np.log(len(q)))
    return float(entropy / max_entropy)


def rank_of_observed_aoi(probabilities: np.ndarray, observed_aoi: int) -> int:
    """
    Rank of the observed AOI in descending probability order.
    Rank 1 = most probable AOI.
    """
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
    """
    Evaluate Stage 0 on one sequence using next-fixation prediction.
    """
    clean = validate_fixations(fixations)

    if len(clean) < 2:
        return {
            "n_predictions": 0.0,
            "mean_log_likelihood": np.nan,
            "mean_entropy": np.nan,
            "mean_rank": np.nan,
            "top1_accuracy": np.nan,
            "top5_accuracy": np.nan,
        }

    log_likelihoods: list[float] = []
    entropies: list[float] = []
    ranks: list[int] = []
    top1_hits: list[int] = []
    top5_hits: list[int] = []

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

        rank = rank_of_observed_aoi(q_t, x_next)
        ranks.append(rank)
        top1_hits.append(int(rank == 1))
        top5_hits.append(int(rank <= 5))

    return {
        "n_predictions": float(len(log_likelihoods)),
        "mean_log_likelihood": float(np.mean(log_likelihoods)),
        "mean_entropy": float(np.mean(entropies)),
        "mean_rank": float(np.mean(ranks)),
        "top1_accuracy": float(np.mean(top1_hits)),
        "top5_accuracy": float(np.mean(top5_hits)),
    }


def aggregate_stage0_metrics(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_comp: float,
    lambda_stat: float = LAMBDA_STAT,
    lambda_br: float = LAMBDA_BR,
) -> dict[str, Any]:
    """
    Evaluate one sigma_comp across all sequences.
    Aggregation is weighted by number of predictive steps.
    """
    params = Stage0Params(
        lambda_stat=lambda_stat,
        lambda_br=lambda_br,
        sigma_comp=float(sigma_comp),
    )

    total_n = 0.0
    weighted_ll = 0.0
    weighted_entropy = 0.0
    weighted_rank = 0.0
    weighted_top1 = 0.0
    weighted_top5 = 0.0

    per_sequence_rows: list[dict[str, Any]] = []

    for player_name, sheet_name, fixations in sequences:
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
        weighted_top1 += n_pred * float(metrics["top1_accuracy"])
        weighted_top5 += n_pred * float(metrics["top5_accuracy"])

        per_sequence_rows.append(
            {
                "player": player_name,
                "game": sheet_name,
                "sigma_comp": float(sigma_comp),
                **metrics,
            }
        )

    if total_n <= 0:
        return {
            "summary": {
                "sigma_comp": float(sigma_comp),
                "n_predictions": 0.0,
                "mean_log_likelihood": np.nan,
                "mean_entropy": np.nan,
                "mean_rank": np.nan,
                "top1_accuracy": np.nan,
                "top5_accuracy": np.nan,
                "normalized_rank": np.nan,
                "objective": np.nan,
            },
            "per_sequence": pd.DataFrame(per_sequence_rows),
        }

    mean_ll = weighted_ll / total_n
    mean_entropy = weighted_entropy / total_n
    mean_rank = weighted_rank / total_n
    top1 = weighted_top1 / total_n
    top5 = weighted_top5 / total_n

    normalized_rank = (mean_rank - 1.0) / 99.0
    normalized_rank = float(np.clip(normalized_rank, 0.0, 1.0))

    objective = (
        float(mean_ll)
        - LAMBDA_ENTROPY_PENALTY * float(mean_entropy)
        - LAMBDA_RANK_PENALTY * float(normalized_rank)
    )

    summary = {
        "sigma_comp": float(sigma_comp),
        "n_predictions": float(total_n),
        "mean_log_likelihood": float(mean_ll),
        "mean_entropy": float(mean_entropy),
        "mean_rank": float(mean_rank),
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "normalized_rank": float(normalized_rank),
        "objective": float(objective),
    }

    return {
        "summary": summary,
        "per_sequence": pd.DataFrame(per_sequence_rows),
    }


# =============================================================================
# GRID SEARCH
# =============================================================================

def run_sigma_comp_grid_search(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_grid: np.ndarray = SIGMA_COMP_GRID,
    lambda_stat: float = LAMBDA_STAT,
    lambda_br: float = LAMBDA_BR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    per_sequence_frames: list[pd.DataFrame] = []

    for sigma in sigma_grid:
        result = aggregate_stage0_metrics(
            sequences=sequences,
            sigma_comp=float(sigma),
            lambda_stat=lambda_stat,
            lambda_br=lambda_br,
        )
        summary_rows.append(result["summary"])

        per_sequence_df = result["per_sequence"]
        if not per_sequence_df.empty:
            per_sequence_frames.append(per_sequence_df)

    summary_df = pd.DataFrame(summary_rows).sort_values("sigma_comp").reset_index(drop=True)

    if per_sequence_frames:
        per_sequence_df = pd.concat(per_sequence_frames, ignore_index=True)
    else:
        per_sequence_df = pd.DataFrame()

    return summary_df, per_sequence_df


def compute_parsimonious_sigma(summary_df: pd.DataFrame) -> dict[str, float]:
    """
    Choose the smallest sigma_comp reaching PARSIMONY_GAIN_FRACTION
    of the maximal log-likelihood gain.
    """
    df = summary_df.dropna(subset=["mean_log_likelihood"]).copy().sort_values("sigma_comp")

    ll_min = float(df["mean_log_likelihood"].min())
    ll_max = float(df["mean_log_likelihood"].max())
    total_gain = ll_max - ll_min

    if total_gain <= 0:
        chosen_row = df.iloc[0]
        return {
            "sigma_comp": float(chosen_row["sigma_comp"]),
            "threshold_ll": ll_min,
            "ll_min": ll_min,
            "ll_max": ll_max,
            "gain_fraction": 0.0,
        }

    threshold_ll = ll_min + PARSIMONY_GAIN_FRACTION * total_gain
    eligible = df[df["mean_log_likelihood"] >= threshold_ll]

    if eligible.empty:
        chosen_row = df.iloc[df["mean_log_likelihood"].idxmax()]
    else:
        chosen_row = eligible.iloc[0]

    achieved_fraction = (float(chosen_row["mean_log_likelihood"]) - ll_min) / total_gain

    return {
        "sigma_comp": float(chosen_row["sigma_comp"]),
        "threshold_ll": float(threshold_ll),
        "ll_min": ll_min,
        "ll_max": ll_max,
        "gain_fraction": float(achieved_fraction),
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sigma_comp_tradeoff(
    summary_df: pd.DataFrame,
    output_path: Path,
    best_objective_sigma: float,
    parsimonious_sigma: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(summary_df["sigma_comp"], summary_df["mean_log_likelihood"], marker="o")
    axes[0].axvline(best_objective_sigma, linestyle="--", linewidth=1.2, label="Best objective")
    axes[0].axvline(parsimonious_sigma, linestyle=":", linewidth=1.5, label="Parsimonious 90%")
    axes[0].set_ylabel("Mean log-likelihood")
    axes[0].set_title(
        f"Stage 0 sigma_comp optimization (lambda_stat={LAMBDA_STAT:.2f}, lambda_br={LAMBDA_BR:.2f})"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        summary_df["sigma_comp"],
        summary_df["mean_entropy"],
        marker="o",
        label="Normalized entropy",
    )
    axes[1].plot(
        summary_df["sigma_comp"],
        summary_df["normalized_rank"],
        marker="s",
        label="Normalized rank",
    )
    axes[1].axvline(best_objective_sigma, linestyle="--", linewidth=1.2)
    axes[1].axvline(parsimonious_sigma, linestyle=":", linewidth=1.5)
    axes[1].set_ylabel("Penalty diagnostics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(summary_df["sigma_comp"], summary_df["objective"], marker="o")
    axes[2].axvline(best_objective_sigma, linestyle="--", linewidth=1.2, label="Best objective")
    axes[2].axvline(parsimonious_sigma, linestyle=":", linewidth=1.5, label="Parsimonious 90%")
    axes[2].set_xlabel("sigma_comp")
    axes[2].set_ylabel("Compromise objective")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    sequences = load_stage0_sequences(player_filter=PLAYER_FILTER)

    if not sequences:
        raise RuntimeError("No valid sequences found for Stage 0 optimization.")

    summary_df, per_sequence_df = run_sigma_comp_grid_search(
        sequences=sequences,
        sigma_grid=SIGMA_COMP_GRID,
        lambda_stat=LAMBDA_STAT,
        lambda_br=LAMBDA_BR,
    )

    best_idx = int(summary_df["objective"].idxmax())
    best_row = summary_df.loc[best_idx]

    parsimonious_info = compute_parsimonious_sigma(summary_df)
    parsimonious_sigma = float(parsimonious_info["sigma_comp"])
    parsimonious_row = summary_df.loc[summary_df["sigma_comp"] == parsimonious_sigma].iloc[0]

    summary_df["best_objective_sigma"] = float(best_row["sigma_comp"])
    summary_df["parsimonious_sigma_95"] = parsimonious_sigma

    filter_suffix = PLAYER_FILTER if PLAYER_FILTER is not None else "all_players"
    summary_path = RESULTS_DIR / f"stage0_sigma_comp_summary_{filter_suffix}.csv"
    per_sequence_path = RESULTS_DIR / f"stage0_sigma_comp_per_sequence_{filter_suffix}.csv"
    figure_path = RESULTS_DIR / f"stage0_sigma_comp_tradeoff_{filter_suffix}.png"

    summary_df.to_csv(summary_path, index=False)

    if not per_sequence_df.empty:
        per_sequence_df.to_csv(per_sequence_path, index=False)

    plot_sigma_comp_tradeoff(
        summary_df=summary_df,
        output_path=figure_path,
        best_objective_sigma=float(best_row["sigma_comp"]),
        parsimonious_sigma=parsimonious_sigma,
    )

    print("=" * 80)
    print("STAGE 0 SIGMA_COMP OPTIMIZATION")
    print("=" * 80)
    print(f"Player filter       : {PLAYER_FILTER if PLAYER_FILTER is not None else 'ALL PLAYERS'}")
    print(f"Number of sequences : {len(sequences)}")
    print(f"Grid size           : {len(SIGMA_COMP_GRID)}")
    print(f"lambda_stat         : {LAMBDA_STAT:.3f}")
    print(f"lambda_br           : {LAMBDA_BR:.3f}")
    print()
    print("Penalty weights")
    print(f"  lambda_entropy_penalty = {LAMBDA_ENTROPY_PENALTY:.3f}")
    print(f"  lambda_rank_penalty    = {LAMBDA_RANK_PENALTY:.3f}")
    print(f"  parsimony gain fraction= {PARSIMONY_GAIN_FRACTION:.3f}")
    print()

    print("Best compromise sigma_comp (objective)")
    print(f"  sigma_comp          = {best_row['sigma_comp']:.3f}")
    print(f"  mean_log_likelihood = {best_row['mean_log_likelihood']:.6f}")
    print(f"  mean_entropy        = {best_row['mean_entropy']:.6f}")
    print(f"  mean_rank           = {best_row['mean_rank']:.6f}")
    print(f"  top1_accuracy       = {best_row['top1_accuracy']:.6f}")
    print(f"  top5_accuracy       = {best_row['top5_accuracy']:.6f}")
    print(f"  objective           = {best_row['objective']:.6f}")
    print()

    print("Parsimonious sigma_comp (90% of maximal LL gain)")
    print(f"  sigma_comp          = {parsimonious_row['sigma_comp']:.3f}")
    print(f"  mean_log_likelihood = {parsimonious_row['mean_log_likelihood']:.6f}")
    print(f"  mean_entropy        = {parsimonious_row['mean_entropy']:.6f}")
    print(f"  mean_rank           = {parsimonious_row['mean_rank']:.6f}")
    print(f"  top1_accuracy       = {parsimonious_row['top1_accuracy']:.6f}")
    print(f"  top5_accuracy       = {parsimonious_row['top5_accuracy']:.6f}")
    print(f"  objective           = {parsimonious_row['objective']:.6f}")
    print(f"  achieved gain frac  = {parsimonious_info['gain_fraction']:.6f}")
    print()

    print(f"Saved summary CSV   : {summary_path}")
    print(f"Saved sequence CSV  : {per_sequence_path}")
    print(f"Saved figure        : {figure_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()