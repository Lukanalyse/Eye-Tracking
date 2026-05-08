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
from src.stage_0.stage_0 import Stage0Params
from src.stage_0.score_0 import (
    Score0Params,
    stage0_br_count_distribution,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / "resultats_stage_0"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Keep the search range moderate and interpretable
SIGMA_SCORE_GRID = np.round(np.arange(0.5, 5.5, 0.25), 2)

# Fixed Stage 0 map parameters already chosen
SIGMA_COMP_FIXED = 8.0
LAMBDA_STAT = 0.0
LAMBDA_BR = 1.0

PLAYER_FILTER: str | None = None

# Compromise weights for sigma_score
WEIGHT_EXPECTED = 0.50
WEIGHT_NONZERO = 0.35
WEIGHT_ENTROPY = 0.40

# Parsimony threshold on objective
PARSIMONY_OBJECTIVE_FRACTION = 0.90


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
# SCORE EVALUATION
# =============================================================================

def evaluate_sequence_score0(
    fixations: pd.DataFrame,
    game: int | str,
    stage0_params: Stage0Params,
    sigma_score: float,
) -> dict[str, float]:
    score_params = Score0Params(
        sigma_score=float(sigma_score),
    )

    result = stage0_br_count_distribution(
        fixations=fixations,
        game=game,
        stage0_params=stage0_params,
        score0_params=score_params,
    )

    summary = result["summary"]
    distribution = np.asarray(result["distribution"], dtype=float)

    prob_zero = float(distribution[0]) if len(distribution) > 0 else np.nan

    return {
        "n_transitions": float(len(result["transition_df"])),
        "expected_count": float(summary["expected_count"]),
        "variance": float(summary["variance"]),
        "mode": float(summary["mode"]),
        "median": float(summary["median"]),
        "prob_zero": prob_zero,
        "mean_p_t": float(summary["mean_p_t"]),
        "mean_score_entropy": float(summary["mean_score_entropy"]),
        "mean_distance_to_br": float(summary["mean_distance_to_br"]),
    }


def aggregate_score0_metrics(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_score: float,
    sigma_comp: float = SIGMA_COMP_FIXED,
    lambda_stat: float = LAMBDA_STAT,
    lambda_br: float = LAMBDA_BR,
) -> dict[str, Any]:
    stage0_params = Stage0Params(
        lambda_stat=float(lambda_stat),
        lambda_br=float(lambda_br),
        sigma_comp=float(sigma_comp),
    )

    per_sequence_rows: list[dict[str, Any]] = []

    for player_name, sheet_name, fixations in sequences:
        metrics = evaluate_sequence_score0(
            fixations=fixations,
            game=sheet_name,
            stage0_params=stage0_params,
            sigma_score=float(sigma_score),
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
            "mean_distance_to_br": np.nan,
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
        "mean_distance_to_br": float(per_sequence_df["mean_distance_to_br"].mean()),
    }

    return {"summary": summary, "per_sequence": per_sequence_df}


# =============================================================================
# GRID SEARCH
# =============================================================================

def run_sigma_score_grid_search(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_grid: np.ndarray = SIGMA_SCORE_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    per_sequence_frames: list[pd.DataFrame] = []

    for sigma_score in sigma_grid:
        result = aggregate_score0_metrics(
            sequences=sequences,
            sigma_score=float(sigma_score),
            sigma_comp=SIGMA_COMP_FIXED,
            lambda_stat=LAMBDA_STAT,
            lambda_br=LAMBDA_BR,
        )
        summary_rows.append(result["summary"])

        if not result["per_sequence"].empty:
            per_sequence_frames.append(result["per_sequence"])

    summary_df = pd.DataFrame(summary_rows).sort_values("sigma_score").reset_index(drop=True)

    if per_sequence_frames:
        per_sequence_df = pd.concat(per_sequence_frames, ignore_index=True)
    else:
        per_sequence_df = pd.DataFrame()

    return summary_df, per_sequence_df


def add_objective_columns(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()

    expected_count_max = float(df["mean_expected_count"].max()) if len(df) > 0 else 1.0
    if expected_count_max <= 0:
        df["norm_expected_count"] = 0.0
    else:
        df["norm_expected_count"] = df["mean_expected_count"] / expected_count_max

    df["nonzero_score"] = 1.0 - df["mean_prob_zero"]

    df["objective_score"] = (
        WEIGHT_EXPECTED * df["norm_expected_count"]
        + WEIGHT_NONZERO * df["nonzero_score"]
        - WEIGHT_ENTROPY * df["mean_score_entropy"]
    )
    return df


def compute_compromise_sigma_score(summary_df: pd.DataFrame) -> dict[str, float]:
    df = add_objective_columns(summary_df)
    df = df.dropna(subset=["objective_score"]).sort_values("sigma_score")

    if df.empty:
        return {
            "sigma_score": np.nan,
            "expected_count_max": np.nan,
            "normalized_expected": np.nan,
            "nonzero_score": np.nan,
            "mean_score_entropy": np.nan,
            "objective_score": np.nan,
        }

    best_idx = int(df["objective_score"].idxmax())
    chosen_row = df.loc[best_idx]

    return {
        "sigma_score": float(chosen_row["sigma_score"]),
        "expected_count_max": float(df["mean_expected_count"].max()),
        "normalized_expected": float(chosen_row["norm_expected_count"]),
        "nonzero_score": float(chosen_row["nonzero_score"]),
        "mean_score_entropy": float(chosen_row["mean_score_entropy"]),
        "objective_score": float(chosen_row["objective_score"]),
    }


def compute_sigma_score_90(summary_df: pd.DataFrame) -> dict[str, float]:
    """
    Choose the smallest sigma_score whose objective reaches 90% of the
    maximum objective.
    """
    df = add_objective_columns(summary_df)
    df = df.dropna(subset=["objective_score"]).sort_values("sigma_score")

    if df.empty:
        return {
            "sigma_score_90": np.nan,
            "objective_max": np.nan,
            "objective_threshold_90": np.nan,
            "objective_score": np.nan,
            "achieved_fraction": np.nan,
        }

    objective_max = float(df["objective_score"].max())
    threshold_90 = PARSIMONY_OBJECTIVE_FRACTION * objective_max

    eligible = df[df["objective_score"] >= threshold_90]

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
        "sigma_score_90": float(chosen_row["sigma_score"]),
        "objective_max": objective_max,
        "objective_threshold_90": threshold_90,
        "objective_score": float(chosen_row["objective_score"]),
        "achieved_fraction": achieved_fraction,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sigma_score_tradeoff(
    summary_df: pd.DataFrame,
    output_path: Path,
    chosen_sigma_score: float,
    sigma_score_90: float,
) -> None:
    df = add_objective_columns(summary_df)

    fig, axes = plt.subplots(4, 1, figsize=(9, 13), sharex=True)

    axes[0].plot(df["sigma_score"], df["mean_expected_count"], marker="o")
    axes[0].axvline(chosen_sigma_score, linestyle="--", linewidth=1.5, label="Best objective")
    axes[0].axvline(sigma_score_90, linestyle=":", linewidth=1.5, label="Sigma 90% objective")
    axes[0].set_ylabel("Mean expected BR count")
    axes[0].set_title(f"Stage 0 sigma_score compromise (sigma_comp={SIGMA_COMP_FIXED:.2f})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["sigma_score"], df["mean_median"], marker="o", label="Mean median")
    axes[1].plot(df["sigma_score"], df["mean_mode"], marker="s", label="Mean mode")
    axes[1].axvline(chosen_sigma_score, linestyle="--", linewidth=1.5)
    axes[1].axvline(sigma_score_90, linestyle=":", linewidth=1.5)
    axes[1].set_ylabel("Central tendency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["sigma_score"], df["mean_prob_zero"], marker="o", label="Mean P(N=0)")
    axes[2].plot(df["sigma_score"], df["mean_p_t"], marker="s", label="Mean local p_t")
    axes[2].plot(df["sigma_score"], df["mean_score_entropy"], marker="^", label="Mean score entropy")
    axes[2].axvline(chosen_sigma_score, linestyle="--", linewidth=1.5)
    axes[2].axvline(sigma_score_90, linestyle=":", linewidth=1.5)
    axes[2].set_ylabel("Score diagnostics")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(df["sigma_score"], df["norm_expected_count"], marker="o", label="Normalized expected count")
    axes[3].plot(df["sigma_score"], df["nonzero_score"], marker="^", label="1 - P(N=0)")
    axes[3].plot(df["sigma_score"], df["objective_score"], marker="s", label="Compromise objective")
    axes[3].axvline(chosen_sigma_score, linestyle="--", linewidth=1.5, label="Best objective")
    axes[3].axvline(sigma_score_90, linestyle=":", linewidth=1.5, label="Sigma 90% objective")
    axes[3].set_xlabel("sigma_score")
    axes[3].set_ylabel("Normalized objective")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    sequences = load_stage0_sequences(player_filter=PLAYER_FILTER)

    if not sequences:
        raise RuntimeError("No valid sequences found for Stage 0 sigma_score optimization.")

    summary_df, per_sequence_df = run_sigma_score_grid_search(
        sequences=sequences,
        sigma_grid=SIGMA_SCORE_GRID,
    )

    summary_df = add_objective_columns(summary_df)

    compromise_info = compute_compromise_sigma_score(summary_df)
    chosen_sigma_score = float(compromise_info["sigma_score"])
    chosen_row = summary_df.loc[summary_df["sigma_score"] == chosen_sigma_score].iloc[0]

    sigma90_info = compute_sigma_score_90(summary_df)
    sigma_score_90 = float(sigma90_info["sigma_score_90"])
    sigma90_row = summary_df.loc[summary_df["sigma_score"] == sigma_score_90].iloc[0]

    filter_suffix = PLAYER_FILTER if PLAYER_FILTER is not None else "all_players"
    summary_path = RESULTS_DIR / f"stage0_sigma_score_summary_{filter_suffix}.csv"
    per_sequence_path = RESULTS_DIR / f"stage0_sigma_score_per_sequence_{filter_suffix}.csv"
    figure_path = RESULTS_DIR / f"stage0_sigma_score_tradeoff_{filter_suffix}.png"

    summary_df["chosen_sigma_score"] = chosen_sigma_score
    summary_df["sigma_score_90"] = sigma_score_90

    summary_df.to_csv(summary_path, index=False)

    if not per_sequence_df.empty:
        per_sequence_df.to_csv(per_sequence_path, index=False)

    plot_sigma_score_tradeoff(
        summary_df=summary_df,
        output_path=figure_path,
        chosen_sigma_score=chosen_sigma_score,
        sigma_score_90=sigma_score_90,
    )

    print("=" * 80)
    print("STAGE 0 SIGMA_SCORE OPTIMIZATION")
    print("=" * 80)
    print(f"Player filter        : {PLAYER_FILTER if PLAYER_FILTER is not None else 'ALL PLAYERS'}")
    print(f"Number of sequences  : {len(sequences)}")
    print(f"Grid size            : {len(SIGMA_SCORE_GRID)}")
    print(f"sigma_comp fixed     : {SIGMA_COMP_FIXED:.3f}")
    print(f"lambda_stat          : {LAMBDA_STAT:.3f}")
    print(f"lambda_br            : {LAMBDA_BR:.3f}")
    print()
    print("Compromise rule")
    print(f"  weight_expected    = {WEIGHT_EXPECTED:.3f}")
    print(f"  weight_nonzero     = {WEIGHT_NONZERO:.3f}")
    print(f"  weight_entropy     = {WEIGHT_ENTROPY:.3f}")
    print(f"  objective 90% frac = {PARSIMONY_OBJECTIVE_FRACTION:.3f}")
    print(f"  expected_count_max = {compromise_info['expected_count_max']:.6f}")
    print()

    print("Best objective sigma_score")
    print(f"  sigma_score          = {chosen_row['sigma_score']:.3f}")
    print(f"  mean_expected_count  = {chosen_row['mean_expected_count']:.6f}")
    print(f"  normalized_expected  = {chosen_row['norm_expected_count']:.6f}")
    print(f"  nonzero_score        = {chosen_row['nonzero_score']:.6f}")
    print(f"  mean_variance        = {chosen_row['mean_variance']:.6f}")
    print(f"  mean_mode            = {chosen_row['mean_mode']:.6f}")
    print(f"  mean_median          = {chosen_row['mean_median']:.6f}")
    print(f"  mean_prob_zero       = {chosen_row['mean_prob_zero']:.6f}")
    print(f"  mean_p_t             = {chosen_row['mean_p_t']:.6f}")
    print(f"  mean_score_entropy   = {chosen_row['mean_score_entropy']:.6f}")
    print(f"  mean_distance_to_br  = {chosen_row['mean_distance_to_br']:.6f}")
    print(f"  objective_score      = {chosen_row['objective_score']:.6f}")
    print()

    print("Parsimonious sigma_score (90% objective)")
    print(f"  sigma_score_90       = {sigma90_row['sigma_score']:.3f}")
    print(f"  mean_expected_count  = {sigma90_row['mean_expected_count']:.6f}")
    print(f"  normalized_expected  = {sigma90_row['norm_expected_count']:.6f}")
    print(f"  nonzero_score        = {sigma90_row['nonzero_score']:.6f}")
    print(f"  mean_variance        = {sigma90_row['mean_variance']:.6f}")
    print(f"  mean_mode            = {sigma90_row['mean_mode']:.6f}")
    print(f"  mean_median          = {sigma90_row['mean_median']:.6f}")
    print(f"  mean_prob_zero       = {sigma90_row['mean_prob_zero']:.6f}")
    print(f"  mean_p_t             = {sigma90_row['mean_p_t']:.6f}")
    print(f"  mean_score_entropy   = {sigma90_row['mean_score_entropy']:.6f}")
    print(f"  mean_distance_to_br  = {sigma90_row['mean_distance_to_br']:.6f}")
    print(f"  objective_score      = {sigma90_row['objective_score']:.6f}")
    print(f"  achieved_fraction    = {sigma90_info['achieved_fraction']:.6f}")
    print()

    print(f"Saved summary CSV   : {summary_path}")
    print(f"Saved sequence CSV  : {per_sequence_path}")
    print(f"Saved figure        : {figure_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()