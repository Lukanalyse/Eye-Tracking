## OPTI SIGMA DISTRIBUTION


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

SIGMA_SCORE_GRID = np.round(np.arange(0.5, 15.5, 0.5), 2)

# Fixed Stage 0 parameters already chosen
SIGMA_COMP_FIXED = 8.0
LAMBDA_STAT = 0.0
LAMBDA_BR = 1.0

# Optional player filter
PLAYER_FILTER: str | None = None

# Stability rule:
# choose the smallest sigma_score reaching this fraction of the plateau
STABILITY_FRACTION = 0.90
PLATEAU_LAST_K = 5

USE_RELATIVE_OVERLAP = True


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
# SCORE EVALUATION
# =============================================================================

def evaluate_sequence_score0(
    fixations: pd.DataFrame,
    game: int | str,
    stage0_params: Stage0Params,
    sigma_score: float,
    use_relative_overlap: bool,
) -> dict[str, float]:
    score_params = Score0Params(
        sigma_score=float(sigma_score),
        use_relative_overlap=bool(use_relative_overlap),
    )

    result = stage0_br_count_distribution(
        fixations=fixations,
        game=game,
        stage0_params=stage0_params,
        score0_params=score_params,
    )

    summary = result["summary"]
    transition_df = result["transition_df"]

    mean_p_t = float(transition_df["p_t"].mean()) if not transition_df.empty else np.nan

    return {
        "n_transitions": float(len(transition_df)),
        "expected_count": float(summary["expected_count"]),
        "variance": float(summary["variance"]),
        "mode": float(summary["mode"]),
        "median": float(summary["median"]),
        "prob_zero": float(result["distribution"][0]),
        "mean_p_t": mean_p_t,
    }


def aggregate_score0_metrics(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_score: float,
    sigma_comp: float = SIGMA_COMP_FIXED,
    lambda_stat: float = LAMBDA_STAT,
    lambda_br: float = LAMBDA_BR,
    use_relative_overlap: bool = USE_RELATIVE_OVERLAP,
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
            use_relative_overlap=use_relative_overlap,
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
            use_relative_overlap=USE_RELATIVE_OVERLAP,
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


def compute_stable_sigma_score(summary_df: pd.DataFrame) -> dict[str, float]:
    """
    Choose the smallest sigma_score reaching STABILITY_FRACTION of the plateau
    of mean_expected_count.

    Plateau = average of the last PLATEAU_LAST_K points.
    """
    df = summary_df.dropna(subset=["mean_expected_count"]).copy().sort_values("sigma_score")

    if df.empty:
        return {
            "sigma_score": np.nan,
            "plateau_value": np.nan,
            "threshold_value": np.nan,
            "achieved_fraction": np.nan,
        }

    plateau_slice = df.tail(PLATEAU_LAST_K)
    plateau_value = float(plateau_slice["mean_expected_count"].mean())
    threshold_value = STABILITY_FRACTION * plateau_value

    eligible = df[df["mean_expected_count"] >= threshold_value]

    if eligible.empty:
        chosen_row = df.iloc[-1]
    else:
        chosen_row = eligible.iloc[0]

    achieved_fraction = float(chosen_row["mean_expected_count"]) / max(plateau_value, 1e-12)

    return {
        "sigma_score": float(chosen_row["sigma_score"]),
        "plateau_value": plateau_value,
        "threshold_value": threshold_value,
        "achieved_fraction": achieved_fraction,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_sigma_score_tradeoff(
    summary_df: pd.DataFrame,
    output_path: Path,
    stable_sigma_score: float,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(summary_df["sigma_score"], summary_df["mean_expected_count"], marker="o")
    axes[0].axvline(stable_sigma_score, linestyle="--", linewidth=1.5, label="Stable sigma_score")
    axes[0].set_ylabel("Mean expected BR count")
    axes[0].set_title(
        f"Stage 0 sigma_score stability (sigma_comp={SIGMA_COMP_FIXED:.2f}, "
        f"lambda_stat={LAMBDA_STAT:.2f}, lambda_br={LAMBDA_BR:.2f})"
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(summary_df["sigma_score"], summary_df["mean_median"], marker="o", label="Mean median")
    axes[1].plot(summary_df["sigma_score"], summary_df["mean_mode"], marker="s", label="Mean mode")
    axes[1].axvline(stable_sigma_score, linestyle="--", linewidth=1.5)
    axes[1].set_ylabel("Central tendency")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(summary_df["sigma_score"], summary_df["mean_prob_zero"], marker="o", label="Mean P(N=0)")
    axes[2].plot(summary_df["sigma_score"], summary_df["mean_p_t"], marker="s", label="Mean local p_t")
    axes[2].axvline(stable_sigma_score, linestyle="--", linewidth=1.5)
    axes[2].set_xlabel("sigma_score")
    axes[2].set_ylabel("Score diagnostics")
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
        raise RuntimeError("No valid sequences found for Stage 0 sigma_score optimization.")

    summary_df, per_sequence_df = run_sigma_score_grid_search(
        sequences=sequences,
        sigma_grid=SIGMA_SCORE_GRID,
    )

    stable_info = compute_stable_sigma_score(summary_df)
    stable_sigma_score = float(stable_info["sigma_score"])
    stable_row = summary_df.loc[summary_df["sigma_score"] == stable_sigma_score].iloc[0]

    summary_df["stable_sigma_score"] = stable_sigma_score

    filter_suffix = PLAYER_FILTER if PLAYER_FILTER is not None else "all_players"
    summary_path = RESULTS_DIR / f"stage0_sigma_score_summary_{filter_suffix}.csv"
    per_sequence_path = RESULTS_DIR / f"stage0_sigma_score_per_sequence_{filter_suffix}.csv"
    figure_path = RESULTS_DIR / f"stage0_sigma_score_tradeoff_{filter_suffix}.png"

    summary_df.to_csv(summary_path, index=False)

    if not per_sequence_df.empty:
        per_sequence_df.to_csv(per_sequence_path, index=False)

    plot_sigma_score_tradeoff(
        summary_df=summary_df,
        output_path=figure_path,
        stable_sigma_score=stable_sigma_score,
    )

    print("=" * 80)
    print("STAGE 0 SIGMA_SCORE OPTIMIZATION")
    print("=" * 80)
    print(f"Player filter       : {PLAYER_FILTER if PLAYER_FILTER is not None else 'ALL PLAYERS'}")
    print(f"Number of sequences : {len(sequences)}")
    print(f"Grid size           : {len(SIGMA_SCORE_GRID)}")
    print(f"sigma_comp fixed    : {SIGMA_COMP_FIXED:.3f}")
    print(f"lambda_stat         : {LAMBDA_STAT:.3f}")
    print(f"lambda_br           : {LAMBDA_BR:.3f}")
    print(f"use_relative_overlap: {USE_RELATIVE_OVERLAP}")
    print()
    print("Stability rule")
    print(f"  stability fraction = {STABILITY_FRACTION:.3f}")
    print(f"  plateau last k     = {PLATEAU_LAST_K}")
    print()

    print("Stable sigma_score")
    print(f"  sigma_score        = {stable_row['sigma_score']:.3f}")
    print(f"  mean_expected_count= {stable_row['mean_expected_count']:.6f}")
    print(f"  mean_variance      = {stable_row['mean_variance']:.6f}")
    print(f"  mean_mode          = {stable_row['mean_mode']:.6f}")
    print(f"  mean_median        = {stable_row['mean_median']:.6f}")
    print(f"  mean_prob_zero     = {stable_row['mean_prob_zero']:.6f}")
    print(f"  mean_p_t           = {stable_row['mean_p_t']:.6f}")
    print(f"  achieved fraction  = {stable_info['achieved_fraction']:.6f}")
    print()

    print(f"Saved summary CSV   : {summary_path}")
    print(f"Saved sequence CSV  : {per_sequence_path}")
    print(f"Saved figure        : {figure_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()