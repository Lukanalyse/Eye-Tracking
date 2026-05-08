from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from src.pages.page_1 import (
    discover_player_files,
    is_supported_game_sheet,
    load_workbook_sheets,
)
from src.stage_1.stage_1 import Stage1Params, stage_1_step


# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path(__file__).resolve().parents[2] / "resultats_stage_1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2D grid for joint calibration
SIGMA_COMP_GRID = np.round(np.arange(2.0, 12.5, 0.5), 2)
SIGMA_SPACE_GRID = np.round(np.arange(0.5, 4.5, 0.5), 2)

# Stage 1 map structure
LAMBDA_STAT = 0.0
LAMBDA_BR = 1.0

PLAYER_FILTER: str | None = None

# Regularization weights
LAMBDA_ENTROPY_PENALTY = 0.15
LAMBDA_RANK_PENALTY = 0.05

# Parsimony rule:
# keep all pairs reaching this fraction of maximal LL gain,
# then select the one with the smallest diffusion budget
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


def evaluate_sequence_stage1(
    fixations: pd.DataFrame,
    game: int | str,
    params: Stage1Params,
) -> dict[str, float]:
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

        step = stage_1_step(
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


def aggregate_stage1_metrics(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_comp: float,
    sigma_space: float,
) -> dict[str, Any]:
    params = Stage1Params(
        lambda_stat=LAMBDA_STAT,
        lambda_br=LAMBDA_BR,
        sigma_comp=float(sigma_comp),
        sigma_space=float(sigma_space),
    )

    total_n = 0.0
    weighted_ll = 0.0
    weighted_entropy = 0.0
    weighted_rank = 0.0
    weighted_top1 = 0.0
    weighted_top5 = 0.0

    per_sequence_rows: list[dict[str, Any]] = []

    for player_name, sheet_name, fixations in sequences:
        metrics = evaluate_sequence_stage1(
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
                "sigma_space": float(sigma_space),
                **metrics,
            }
        )

    if total_n <= 0:
        return {
            "summary": {
                "sigma_comp": float(sigma_comp),
                "sigma_space": float(sigma_space),
                "n_predictions": 0.0,
                "mean_log_likelihood": np.nan,
                "mean_entropy": np.nan,
                "mean_rank": np.nan,
                "top1_accuracy": np.nan,
                "top5_accuracy": np.nan,
                "normalized_rank": np.nan,
                "objective": np.nan,
                "diffusion_budget": np.nan,
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

    diffusion_budget = float(sigma_comp + sigma_space)

    summary = {
        "sigma_comp": float(sigma_comp),
        "sigma_space": float(sigma_space),
        "n_predictions": float(total_n),
        "mean_log_likelihood": float(mean_ll),
        "mean_entropy": float(mean_entropy),
        "mean_rank": float(mean_rank),
        "top1_accuracy": float(top1),
        "top5_accuracy": float(top5),
        "normalized_rank": float(normalized_rank),
        "objective": float(objective),
        "diffusion_budget": diffusion_budget,
    }

    return {
        "summary": summary,
        "per_sequence": pd.DataFrame(per_sequence_rows),
    }


# =============================================================================
# 2D GRID SEARCH
# =============================================================================

def run_joint_grid_search(
    sequences: list[tuple[str, str, pd.DataFrame]],
    sigma_comp_grid: np.ndarray = SIGMA_COMP_GRID,
    sigma_space_grid: np.ndarray = SIGMA_SPACE_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    per_sequence_frames: list[pd.DataFrame] = []

    total_pairs = len(sigma_comp_grid) * len(sigma_space_grid)
    counter = 0

    for sigma_comp in sigma_comp_grid:
        for sigma_space in sigma_space_grid:
            counter += 1
            print(
                f"Evaluating pair {counter}/{total_pairs}: "
                f"sigma_comp={sigma_comp:.2f}, sigma_space={sigma_space:.2f}"
            )

            result = aggregate_stage1_metrics(
                sequences=sequences,
                sigma_comp=float(sigma_comp),
                sigma_space=float(sigma_space),
            )
            summary_rows.append(result["summary"])

            if not result["per_sequence"].empty:
                per_sequence_frames.append(result["per_sequence"])

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["sigma_comp", "sigma_space"]
    ).reset_index(drop=True)

    if per_sequence_frames:
        per_sequence_df = pd.concat(per_sequence_frames, ignore_index=True)
    else:
        per_sequence_df = pd.DataFrame()

    return summary_df, per_sequence_df


def compute_parsimonious_pair(summary_df: pd.DataFrame) -> dict[str, float]:
    """
    Step 1: keep pairs reaching PARSIMONY_GAIN_FRACTION of the maximal LL gain.
    Step 2: among them, choose the smallest diffusion budget:
            sigma_comp + sigma_space
    Step 3: tie-break with the best objective.
    """
    df = summary_df.dropna(subset=["mean_log_likelihood"]).copy()

    ll_min = float(df["mean_log_likelihood"].min())
    ll_max = float(df["mean_log_likelihood"].max())
    total_gain = ll_max - ll_min

    if total_gain <= 0:
        chosen_row = df.iloc[0]
        return {
            "sigma_comp": float(chosen_row["sigma_comp"]),
            "sigma_space": float(chosen_row["sigma_space"]),
            "threshold_ll": ll_min,
            "ll_min": ll_min,
            "ll_max": ll_max,
            "gain_fraction": 0.0,
        }

    threshold_ll = ll_min + PARSIMONY_GAIN_FRACTION * total_gain
    eligible = df[df["mean_log_likelihood"] >= threshold_ll].copy()

    if eligible.empty:
        chosen_row = df.loc[df["mean_log_likelihood"].idxmax()]
    else:
        eligible = eligible.sort_values(
            by=["diffusion_budget", "objective"],
            ascending=[True, False],
        )
        chosen_row = eligible.iloc[0]

    achieved_fraction = (float(chosen_row["mean_log_likelihood"]) - ll_min) / total_gain

    return {
        "sigma_comp": float(chosen_row["sigma_comp"]),
        "sigma_space": float(chosen_row["sigma_space"]),
        "threshold_ll": float(threshold_ll),
        "ll_min": ll_min,
        "ll_max": ll_max,
        "gain_fraction": float(achieved_fraction),
    }


# =============================================================================
# PLOTS
# =============================================================================

def plot_joint_surface(
    summary_df: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
    best_pair: tuple[float, float] | None = None,
    parsimonious_pair: tuple[float, float] | None = None,
) -> None:
    pivot_df = summary_df.pivot(
        index="sigma_comp",
        columns="sigma_space",
        values=value_column,
    ).sort_index().sort_index(axis=1)

    x_vals = pivot_df.columns.to_numpy(dtype=float)
    y_vals = pivot_df.index.to_numpy(dtype=float)
    z_vals = pivot_df.to_numpy(dtype=float)

    X, Y = np.meshgrid(x_vals, y_vals)

    fig, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contourf(X, Y, z_vals, levels=20, cmap="viridis")
    cbar = plt.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(value_column, rotation=90)

    contour_lines = ax.contour(X, Y, z_vals, levels=10, colors="white", linewidths=0.5, alpha=0.6)
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    if best_pair is not None:
        ax.scatter(
            best_pair[1],   # x = sigma_space
            best_pair[0],   # y = sigma_comp
            s=120,
            marker="*",
            color="red",
            edgecolor="black",
            linewidth=0.8,
            label="Best pair",
            zorder=5,
        )

    if parsimonious_pair is not None:
        ax.scatter(
            parsimonious_pair[1],   # x = sigma_space
            parsimonious_pair[0],   # y = sigma_comp
            s=90,
            marker="o",
            color="orange",
            edgecolor="black",
            linewidth=0.8,
            label="Parsimonious pair",
            zorder=5,
        )

    ax.set_xlabel("sigma_space")
    ax.set_ylabel("sigma_comp")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_joint_surfaces(
    summary_df: pd.DataFrame,
    output_dir: Path,
    best_pair: tuple[float, float],
    parsimonious_pair: tuple[float, float],
) -> None:
    plots = [
        ("mean_log_likelihood", "Stage 1 joint optimization - mean log-likelihood"),
        ("mean_entropy", "Stage 1 joint optimization - mean entropy"),
        ("objective", "Stage 1 joint optimization - objective"),
    ]

    for metric_name, metric_title in plots:
        output_path = output_dir / f"{metric_name}_surface.png"
        plot_joint_surface(
            summary_df=summary_df,
            value_column=metric_name,
            title=metric_title,
            output_path=output_path,
            best_pair=best_pair,
            parsimonious_pair=parsimonious_pair,
        )

def plot_joint_surface_3d(
    summary_df: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
    best_pair: tuple[float, float] | None = None,
    parsimonious_pair: tuple[float, float] | None = None,
) -> None:
    pivot_df = summary_df.pivot(
        index="sigma_comp",
        columns="sigma_space",
        values=value_column,
    ).sort_index().sort_index(axis=1)

    x_vals = pivot_df.columns.to_numpy(dtype=float)   # sigma_space
    y_vals = pivot_df.index.to_numpy(dtype=float)     # sigma_comp
    z_vals = pivot_df.to_numpy(dtype=float)

    X, Y = np.meshgrid(x_vals, y_vals)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        X,
        Y,
        z_vals,
        cmap="viridis",
        edgecolor="k",
        linewidth=0.25,
        alpha=0.9,
    )

    cbar = fig.colorbar(surface, ax=ax, shrink=0.7, aspect=18)
    cbar.set_label(value_column)

    if best_pair is not None:
        best_sigma_comp, best_sigma_space = best_pair
        best_z = summary_df[
            (summary_df["sigma_comp"] == best_sigma_comp)
            & (summary_df["sigma_space"] == best_sigma_space)
        ][value_column].iloc[0]

        ax.scatter(
            best_sigma_space,
            best_sigma_comp,
            best_z,
            color="red",
            s=120,
            marker="*",
            edgecolor="black",
            label="Best pair",
            zorder=10,
        )

    if parsimonious_pair is not None:
        pars_sigma_comp, pars_sigma_space = parsimonious_pair
        pars_z = summary_df[
            (summary_df["sigma_comp"] == pars_sigma_comp)
            & (summary_df["sigma_space"] == pars_sigma_space)
        ][value_column].iloc[0]

        ax.scatter(
            pars_sigma_space,
            pars_sigma_comp,
            pars_z,
            color="orange",
            s=80,
            marker="o",
            edgecolor="black",
            label="Parsimonious pair",
            zorder=10,
        )

    ax.set_xlabel("sigma_space")
    ax.set_ylabel("sigma_comp")
    ax.set_zlabel(value_column)
    ax.set_title(title)
    ax.view_init(elev=28, azim=-130)
    ax.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_joint_surfaces_3d(
    summary_df: pd.DataFrame,
    output_dir: Path,
    best_pair: tuple[float, float],
    parsimonious_pair: tuple[float, float],
) -> None:
    plots = [
        ("mean_log_likelihood", "Stage 1 joint optimization - mean log-likelihood (3D)"),
        ("objective", "Stage 1 joint optimization - objective (3D)"),
        ("mean_entropy", "Stage 1 joint optimization - mean entropy (3D)"),
    ]

    for metric_name, metric_title in plots:
        output_path = output_dir / f"{metric_name}_surface_3d.png"
        plot_joint_surface_3d(
            summary_df=summary_df,
            value_column=metric_name,
            title=metric_title,
            output_path=output_path,
            best_pair=best_pair,
            parsimonious_pair=parsimonious_pair,
        )
# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    sequences = load_stage1_sequences(player_filter=PLAYER_FILTER)

    if not sequences:
        raise RuntimeError("No valid sequences found for Stage 1 joint optimization.")

    summary_df, per_sequence_df = run_joint_grid_search(
        sequences=sequences,
        sigma_comp_grid=SIGMA_COMP_GRID,
        sigma_space_grid=SIGMA_SPACE_GRID,
    )

    best_idx = int(summary_df["objective"].idxmax())
    best_row = summary_df.loc[best_idx]

    parsimonious_info = compute_parsimonious_pair(summary_df)
    parsimonious_row = summary_df[
        (summary_df["sigma_comp"] == parsimonious_info["sigma_comp"])
        & (summary_df["sigma_space"] == parsimonious_info["sigma_space"])
    ].iloc[0]

    best_pair = (float(best_row["sigma_comp"]), float(best_row["sigma_space"]))
    parsimonious_pair = (
        float(parsimonious_row["sigma_comp"]),
        float(parsimonious_row["sigma_space"]),
    )

    filter_suffix = PLAYER_FILTER if PLAYER_FILTER is not None else "all_players"

    summary_path = RESULTS_DIR / f"stage1_joint_summary_{filter_suffix}.csv"
    per_sequence_path = RESULTS_DIR / f"stage1_joint_per_sequence_{filter_suffix}.csv"

    summary_df.to_csv(summary_path, index=False)
    if not per_sequence_df.empty:
        per_sequence_df.to_csv(per_sequence_path, index=False)

    save_joint_surfaces(
        summary_df=summary_df,
        output_dir=RESULTS_DIR,
        best_pair=best_pair,
        parsimonious_pair=parsimonious_pair,
    )
    save_joint_surfaces_3d(
        summary_df=summary_df,
        output_dir=RESULTS_DIR,
        best_pair=best_pair,
        parsimonious_pair=parsimonious_pair,
    )

    print("=" * 80)
    print("STAGE 1 JOINT OPTIMIZATION")
    print("=" * 80)
    print(f"Player filter        : {PLAYER_FILTER if PLAYER_FILTER is not None else 'ALL PLAYERS'}")
    print(f"Number of sequences  : {len(sequences)}")
    print(f"Grid size sigma_comp : {len(SIGMA_COMP_GRID)}")
    print(f"Grid size sigma_space: {len(SIGMA_SPACE_GRID)}")
    print(f"Total pairs          : {len(SIGMA_COMP_GRID) * len(SIGMA_SPACE_GRID)}")
    print(f"lambda_stat          : {LAMBDA_STAT:.3f}")
    print(f"lambda_br            : {LAMBDA_BR:.3f}")
    print()
    print("Penalty weights")
    print(f"  lambda_entropy_penalty = {LAMBDA_ENTROPY_PENALTY:.3f}")
    print(f"  lambda_rank_penalty    = {LAMBDA_RANK_PENALTY:.3f}")
    print(f"  parsimony gain fraction= {PARSIMONY_GAIN_FRACTION:.3f}")
    print()

    print("Best pair (objective)")
    print(f"  sigma_comp          = {best_row['sigma_comp']:.3f}")
    print(f"  sigma_space         = {best_row['sigma_space']:.3f}")
    print(f"  mean_log_likelihood = {best_row['mean_log_likelihood']:.6f}")
    print(f"  mean_entropy        = {best_row['mean_entropy']:.6f}")
    print(f"  mean_rank           = {best_row['mean_rank']:.6f}")
    print(f"  top1_accuracy       = {best_row['top1_accuracy']:.6f}")
    print(f"  top5_accuracy       = {best_row['top5_accuracy']:.6f}")
    print(f"  objective           = {best_row['objective']:.6f}")
    print(f"  diffusion_budget    = {best_row['diffusion_budget']:.6f}")
    print()

    print("Parsimonious pair")
    print(f"  sigma_comp          = {parsimonious_row['sigma_comp']:.3f}")
    print(f"  sigma_space         = {parsimonious_row['sigma_space']:.3f}")
    print(f"  mean_log_likelihood = {parsimonious_row['mean_log_likelihood']:.6f}")
    print(f"  mean_entropy        = {parsimonious_row['mean_entropy']:.6f}")
    print(f"  mean_rank           = {parsimonious_row['mean_rank']:.6f}")
    print(f"  top1_accuracy       = {parsimonious_row['top1_accuracy']:.6f}")
    print(f"  top5_accuracy       = {parsimonious_row['top5_accuracy']:.6f}")
    print(f"  objective           = {parsimonious_row['objective']:.6f}")
    print(f"  diffusion_budget    = {parsimonious_row['diffusion_budget']:.6f}")
    print(f"  achieved gain frac  = {parsimonious_info['gain_fraction']:.6f}")
    print()

    print(f"Saved summary CSV      : {summary_path}")
    print(f"Saved sequence CSV     : {per_sequence_path}")
    print(f"Saved surface plots in : {RESULTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()