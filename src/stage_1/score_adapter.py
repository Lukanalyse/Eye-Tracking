from __future__ import annotations

import numpy as np
import pandas as pd

from src.pages.page_1 import AOI_COUNT
from src.stage_0.score_0 import (
    normalize_probability,
    relative_overlap_score,
    poisson_binomial_distribution,
    summarize_br_distribution,
)
from src.stage_1.stage_1 import Stage1Params, stage_1_step


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

    clean = clean[(clean["AOI"] >= 1) & (clean["AOI"] <= AOI_COUNT)]
    clean = clean[clean["Time"] > 0].reset_index(drop=True)

    return clean


def br_region_weights(y: np.ndarray, target: float, sigma_score: float) -> np.ndarray:
    sigma_safe = max(float(sigma_score), 1e-9)
    raw = np.exp(-((y - target) ** 2) / (2.0 * sigma_safe**2))
    return normalize_probability(raw)


def compute_transition_br_probabilities_stage1(
    fixations: pd.DataFrame,
    game: int | str,
    stage1_params: Stage1Params,
    sigma_score: float,
    use_relative_overlap: bool,
) -> pd.DataFrame:
    clean = validate_fixations(fixations)

    if len(clean) < 2:
        return pd.DataFrame(
            columns=["t", "x_t", "x_t_plus_1", "T_g", "p_t", "q_next_exact", "q_zone_overlap", "q_max"]
        )

    y = np.arange(1, AOI_COUNT + 1, dtype=float)
    rows = []

    for t in range(len(clean) - 1):
        x_t = int(clean.iloc[t]["AOI"])
        x_t_plus_1 = int(clean.iloc[t + 1]["AOI"])

        step = stage_1_step(
            game=game,
            x_t=float(x_t),
            params=stage1_params,
        )

        q_t = np.asarray(step["q_t"], dtype=float)
        target = float(step["T_g"])

        w_t = br_region_weights(y=y, target=target, sigma_score=sigma_score)

        q_next_exact = float(np.clip(q_t[x_t_plus_1 - 1], 0.0, 1.0))
        q_zone_overlap = float(np.clip(relative_overlap_score(q_t, w_t), 0.0, 1.0))
        q_max = float(np.max(q_t))

        p_t = q_zone_overlap if use_relative_overlap else q_next_exact
        p_t = float(np.clip(p_t, 0.0, 1.0))

        rows.append(
            {
                "t": t,
                "x_t": x_t,
                "x_t_plus_1": x_t_plus_1,
                "T_g": target,
                "p_t": p_t,
                "q_next_exact": q_next_exact,
                "q_zone_overlap": q_zone_overlap,
                "q_max": q_max,
            }
        )

    return pd.DataFrame(rows)


def stage1_br_count_distribution(
    fixations: pd.DataFrame,
    game: int | str,
    stage1_params: Stage1Params,
    score_params,
) -> dict[str, object]:
    transition_df = compute_transition_br_probabilities_stage1(
        fixations=fixations,
        game=game,
        stage1_params=stage1_params,
        sigma_score=score_params.sigma_score,
        use_relative_overlap=score_params.use_relative_overlap,
    )

    if transition_df.empty:
        dist = np.array([1.0], dtype=float)
        summary = summarize_br_distribution(dist)
        return {
            "transition_df": transition_df,
            "distribution": dist,
            "summary": summary,
        }

    probs = transition_df["p_t"].to_numpy(dtype=float)
    dist = poisson_binomial_distribution(probs)
    summary = summarize_br_distribution(dist)

    return {
        "transition_df": transition_df,
        "distribution": dist,
        "summary": summary,
    }