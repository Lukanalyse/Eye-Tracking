from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.pages.page_1 import AOI_COUNT
from src.stage_0.stage_0 import Stage0Params, stage_0_step


@dataclass(frozen=True)
class Score0Params:
    sigma_score: float = 6.0
    use_relative_overlap: bool = True


def normalize_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(np.sum(arr))
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def br_region_weights(y: np.ndarray, target: float, sigma_score: float) -> np.ndarray:
    """
    Gaussian BR zone centered on the strategic target.
    This defines a soft region rather than a single exact AOI.
    """
    sigma_safe = max(float(sigma_score), 1e-9)
    raw = np.exp(-((y - target) ** 2) / (2.0 * sigma_safe**2))
    return normalize_probability(raw)


def relative_overlap_score(q_t: np.ndarray, w_t: np.ndarray) -> float:
    """
    Relative overlap between the salience probability map q_t and the BR zone w_t.

    Raw overlap:
        <q_t, w_t> = sum_y q_t(y) w_t(y)

    We normalize by the maximal self-overlap of w_t:
        sum_y w_t(y)^2

    so that if q_t == w_t, the score is 1.
    """
    q = np.asarray(q_t, dtype=float)
    w = np.asarray(w_t, dtype=float)

    numerator = float(np.sum(q * w))
    denominator = float(np.sum(w * w))

    if denominator <= 0:
        return 0.0

    score = numerator / denominator
    return float(np.clip(score, 0.0, 1.0))


def compute_transition_br_probabilities(
    fixations: pd.DataFrame,
    game: int | str,
    stage0_params: Stage0Params,
    score0_params: Score0Params,
) -> pd.DataFrame:
    """
    Compute one BR probability per transition.

    Stage 0 logic:
    - At fixation t, build q_t from x_t.
    - Define a BR zone around T_g(x_t).
    - Score how aligned q_t is with that BR zone.

    This is intentionally soft:
    if BR is around 56 and the salience is high around 52, 53, 54, 55, 56,
    the transition still gets a high BR probability.
    """
    required_cols = {"AOI", "Time"}
    missing = required_cols - set(fixations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = fixations.copy()
    clean = clean.dropna(subset=["AOI", "Time"]).reset_index(drop=True)

    if len(clean) < 2:
        return pd.DataFrame(
            columns=[
                "t",
                "x_t",
                "x_t_plus_1",
                "T_g",
                "p_t",
                "q_next_exact",
                "q_zone_overlap",
                "q_max",
            ]
        )

    clean["AOI"] = clean["AOI"].astype(int)

    y = np.arange(1, AOI_COUNT + 1, dtype=float)
    rows = []

    for t in range(len(clean) - 1):
        x_t = int(clean.iloc[t]["AOI"])
        x_t_plus_1 = int(clean.iloc[t + 1]["AOI"])

        step = stage_0_step(
            game=game,
            x_t=float(x_t),
            params=stage0_params,
        )

        q_t = np.asarray(step["q_t"], dtype=float)
        target = float(step["T_g"])

        w_t = br_region_weights(
            y=y,
            target=target,
            sigma_score=score0_params.sigma_score,
        )

        q_next_exact = float(np.clip(q_t[x_t_plus_1 - 1], 0.0, 1.0))
        q_zone_overlap = relative_overlap_score(q_t, w_t)
        q_max = float(np.max(q_t))

        if score0_params.use_relative_overlap:
            p_t = q_zone_overlap
        else:
            p_t = q_next_exact

        rows.append(
            {
                "t": t,
                "x_t": x_t,
                "x_t_plus_1": x_t_plus_1,
                "T_g": target,
                "p_t": float(np.clip(p_t, 0.0, 1.0)),
                "q_next_exact": q_next_exact,
                "q_zone_overlap": q_zone_overlap,
                "q_max": q_max,
            }
        )

    return pd.DataFrame(rows)


def poisson_binomial_distribution(probs: np.ndarray) -> np.ndarray:
    """
    Dynamic-programming computation of the Poisson-binomial distribution.

    If N = sum_t B_t with independent Bernoulli(B_t; p_t),
    returns dist[k] = P(N = k).
    """
    probs = np.asarray(probs, dtype=float)
    n = len(probs)

    dist = np.zeros(n + 1, dtype=float)
    dist[0] = 1.0

    for p in probs:
        p = float(np.clip(p, 0.0, 1.0))
        new_dist = np.zeros_like(dist)

        for k in range(n + 1):
            new_dist[k] += dist[k] * (1.0 - p)
            if k + 1 <= n:
                new_dist[k + 1] += dist[k] * p

        dist = new_dist

    total = float(np.sum(dist))
    if total > 0:
        dist /= total

    return dist


def summarize_br_distribution(dist: np.ndarray) -> dict[str, float | np.ndarray]:
    dist = np.asarray(dist, dtype=float)
    k = np.arange(len(dist), dtype=float)

    expected_count = float(np.sum(k * dist))
    variance = float(np.sum(((k - expected_count) ** 2) * dist))
    mode = int(np.argmax(dist))
    cumulative = np.cumsum(dist)

    median = 0
    for idx, c in enumerate(cumulative):
        if c >= 0.5:
            median = idx
            break

    return {
        "expected_count": expected_count,
        "variance": variance,
        "mode": mode,
        "median": int(median),
        "prob_at_least_one": float(1.0 - dist[0]),
        "cumulative_probabilities": cumulative,
    }


def stage0_br_count_distribution(
    fixations: pd.DataFrame,
    game: int | str,
    stage0_params: Stage0Params,
    score0_params: Score0Params,
) -> dict[str, object]:
    """
    Full Stage 0 BR scoring pipeline.

    Output:
    - transition_df : one BR probability per transition
    - distribution  : Poisson-binomial distribution of total BR count
    - summary       : expected count, variance, mode, median, etc.
    """
    transition_df = compute_transition_br_probabilities(
        fixations=fixations,
        game=game,
        stage0_params=stage0_params,
        score0_params=score0_params,
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