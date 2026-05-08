from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.pages.page_1 import AOI_COUNT
from src.stage_0.stage_0 import Stage0Params, stage_0_step


@dataclass(frozen=True)
class Score0Params:
    sigma_score: float = 4.75
    use_relative_overlap: bool = False


def normalize_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(np.sum(arr))
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def normalized_shannon_entropy(probabilities: np.ndarray) -> float:
    """
    Shannon entropy normalized to [0, 1].

    0 -> very concentrated distribution
    1 -> perfectly uniform distribution
    """
    q = np.asarray(probabilities, dtype=float)
    q = np.clip(q, 1e-12, 1.0)
    entropy = -float(np.sum(q * np.log(q)))
    max_entropy = float(np.log(len(q)))
    if max_entropy <= 0:
        return 0.0
    return float(entropy / max_entropy)


def br_region_weights(y: np.ndarray, target: float, sigma_score: float) -> np.ndarray:
    """
    Gaussian BR region centered on the strategic target.
    Used only for diagnostics / entropy of the BR scoring region.
    """
    sigma_safe = max(float(sigma_score), 1e-9)
    raw = np.exp(-((y - target) ** 2) / (2.0 * sigma_safe**2))
    return normalize_probability(raw)


def relative_overlap_score(q: np.ndarray, w: np.ndarray) -> float:
    """
    Relative overlap between two probability distributions on the same AOI support.

    Returns a score in [0, 1]:
    - 1 if the two distributions are identical
    - 0 if they do not overlap at all
    """
    q = np.asarray(q, dtype=float)
    w = np.asarray(w, dtype=float)

    q = normalize_probability(q)
    w = normalize_probability(w)

    return float(np.clip(np.sum(np.minimum(q, w)), 0.0, 1.0))


def br_distance_probability(
    observed_aoi: float,
    target: float,
    sigma_score: float,
) -> float:
    """
    Local BR probability based only on the distance between the observed
    next AOI and the BR target.

    p_t = exp( - (x_{t+1} - T_g(x_t))^2 / (2 sigma_score^2) )

    Properties:
    - exact BR -> p_t = 1
    - close to BR -> high p_t
    - far from BR -> low p_t
    """
    sigma_safe = max(float(sigma_score), 1e-9)
    distance = float(observed_aoi) - float(target)
    p_t = np.exp(-(distance**2) / (2.0 * sigma_safe**2))
    return float(np.clip(p_t, 0.0, 1.0))


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


def compute_transition_br_probabilities(
    fixations: pd.DataFrame,
    game: int | str,
    stage0_params: Stage0Params,
    score0_params: Score0Params,
) -> pd.DataFrame:
    """
    Compute one BR probability per transition.

    New simple Stage 0 scoring logic:
    - At fixation t, compute the BR target T_g(x_t)
    - Observe next fixation x_{t+1}
    - Measure the distance to the BR target
    - Convert this distance into a soft BR probability

    This creates one piece of evidence per transition.
    """
    clean = validate_fixations(fixations)

    if len(clean) < 2:
        return pd.DataFrame(
            columns=[
                "t",
                "x_t",
                "x_t_plus_1",
                "T_g",
                "distance_to_br",
                "p_t",
                "q_next_exact",
                "q_max",
                "score_entropy",
            ]
        )

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

        distance_to_br = float(abs(float(x_t_plus_1) - target))
        p_t = br_distance_probability(
            observed_aoi=float(x_t_plus_1),
            target=target,
            sigma_score=score0_params.sigma_score,
        )

        q_next_exact = float(np.clip(q_t[x_t_plus_1 - 1], 0.0, 1.0))
        q_max = float(np.max(q_t))

        w_t = br_region_weights(
            y=y,
            target=target,
            sigma_score=score0_params.sigma_score,
        )
        score_entropy = normalized_shannon_entropy(w_t)

        rows.append(
            {
                "t": t,
                "x_t": x_t,
                "x_t_plus_1": x_t_plus_1,
                "T_g": target,
                "distance_to_br": distance_to_br,
                "p_t": p_t,
                "q_next_exact": q_next_exact,
                "q_max": q_max,
                "score_entropy": score_entropy,
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
        summary.update(
            {
                "mean_p_t": np.nan,
                "mean_score_entropy": np.nan,
                "mean_distance_to_br": np.nan,
            }
        )
        return {
            "transition_df": transition_df,
            "distribution": dist,
            "summary": summary,
        }

    probs = transition_df["p_t"].to_numpy(dtype=float)
    dist = poisson_binomial_distribution(probs)
    summary = summarize_br_distribution(dist)

    summary.update(
        {
            "mean_p_t": float(transition_df["p_t"].mean()),
            "mean_score_entropy": float(transition_df["score_entropy"].mean()),
            "mean_distance_to_br": float(transition_df["distance_to_br"].mean()),
        }
    )

    return {
        "transition_df": transition_df,
        "distribution": dist,
        "summary": summary,
    }