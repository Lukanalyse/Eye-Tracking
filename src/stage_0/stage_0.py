# src/saliency/stage_0.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.saliency.static_salience import AOI_COUNT, get_game_metadata


@dataclass(frozen=True)
class Stage0Params:
    lambda_stat: float = 0.5
    lambda_br: float = 0.5
    sigma_comp: float = 6.0


def normalize_stage0_weights(lambda_stat: float, lambda_br: float) -> tuple[float, float]:
    """
    Enforce nonnegative weights summing to 1.
    """
    ls = max(float(lambda_stat), 0.0)
    lb = max(float(lambda_br), 0.0)
    total = ls + lb
    if total <= 0:
        return 0.5, 0.5
    return ls / total, lb / total


def normalize_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(np.sum(arr))
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def strategic_target(game: int | str, x_t: float) -> float:
    meta = get_game_metadata(game)
    x_val = float(x_t)

    if meta["game_type"] == "BCG+":
        target = (2.0 / 3.0) * x_val + 20.0
    elif meta["game_type"] == "BCG-":
        target = 100.0 - (2.0 / 3.0) * x_val
    else:
        raise ValueError(f"Unsupported game type: {meta['game_type']}")

    return float(np.clip(target, 1.0, float(AOI_COUNT)))


def computational_br_map(y: np.ndarray, target: float, sigma_comp: float) -> np.ndarray:
    sigma_safe = max(float(sigma_comp), 1e-9)
    return np.exp(-((y - target) ** 2) / (2.0 * sigma_safe ** 2))


def stage_0_step(
    game: int | str,
    x_t: float,
    params: Stage0Params,
) -> dict[str, np.ndarray | float]:
    y = np.arange(1, AOI_COUNT + 1, dtype=float)

    lambda_stat, lambda_br = normalize_stage0_weights(
        params.lambda_stat,
        params.lambda_br,
    )

    s_stat = np.ones(AOI_COUNT, dtype=float)
    target = strategic_target(game, x_t)
    br_comp = computational_br_map(y=y, target=target, sigma_comp=params.sigma_comp)

    s_stage0 = lambda_stat * s_stat + lambda_br * br_comp
    q_t = normalize_probability(s_stage0)

    return {
        "y": y,
        "lambda_stat": float(lambda_stat),
        "lambda_br": float(lambda_br),
        "S_stat": s_stat,
        "T_g": float(target),
        "sigma_comp": float(params.sigma_comp),
        "BR_comp_t": br_comp,
        "S_stage0_t": s_stage0,
        "q_t": q_t,
    }