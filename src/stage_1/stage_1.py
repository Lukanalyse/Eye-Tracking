from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.saliency.static_salience import AOI_COUNT, GRID_SIZE, get_game_metadata


@dataclass(frozen=True)
class Stage1Params:
    lambda_stat: float = 0.0
    lambda_br: float = 1.0
    sigma_comp: float = 8.0
    sigma_space: float = 1.5


def normalize_probability(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(np.sum(arr))
    if total <= 0:
        return np.ones_like(arr) / len(arr)
    return arr / total


def aoi_grid_coordinates() -> np.ndarray:
    idx = np.arange(AOI_COUNT)
    rows = idx // GRID_SIZE
    cols = idx % GRID_SIZE
    return np.column_stack((rows, cols)).astype(float)


def gaussian_spatial_kernel(sigma_space: float) -> np.ndarray:
    if sigma_space <= 0:
        raise ValueError("sigma_space must be > 0.")

    coords = aoi_grid_coordinates()
    deltas = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(deltas ** 2, axis=2)

    kernel = np.exp(-dist_sq / (2.0 * sigma_space ** 2))
    row_sums = kernel.sum(axis=1, keepdims=True)
    return kernel / row_sums


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


def spatially_blurred_br_map(br_comp: np.ndarray, sigma_space: float) -> np.ndarray:
    kernel = gaussian_spatial_kernel(sigma_space)
    return kernel @ br_comp


def stage_1_step(
    game: int | str,
    x_t: float,
    params: Stage1Params,
) -> dict[str, np.ndarray | float]:
    """
    Stage 1:
    - BR computationnelle sur l'axe 1..100
    - blur spatial gaussien sur la grille 10x10
    - normalisation probabiliste finale
    """
    y = np.arange(1, AOI_COUNT + 1, dtype=float)

    s_stat = np.ones(AOI_COUNT, dtype=float)
    target = strategic_target(game, x_t)

    br_comp = computational_br_map(
        y=y,
        target=target,
        sigma_comp=params.sigma_comp,
    )

    br_space = spatially_blurred_br_map(
        br_comp=br_comp,
        sigma_space=params.sigma_space,
    )

    s_stage1 = params.lambda_stat * s_stat + params.lambda_br * br_space
    q_t = normalize_probability(s_stage1)

    return {
        "y": y,
        "S_stat": s_stat,
        "T_g": target,
        "sigma_comp": float(params.sigma_comp),
        "sigma_space": float(params.sigma_space),
        "BR_comp_t": br_comp,
        "BR_space_t": br_space,
        "S_stage1_t": s_stage1,
        "q_t": q_t,
    }