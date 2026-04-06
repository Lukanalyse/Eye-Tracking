from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.saliency.static_salience import AOI_COUNT, GRID_SIZE, get_game_metadata


@dataclass
class DynamicStage1Params:
    lambda_stat: float = 0.3
    lambda_dyn: float = 2.0
    sigma_comp_min: float = 2.0
    sigma_comp_max: float = 10.0
    sigma_space: float = 1.5
    kappa_game: float = 20.0
    tau_rho_ms: float = 120.0
    beta: float | None = None


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

    kernel = np.exp(-dist_sq / (2.0 * sigma_space**2))
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


def fixation_activation(u_ms: float, tau_rho_ms: float) -> float:
    """
    Saturating intra-fixation activation based on absolute elapsed time.

    rho(u) = 1 - exp(-u / tau_rho)
    """
    u_safe = max(float(u_ms), 0.0)
    tau_safe = max(float(tau_rho_ms), 1.0)
    return float(1.0 - np.exp(-u_safe / tau_safe))


def sigma_comp_over_game(
    fixation_index: int,
    sigma_comp_min: float,
    sigma_comp_max: float,
    kappa_game: float,
) -> float:
    t = max(int(fixation_index), 0)
    kappa_safe = max(float(kappa_game), 1e-9)

    sigma_min = float(sigma_comp_min)
    sigma_max = float(sigma_comp_max)

    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_comp_min and sigma_comp_max must be > 0.")
    if sigma_min > sigma_max:
        raise ValueError("sigma_comp_min must be <= sigma_comp_max.")

    return float(sigma_min + (sigma_max - sigma_min) * np.exp(-t / kappa_safe))


def computational_br_map(y: np.ndarray, target: float, sigma_comp: float) -> np.ndarray:
    sigma_safe = max(float(sigma_comp), 1e-9)
    return np.exp(-((y - target) ** 2) / (2.0 * sigma_safe**2))


def spatially_blurred_br_map(br_comp: np.ndarray, sigma_space: float) -> np.ndarray:
    kernel = gaussian_spatial_kernel(sigma_space)
    return kernel @ br_comp


def stable_softmax(values: np.ndarray, beta: float) -> np.ndarray:
    beta_safe = max(float(beta), 1e-12)
    shifted = values - np.max(values)
    logits = np.exp(beta_safe * shifted)
    return logits / np.sum(logits)


def dynamic_stage1_step(
    game: int | str,
    x_t: float,
    u_ms: float,
    fixation_index: int,
    params: DynamicStage1Params,
) -> dict[str, np.ndarray | float]:
    y = np.arange(1, AOI_COUNT + 1, dtype=float)

    s_stat = np.ones(AOI_COUNT, dtype=float)
    target = strategic_target(game, x_t)
    rho_t = fixation_activation(u_ms, tau_rho_ms=params.tau_rho_ms)

    sigma_comp_t = sigma_comp_over_game(
        fixation_index=fixation_index,
        sigma_comp_min=params.sigma_comp_min,
        sigma_comp_max=params.sigma_comp_max,
        kappa_game=params.kappa_game,
    )

    br_comp = computational_br_map(y=y, target=target, sigma_comp=sigma_comp_t)
    br_space = spatially_blurred_br_map(br_comp=br_comp, sigma_space=params.sigma_space)

    s_dyn = params.lambda_stat * s_stat + params.lambda_dyn * rho_t * br_space

    result: dict[str, np.ndarray | float] = {
        "y": y,
        "S_stat": s_stat,
        "T_g": target,
        "rho": rho_t,
        "sigma_comp_t": sigma_comp_t,
        "BR_comp_t": br_comp,
        "BR_space_t": br_space,
        "S_dyn_t": s_dyn,
    }

    if params.beta is not None:
        result["q_t"] = stable_softmax(s_dyn, beta=params.beta)
    return result