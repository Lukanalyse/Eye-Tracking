from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from src.saliency.static_salience import AOI_COUNT
from src.saliency.dynamique_1 import (
    DynamicStage1Params,
    strategic_target,
    fixation_activation,
    sigma_comp_over_game,
    computational_br_map,
    spatially_blurred_br_map,
    stable_softmax,
)


@dataclass
class DynamicStage2Params(DynamicStage1Params):
    memory_depth: int = 4
    memory_decay: float = 0.5
    tau_memory_ms: float = 200.0


def transition_memory_weights(memory_depth: int, memory_decay: float) -> np.ndarray:
    """
    Transition-based decay weights:
        w_j = delta^j,   j = 1,...,L
    """
    depth = max(int(memory_depth), 0)
    delta = float(np.clip(memory_decay, 0.0, 1.0))

    if depth == 0:
        return np.array([], dtype=float)

    return np.array([delta ** j for j in range(1, depth + 1)], dtype=float)


def temporal_memory_factors(
    previous_delta_times_ms: list[float] | None,
    tau_memory_ms: float,
) -> np.ndarray:
    """
    Time-based memory decay:
        m_j = exp(-DeltaT_j / tau_mem)

    If previous_delta_times_ms is None, returns an empty array.
    """
    if previous_delta_times_ms is None or len(previous_delta_times_ms) == 0:
        return np.array([], dtype=float)

    tau_safe = max(float(tau_memory_ms), 1.0)
    delta_times = np.maximum(np.asarray(previous_delta_times_ms, dtype=float), 0.0)
    return np.exp(-delta_times / tau_safe)


def memory_map_from_previous_br(
    previous_br_space_maps: list[np.ndarray] | None,
    previous_delta_times_ms: list[float] | None,
    memory_depth: int,
    memory_decay: float,
    tau_memory_ms: float,
) -> dict[str, np.ndarray]:
    """
    Build the memory map:
        M_t(y) = sum_{j=1}^L delta^j * exp(-DeltaT_j / tau_mem) * BR_{t-j}^{space}(y)

    previous_br_space_maps must be ordered:
        [BR_{t-1}^{space}, BR_{t-2}^{space}, ..., BR_{t-k}^{space}]
    """
    if previous_br_space_maps is None or len(previous_br_space_maps) == 0 or memory_depth <= 0:
        empty = np.zeros(AOI_COUNT, dtype=float)
        return {
            "memory_map": empty,
            "transition_weights": np.array([], dtype=float),
            "temporal_factors": np.array([], dtype=float),
            "effective_weights": np.array([], dtype=float),
        }

    usable_depth = min(int(memory_depth), len(previous_br_space_maps))
    prev_maps = previous_br_space_maps[:usable_depth]

    if previous_delta_times_ms is None:
        prev_times = [0.0] * usable_depth
    else:
        if len(previous_delta_times_ms) < usable_depth:
            raise ValueError("previous_delta_times_ms has fewer elements than previous_br_space_maps.")
        prev_times = previous_delta_times_ms[:usable_depth]

    trans_weights = transition_memory_weights(
        memory_depth=usable_depth,
        memory_decay=memory_decay,
    )
    temp_factors = temporal_memory_factors(
        previous_delta_times_ms=prev_times,
        tau_memory_ms=tau_memory_ms,
    )

    effective_weights = trans_weights * temp_factors
    memory_map = np.zeros(AOI_COUNT, dtype=float)

    for j, br_map in enumerate(prev_maps):
        br_array = np.asarray(br_map, dtype=float)
        if br_array.shape != (AOI_COUNT,):
            raise ValueError(f"Each previous BR map must have shape ({AOI_COUNT},), got {br_array.shape}.")
        memory_map += effective_weights[j] * br_array

    return {
        "memory_map": memory_map,
        "transition_weights": trans_weights,
        "temporal_factors": temp_factors,
        "effective_weights": effective_weights,
    }


def current_br_space_map(
    game: int | str,
    x_t: float,
    fixation_index: int,
    sigma_comp_inf: float,
    sigma_comp_amp: float,
    kappa_game: float,
    sigma_space: float,
) -> dict[str, np.ndarray | float]:
    """
    Compute the current Stage-1-style BR map:
        target
        sigma_comp_t
        BR_comp_t
        BR_space_t
    """
    y = np.arange(1, AOI_COUNT + 1, dtype=float)

    target = strategic_target(game, x_t)
    sigma_comp_t = sigma_comp_over_game(
        fixation_index=fixation_index,
        sigma_comp_inf=sigma_comp_inf,
        sigma_comp_amp=sigma_comp_amp,
        kappa_game=kappa_game,
    )
    br_comp = computational_br_map(y=y, target=target, sigma_comp=sigma_comp_t)
    br_space = spatially_blurred_br_map(br_comp=br_comp, sigma_space=sigma_space)

    return {
        "y": y,
        "T_g": target,
        "sigma_comp_t": sigma_comp_t,
        "BR_comp_t": br_comp,
        "BR_space_t": br_space,
    }


def dynamic_stage2_step(
    game: int | str,
    x_t: float,
    u_ms: float,
    fixation_index: int,
    params: DynamicStage2Params,
    previous_br_space_maps: list[np.ndarray] | None = None,
    previous_delta_times_ms: list[float] | None = None,
) -> dict[str, np.ndarray | float]:
    """
    Stage 2 dynamic salience with local memory.

    Model:
        S_t(y) = lambda_stat
                 + lambda_dyn * [ rho_t * BR_t^{space}(y) + M_t(y) ]

    where
        rho_t = 1 - exp(-u_t / tau_rho)

        M_t(y) = sum_{j=1}^L delta^j * exp(-DeltaT_j / tau_mem) * BR_{t-j}^{space}(y)
    """
    current = current_br_space_map(
        game=game,
        x_t=x_t,
        fixation_index=fixation_index,
        sigma_comp_inf=params.sigma_comp_inf,
        sigma_comp_amp=params.sigma_comp_amp,
        kappa_game=params.kappa_game,
        sigma_space=params.sigma_space,
    )

    y = np.asarray(current["y"], dtype=float)
    s_stat = np.ones(AOI_COUNT, dtype=float)

    rho_t = fixation_activation(u_ms=u_ms, tau_rho_ms=params.tau_rho_ms)

    br_comp_t = np.asarray(current["BR_comp_t"], dtype=float)
    br_space_t = np.asarray(current["BR_space_t"], dtype=float)

    current_dynamic = rho_t * br_space_t

    memory_result = memory_map_from_previous_br(
        previous_br_space_maps=previous_br_space_maps,
        previous_delta_times_ms=previous_delta_times_ms,
        memory_depth=params.memory_depth,
        memory_decay=params.memory_decay,
        tau_memory_ms=params.tau_memory_ms,
    )

    memory_map = np.asarray(memory_result["memory_map"], dtype=float)
    combined_dynamic = current_dynamic + memory_map

    s_dyn_t = params.lambda_stat * s_stat + params.lambda_dyn * combined_dynamic

    result: dict[str, np.ndarray | float] = {
        "y": y,
        "S_stat": s_stat,
        "T_g": float(current["T_g"]),
        "rho": float(rho_t),
        "sigma_comp_t": float(current["sigma_comp_t"]),
        "BR_comp_t": br_comp_t,
        "BR_space_t": br_space_t,
        "current_dynamic_t": current_dynamic,
        "memory_map_t": memory_map,
        "combined_dynamic_t": combined_dynamic,
        "S_dyn_t": s_dyn_t,
        "transition_weights": np.asarray(memory_result["transition_weights"], dtype=float),
        "temporal_factors": np.asarray(memory_result["temporal_factors"], dtype=float),
        "effective_memory_weights": np.asarray(memory_result["effective_weights"], dtype=float),
    }

    if params.beta is not None:
        result["q_t"] = stable_softmax(s_dyn_t, beta=params.beta)

    return result


def build_stage2_sequence(
    game: int | str,
    fixations,
    params: DynamicStage2Params,
) -> list[dict[str, np.ndarray | float]]:
    """
    Build the full Stage 2 sequence over all fixations.

    Expected columns in `fixations`:
        - AOI
        - Time

    The current BR map at each fixation is stored and then reused as memory
    for subsequent fixations.
    """
    if fixations.empty:
        return []

    results: list[dict[str, np.ndarray | float]] = []
    previous_br_space_maps: list[np.ndarray] = []
    previous_delta_times_ms: list[float] = []

    for fixation_index in range(len(fixations)):
        current_aoi = float(fixations.iloc[fixation_index]["AOI"])
        tau_t_ms = float(fixations.iloc[fixation_index]["Time"])

        # For the sequence builder, we evaluate each fixation at its full duration.
        u_ms = tau_t_ms

        step_result = dynamic_stage2_step(
            game=game,
            x_t=current_aoi,
            u_ms=u_ms,
            fixation_index=fixation_index,
            params=params,
            previous_br_space_maps=previous_br_space_maps,
            previous_delta_times_ms=previous_delta_times_ms,
        )

        results.append(step_result)

        current_br_space = np.asarray(step_result["BR_space_t"], dtype=float)

        previous_br_space_maps.insert(0, current_br_space)
        previous_delta_times_ms.insert(0, 0.0)

        for i in range(len(previous_delta_times_ms)):
            previous_delta_times_ms[i] += tau_t_ms

        if len(previous_br_space_maps) > params.memory_depth:
            previous_br_space_maps = previous_br_space_maps[: params.memory_depth]
            previous_delta_times_ms = previous_delta_times_ms[: params.memory_depth]

    return results