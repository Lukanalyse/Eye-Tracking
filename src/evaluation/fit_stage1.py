from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.pages.page_1 import (
    discover_player_files,
    is_supported_game_sheet,
    load_workbook_sheets,
)
from src.saliency.dynamique_1 import (
    DynamicStage1Params,
    dynamic_stage1_step,
)


EPS = 1e-12

# Hard plausible ranges used to reject absurd values.
SIGMA_COMP_INF_RANGE = (0.2, 12.0)
SIGMA_COMP_AMP_RANGE = (0.0, 20.0)
KAPPA_GAME_RANGE = (1.0, 100.0)
TAU_RHO_MS_RANGE = (10.0, 500.0)
HARD_PENALTY_SCALE = 1e6

# Soft regularization weights used to discourage overly diffuse solutions.
LAMBDA_SIGMA_INF = 0.05
LAMBDA_SIGMA_AMP = 0.05


FIXED_STAGE1_PARAMS = DynamicStage1Params(
    lambda_stat=0.3,
    lambda_dyn=2.0,
    sigma_comp_inf=2.0,
    sigma_comp_amp=8.0,
    sigma_space=1.6,
    kappa_game=20.0,
    tau_rho_ms=120.0,
    beta=None,
)


def unpack_stage1_params(theta_free: np.ndarray, base_params: DynamicStage1Params) -> DynamicStage1Params:
    """
    Map unconstrained parameters to valid model parameters.

    theta_free = [a, b, c, d]
    with:
        sigma_comp_inf = exp(a)
        sigma_comp_amp = exp(b)
        kappa_game     = exp(c)
        tau_rho_ms     = exp(d)
    """
    a, b, c, d = [float(x) for x in theta_free]

    sigma_comp_inf = np.exp(a)
    sigma_comp_amp = np.exp(b)
    kappa_game = np.exp(c)
    tau_rho_ms = np.exp(d)

    return replace(
        base_params,
        sigma_comp_inf=float(sigma_comp_inf),
        sigma_comp_amp=float(sigma_comp_amp),
        kappa_game=float(kappa_game),
        tau_rho_ms=float(tau_rho_ms),
        beta=None,
    )


def pack_stage1_params(params: DynamicStage1Params) -> np.ndarray:
    """Inverse transform for initialization."""
    sigma_inf = max(float(params.sigma_comp_inf), EPS)
    sigma_amp = max(float(params.sigma_comp_amp), EPS)
    kappa = max(float(params.kappa_game), EPS)
    tau_rho = max(float(params.tau_rho_ms), EPS)

    return np.array(
        [
            np.log(sigma_inf),
            np.log(sigma_amp),
            np.log(kappa),
            np.log(tau_rho),
        ],
        dtype=float,
    )


def _range_penalty(value: float, lower: float, upper: float) -> float:
    """Quadratic penalty outside a plausible interval. Returns 0 inside."""
    if value < lower:
        return (lower - value) ** 2
    if value > upper:
        return (value - upper) ** 2
    return 0.0


def stage1_hard_penalty(params: DynamicStage1Params) -> float:
    """
    Reject implausible values very strongly so the optimizer stays in a
    scientifically interpretable region.
    """
    sigma_inf = float(params.sigma_comp_inf)
    sigma_amp = float(params.sigma_comp_amp)
    kappa = float(params.kappa_game)
    tau_rho = float(params.tau_rho_ms)

    penalty = 0.0
    penalty += _range_penalty(sigma_inf, *SIGMA_COMP_INF_RANGE)
    penalty += _range_penalty(sigma_amp, *SIGMA_COMP_AMP_RANGE)
    penalty += _range_penalty(kappa, *KAPPA_GAME_RANGE)
    penalty += _range_penalty(tau_rho, *TAU_RHO_MS_RANGE)

    return float(HARD_PENALTY_SCALE * penalty)


def stage1_soft_regularization(params: DynamicStage1Params) -> float:
    """
    Softly penalize overly diffuse solutions.

    This is not an arbitrary visual adjustment. It encodes a modeling prior:
    computational dispersion should decrease over the game, but should not stay
    unrealistically large at all times.
    """
    sigma_inf = float(params.sigma_comp_inf)
    sigma_amp = float(params.sigma_comp_amp)

    return float(
        LAMBDA_SIGMA_INF * sigma_inf
        + LAMBDA_SIGMA_AMP * sigma_amp
    )


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


def sequence_log_likelihood(
    fixations: pd.DataFrame,
    game: int | str,
    params: DynamicStage1Params,
) -> float:
    """
    Compute causal log-likelihood for next-fixation prediction.

    At fixation t, the model predicts q_t(.).
    We then evaluate the probability assigned to the observed next AOI x_{t+1}.
    """
    fixations = validate_fixations(fixations)

    if len(fixations) < 2:
        return np.nan

    log_likelihood = 0.0
    n_terms = 0

    for t in range(len(fixations) - 1):
        current_aoi = int(fixations.iloc[t]["AOI"])
        current_time_ms = float(fixations.iloc[t]["Time"])
        next_aoi = int(fixations.iloc[t + 1]["AOI"])

        step = dynamic_stage1_step(
            game=game,
            x_t=float(current_aoi),
            u_ms=current_time_ms,
            fixation_index=t,
            params=params,
        )

        q_t = np.asarray(step["q_t"], dtype=float)
        prob_next = max(float(q_t[next_aoi - 1]), EPS)
        log_likelihood += np.log(prob_next)
        n_terms += 1

    if n_terms == 0:
        return np.nan
    return float(log_likelihood)


def mean_log_likelihood(
    fixations: pd.DataFrame,
    game: int | str,
    params: DynamicStage1Params,
) -> float:
    fixations = validate_fixations(fixations)
    if len(fixations) < 2:
        return np.nan
    ll = sequence_log_likelihood(fixations, game, params)
    return float(ll / (len(fixations) - 1))


def dataset_mean_log_likelihood(
    sequences: list[tuple[pd.DataFrame, str]],
    params: DynamicStage1Params,
) -> float:
    """
    Average log-likelihood across multiple sequences.

    Each sequence contributes proportionally to its number of predictive steps.
    """
    total_ll = 0.0
    total_terms = 0

    for fixations, game in sequences:
        clean = validate_fixations(fixations)
        if len(clean) < 2:
            continue
        ll = sequence_log_likelihood(clean, game, params)
        if np.isnan(ll) or np.isinf(ll):
            continue
        total_ll += ll
        total_terms += len(clean) - 1

    if total_terms == 0:
        return np.nan
    return float(total_ll / total_terms)


def objective_stage1(
    theta_free: np.ndarray,
    sequences: list[tuple[pd.DataFrame, str]],
    base_params: DynamicStage1Params,
) -> float:
    """Objective to minimize = negative dataset mean log-likelihood + regularization."""
    try:
        params = unpack_stage1_params(theta_free, base_params)
        mll = dataset_mean_log_likelihood(sequences=sequences, params=params)

        if np.isnan(mll) or np.isinf(mll):
            return 1e9

        hard_penalty = stage1_hard_penalty(params)
        soft_penalty = stage1_soft_regularization(params)
        return float(-mll + hard_penalty + soft_penalty)
    except Exception:
        return 1e9


def fit_stage1_on_sequences(
    sequences: list[tuple[pd.DataFrame, str]],
    base_params: DynamicStage1Params | None = None,
    maxiter: int = 300,
) -> dict[str, Any]:
    """Fit Stage 1 parameters on several sequences simultaneously."""
    valid_sequences = []
    for fixations, game in sequences:
        clean = validate_fixations(fixations)
        if len(clean) >= 2:
            valid_sequences.append((clean, game))

    if not valid_sequences:
        raise ValueError("No valid sequences with at least 2 fixations.")

    if base_params is None:
        base_params = FIXED_STAGE1_PARAMS

    theta0 = pack_stage1_params(base_params)

    initial_params = unpack_stage1_params(theta0, base_params)
    initial_mll = dataset_mean_log_likelihood(valid_sequences, initial_params)

    result = minimize(
        fun=objective_stage1,
        x0=theta0,
        args=(valid_sequences, base_params),
        method="Powell",
        options={"maxiter": maxiter, "xtol": 1e-3, "ftol": 1e-3},
    )

    fitted_params = unpack_stage1_params(result.x, base_params)
    final_mll = dataset_mean_log_likelihood(valid_sequences, fitted_params)
    final_hard_penalty = stage1_hard_penalty(fitted_params)
    final_soft_penalty = stage1_soft_regularization(fitted_params)

    return {
        "success": bool(result.success),
        "message": result.message,
        "n_iterations": int(result.nit) if hasattr(result, "nit") else None,
        "n_sequences": len(valid_sequences),
        "initial_mean_log_likelihood": float(initial_mll),
        "final_mean_log_likelihood": float(final_mll),
        "improvement": float(final_mll - initial_mll),
        "final_hard_penalty": float(final_hard_penalty),
        "final_soft_penalty": float(final_soft_penalty),
        "fitted_params": fitted_params,
        "optimizer_result": result,
    }


def load_one_player_all_games(player_name: str) -> list[tuple[pd.DataFrame, str]]:
    project_root = Path(__file__).resolve().parents[2]
    player_files = discover_player_files(str(project_root))

    if player_name not in player_files:
        raise ValueError(f"Player '{player_name}' not found.")

    workbook = load_workbook_sheets(player_files[player_name])
    sequences: list[tuple[pd.DataFrame, str]] = []

    for sheet_name, fixations in workbook.items():
        if is_supported_game_sheet(sheet_name):
            sequences.append((validate_fixations(fixations), sheet_name))

    return sequences


def summarize_sequence_lengths(sequences: list[tuple[pd.DataFrame, str]]) -> pd.DataFrame:
    rows = []
    for fixations, game in sequences:
        rows.append(
            {
                "game": game,
                "n_fixations": len(fixations),
                "n_predictions": max(len(fixations) - 1, 0),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    player_name = "Player12_AOI_TIME"

    sequences = load_one_player_all_games(player_name)
    summary_df = summarize_sequence_lengths(sequences)

    print("=" * 80)
    print("FIT STAGE 1")
    print("=" * 80)
    print(f"Player: {player_name}")
    print(f"Number of games used: {len(sequences)}")
    print()
    print(summary_df.to_string(index=False))
    print()

    fit_result = fit_stage1_on_sequences(
        sequences=sequences,
        base_params=FIXED_STAGE1_PARAMS,
        maxiter=300,
    )

    fitted = fit_result["fitted_params"]

    print(f"Success: {fit_result['success']}")
    print(f"Message: {fit_result['message']}")
    print(f"Iterations: {fit_result['n_iterations']}")
    print(f"Sequences: {fit_result['n_sequences']}")
    print()
    print(f"Initial mean log-likelihood: {fit_result['initial_mean_log_likelihood']:.6f}")
    print(f"Final   mean log-likelihood: {fit_result['final_mean_log_likelihood']:.6f}")
    print(f"Improvement                : {fit_result['improvement']:.6f}")
    print(f"Final hard penalty         : {fit_result['final_hard_penalty']:.6f}")
    print(f"Final soft penalty         : {fit_result['final_soft_penalty']:.6f}")
    print()
    print("Fitted parameters:")
    print(f"  lambda_stat    = {fitted.lambda_stat:.6f}  (fixed)")
    print(f"  lambda_dyn     = {fitted.lambda_dyn:.6f}  (fixed)")
    print(f"  sigma_space    = {fitted.sigma_space:.6f}  (fixed)")
    print(f"  sigma_comp_inf = {fitted.sigma_comp_inf:.6f}")
    print(f"  sigma_comp_amp = {fitted.sigma_comp_amp:.6f}")
    print(f"  kappa_game     = {fitted.kappa_game:.6f}")
    print(f"  tau_rho_ms     = {fitted.tau_rho_ms:.6f}")
    print(f"  beta           = {fitted.beta}")
    print(f"  sigma_comp_0   = {fitted.sigma_comp_inf + fitted.sigma_comp_amp:.6f}")
    print("=" * 80)