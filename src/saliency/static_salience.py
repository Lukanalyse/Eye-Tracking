from __future__ import annotations

import re

import numpy as np


# ---------------------------------------------------------------------------
# Section 3.4 — "Deuxième composante : les nombres ronds facilement calculables"
# Subset of multiples of 3 that are also cognitively easy to manipulate.
# These are the most natural "round" anchors in the [1, 100] range.
# ---------------------------------------------------------------------------
A_ROUND_3: frozenset[int] = frozenset({15, 30, 45, 60, 75, 90})
GRID_SIZE = 10
AOI_COUNT = GRID_SIZE * GRID_SIZE


GAME_RULES: dict[int, tuple[str, str]] = {
    1: ("BCG+", "2/3 x + 20"),
    2: ("BCG-", "100 - 2/3 x"),
    3: ("BCG-", "100 - 2/3 x"),
    4: ("BCG-", "100 - 2/3 x"),
    5: ("BCG+", "2/3 x + 20"),
    6: ("BCG+", "2/3 x + 20"),
}


def get_game_metadata(game: int | str) -> dict[str, str | int]:
    """Resolve game metadata from a sheet label (e.g. GAME1) or numeric id."""
    if isinstance(game, int):
        game_id = game
    else:
        game_label = str(game).strip().upper()
        if game_label in {"BCG+", "BCG-"}:
            game_id = 1 if game_label == "BCG+" else 2
        else:
            match = re.search(r"(\d+)", game_label)
            if not match:
                raise ValueError(f"Cannot extract game id from '{game}'.")
            game_id = int(match.group(1))

    if game_id not in GAME_RULES:
        raise ValueError(f"Unsupported game id '{game_id}'. Expected one of: {sorted(GAME_RULES)}")

    game_type, rule = GAME_RULES[game_id]
    return {"game_id": game_id, "game_type": game_type, "rule": rule}


def _aoi_grid_coordinates() -> np.ndarray:
    """Return (row, col) coordinates for AOI ids 1..100 on a 10x10 grid."""
    idx = np.arange(AOI_COUNT)
    rows = idx // GRID_SIZE
    cols = idx % GRID_SIZE
    return np.column_stack((rows, cols)).astype(float)


def _gaussian_spatial_kernel(sigma: float) -> np.ndarray:
    """Build a row-normalized Gaussian kernel K(x, y) from Euclidean distance."""
    if sigma <= 0:
        raise ValueError("blur_sigma must be > 0.")

    coords = _aoi_grid_coordinates()
    deltas = coords[:, None, :] - coords[None, :, :]
    dist_sq = np.sum(deltas**2, axis=2)
    kernel = np.exp(-dist_sq / (2.0 * sigma**2))

    # Normalize each row so diffusion is an averaging operator.
    row_sums = kernel.sum(axis=1, keepdims=True)
    return kernel / row_sums


def _minmax_01(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a vector to [0, 1] with numerical safety."""
    v_min = float(np.min(values))
    v_max = float(np.max(values))
    return (values - v_min) / (v_max - v_min + 1e-12)


def salience_static(
    game: int | str,
    beta: float = 0.02,
    lambda1: float = 1.0,
    lambda2: float = 0.5,
    lambda3: float = 0.3,
    lambda_blur: float = 0.4,
    blur_sigma: float = 1.1,
    normalize_components: bool = True,
) -> dict[str, object]:
    """
    Static salience map with three additive components (section 3.4).

    Components
    ----------
    S_fp    : quadratic proximity to the theoretical fixed point x* = 60.
              Captures the basic rationality gradient — AOIs closer to x*
              are more salient.
    S_mod   : binary indicator for all multiples of 3 in [1, 100].
              Captures a broad divisibility heuristic.
    S_round : binary indicator for the cognitively salient round multiples
              of 3: A_round,3 = {15, 30, 45, 60, 75, 90}.
              These numbers are easy to compute and serve as mental anchors.

    S_blur  : spatial diffusion over the 10x10 AOI grid using a Gaussian
              kernel with Euclidean distance and width blur_sigma.

    Optional normalization
    ----------------------
    If normalize_components=True, each component is min-max normalized to [0, 1]
    before weighting so lambda1/lambda2/lambda3/lambda_blur are directly
    comparable in magnitude for interactive tuning.

    Combined salience
    -----------------
    S = lambda1 * S_fp + lambda2 * S_mod + lambda3 * S_round + lambda_blur * S_blur

    A softmax with temperature beta is applied to S to produce the
    probability distribution q over AOIs 1..100.

    Parameters
    ----------
    game    : game id (1..6), sheet label (e.g. "GAME1"), or type ("BCG+").
    beta    : softmax temperature — higher = sharper peak around x*.
    lambda1 : weight for the fixed-point proximity component.
    lambda2 : weight for the multiples-of-3 heuristic.
    lambda3 : weight for the cognitively salient round-number component.
    lambda_blur : weight for the spatial diffusion component.
    blur_sigma  : Gaussian width for spatial diffusion on the 10x10 grid.
    normalize_components : whether to normalize each component before combining.
    """
    x = np.arange(1, AOI_COUNT + 1)

    metadata = get_game_metadata(game)

    # Both BCG+ and BCG- rules used in this project converge to x* = 60.
    x_star = 60.0

    # --- Component 1: Fixed-point proximity (quadratic) -------------------
    # S_fp(x) = -(x - x*)^2
    # Maximum at x = x*, decreasing symmetrically on both sides.
    S_fp = -(x - x_star) ** 2

    # --- Component 2: Multiples-of-3 heuristic ----------------------------
    # S_mod(x) = 1  if x is divisible by 3, else 0.
    # Represents a broad divisibility salience.
    S_mod = (x % 3 == 0).astype(float)

    # --- Component 3: Cognitively salient round numbers (section 3.4) -----
    # S_round(x) = 1  if x ∈ A_round,3 = {15, 30, 45, 60, 75, 90}, else 0.
    # These numbers are both multiples of 3 and easy to manipulate mentally.
    S_round = np.isin(x, sorted(A_ROUND_3)).astype(float)

    # --- Component 3.5: Spatial diffusion around cognitive anchors ----------
    # Formula:
    # K(x, y) = exp(-d(x, y)^2 / (2 * sigma^2))
    # S_blur(x) = sum_y K(x, y) * (S_mod(y) + S_round(y))
    kernel = _gaussian_spatial_kernel(blur_sigma)
    S_anchor = S_mod + S_round
    S_blur = kernel @ S_anchor

    # Normalized versions make lambda effects easier to compare interactively.
    S_fp_norm = _minmax_01(S_fp)
    S_mod_norm = _minmax_01(S_mod)
    S_round_norm = _minmax_01(S_round)
    S_blur_norm = _minmax_01(S_blur)

    if normalize_components:
        c_fp = S_fp_norm
        c_mod = S_mod_norm
        c_round = S_round_norm
        c_blur = S_blur_norm
    else:
        c_fp = S_fp
        c_mod = S_mod
        c_round = S_round
        c_blur = S_blur

    # --- Combined salience ------------------------------------------------
    S = lambda1 * c_fp + lambda2 * c_mod + lambda3 * c_round + lambda_blur * c_blur

    # --- Softmax ----------------------------------------------------------
    # Stabilize exponentials when sliders create a sharper map.
    S_shifted = S - np.max(S)
    exp_S = np.exp(beta * S_shifted)
    q = exp_S / exp_S.sum()

    return {
        # AOI indices 1..100
        "x": x,
        # Theoretical fixed point
        "x_star": x_star,
        # Intermediate salience components (useful for debug / visualisation)
        "S_fp": S_fp,
        "S_mod": S_mod,
        "S_round": S_round,
        "S_blur": S_blur,
        "S_fp_norm": S_fp_norm,
        "S_mod_norm": S_mod_norm,
        "S_round_norm": S_round_norm,
        "S_blur_norm": S_blur_norm,
        # Combined raw salience and final probability distribution
        "S": S,
        "q": q,
        "blur_sigma": blur_sigma,
        "lambda_blur": lambda_blur,
        "normalize_components": normalize_components,
        # Game metadata
        "game_id": metadata["game_id"],
        "game_type": metadata["game_type"],
        "rule": metadata["rule"],
    }
