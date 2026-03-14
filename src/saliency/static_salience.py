import numpy as np


def salience_static(game, beta=0.02, lambda1=1.0, lambda2=0.5):
    """
    Static salience map based on theoretical fixed point (convergence point)
    + optional heuristic favouring multiples of 3.

    game: 1 (BCG), 2 (BCG+), 3 (BCG-)
    """
    x = np.arange(1, 101)

    # Theoretical fixed points
    if game == 1:
        x_star = 0.0
    elif game in (2, 3):
        x_star = 60.0
    else:
        raise ValueError("Game must be 1, 2, or 3.")

    # Fixed-point proximity (quadratic)
    S_fp = -(x - x_star) ** 2

    # Multiples-of-3 heuristic
    S_mod = (x % 3 == 0).astype(float)

    # Combined salience
    S = lambda1 * S_fp + lambda2 * S_mod

    # Softmax
    exp_S = np.exp(beta * S)
    q = exp_S / exp_S.sum()

    return {"x": x, "S": S, "q": q, "x_star": x_star}