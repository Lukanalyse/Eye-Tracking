import numpy as np

EPS = 1e-12 #Eviter log 0

def logsumexp(a: np.ndarray,
              axis=None, #On fait la somme exp sur tout (pas par ligne ou par colonne dans la matrice
              keepdims=False) -> np.ndarray:
    """
        Stable computation of log(sum(exp(a))).

        Parameters
        ----------
        a : np.ndarray
            Input array (log-values).
        axis : int or None
            Axis over which the sum is computed.
        keepdims : bool
            Whether to keep dimensions.

        Returns
        -------
        np.ndarray
            log-sum-exp of the input.
        """

    a_max = np.max(a, axis = axis, keepdims = True)
    out= a_max + np.log(np.sum(np.exp(a-a_max), axis = axis, keepdims = keepdims)+ EPS)
    if not keepdims and axis is not None:
        out = np.squeeze(out, axis)
        return out