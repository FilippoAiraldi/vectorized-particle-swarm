import logging

import numpy as np

from vpso.typing import Array1d, Array1i, Array2i, Array3d


def update_patience(
    sx: Array3d,
    sf: Array1d,
    sx_new: Array3d,
    sf_new: Array1d,
    lb: Array3d,
    ub: Array3d,
    xtol: Array1d,
    ftol: Array1d,
    current_patience_level: Array2i,
) -> tuple[Array1d, Array1d]:
    """Updates the patience level for each problem as a function of the previous and
    next best particle. The level is increased if tolerances are met, and reset to zero
    if the new best particle is better than the previous one by a margin larger than the
    tolerances. `current_patience_level` is modified in-place.

    Parameters
    ----------
    sx : 3d array
        Social best, i.e., the best particle so far. An array of shape `(N, 1, d)`,
        where `N` is the number of vectorized problems to solve simultaneously, and `d`
        is the dimension of the search space.
    sf : 1d array
        Social best value. An array of shape `(N,)`.
    sx_new : 3d array
        The new social best at the next iteration. An array of shape `(N, 1, d)`.
    sf_new : 1d array
        The new social best value at the next iteration. An array of shape `(N,)`.
    lb : 3d array
        Lower bound of the search space. An array of shape `(N, 1, d)`.
    ub : 3d array
        Upper bound of the search space. An array of shape `(N, 1, d)`.
    xtol : 1d array
        Tolerance for average changes in the objective minimizer before terminating the
        solver. An array of shape `(N,)`.
    ftol : 1d array
        Number of iterations to wait before terminating the solver if no improvement is
        witnessed. An array of shape `(N,)`.
    current_patience_level : Array2i
        Current patience level, i.e., for how many iterations the tolerances have been
        met. An array of shape `(N, 2)`, where the first column corresponds to the
        tolerance in `x`, and the second column in `f`. This array is modified in-place.

    Returns
    -------
    tuple of 1d arrays
        Returns the current termination criteria for `x` and `f`, respectively.
    """
    # NOTE: current patience level is to be modified in-place
    # normalize sx and sx_new and compute normalized euclidean distance (could call
    # batch_cdist here, but it's only one vector per bathc, so this should be faster)
    domain = ub - lb
    sx_norm = sx / domain
    sx_new_norm = sx_new / domain
    D = np.sqrt(np.square(sx_norm - sx_new_norm).sum(2))[:, 0]
    current_patience_level[:, 0] = np.where(
        D <= xtol, current_patience_level[:, 0] + 1, 0
    )

    # compute tolerance for f
    F = np.maximum(0.0, sf - sf_new)
    current_patience_level[:, 1] = np.where(
        F <= ftol, current_patience_level[:, 1] + 1, 0
    )
    return D, F


def termination(
    sx: Array3d,
    sf: Array1d,
    sx_new: Array3d,
    sf_new: Array1d,
    lb: Array3d,
    ub: Array3d,
    xtol: Array1d,
    ftol: Array1d,
    patience: Array1i,
    current_patience_level: Array2i,
    logger: logging.Logger,
) -> tuple[bool, str]:
    """_summary_

    Parameters
    ----------
    sx : 3d array
        Social best, i.e., the best particle so far. An array of shape `(N, 1, d)`,
        where `N` is the number of vectorized problems to solve simultaneously, and `d`
        is the dimension of the search space.
    sf : 1d array
        Social best value. An array of shape `(N,)`.
    sx_new : 3d array
        The new social best at the next iteration. An array of shape `(N, 1, d)`.
    sf_new : 1d array
        The new social best value at the next iteration. An array of shape `(N,)`.
    lb : 3d array
        Lower bound of the search space. An array of shape `(N, 1, d)`.
    ub : 3d array
        Upper bound of the search space. An array of shape `(N, 1, d)`.
    xtol : 1d array
        Tolerance for average changes in the objective minimizer before terminating the
        solver. An array of shape `(N,)`.
    ftol : 1d array
        Number of iterations to wait before terminating the solver if no improvement is
        witnessed. An array of shape `(N,)`.
    patience : int or 1d array_like of ints, optional
        Number of iterations to wait before terminating the solver if no improvement is
        witnessed. An array of shape `(N,)`.
    current_patience_level : Array2i
        Current patience level, i.e., for how many iterations the tolerances have been
        met. An array of shape `(N, 2)`, where the first column corresponds to the
        tolerance in `x`, and the second column in `f`. This array is modified in-place.

    Returns
    -------
    tuple of bool and str
        Returns whether the solver should terminate, and the reason why. The reason can
        be either `xtol` or `ftol`, depending on which tolerance was met first.
    """
    D, F = update_patience(
        sx, sf, sx_new, sf_new, lb, ub, xtol, ftol, current_patience_level
    )
    if logger.level <= logging.DEBUG:
        logger.debug(
            "changes: x ∈ [%e, %e], f ∈ [%e, %e]",
            D.min(),
            D.max(),
            F.min(),
            F.max(),
        )

    if (current_patience_level[:, 0] >= patience).all():
        return True, "xtol"
    if (current_patience_level[:, 1] >= patience).all():
        return True, "ftol"
    return False, ""
