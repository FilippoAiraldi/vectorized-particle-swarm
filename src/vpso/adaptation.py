"""
Implements the adaptation mechanism of PSO described in [1].

References
----------
[1] Z. H. Zhan, J. Zhang, Y. Li and H. S. H. Chung, "Adaptive Particle Swarm
    Optimization," in IEEE Transactions on Systems, Man, and Cybernetics, Part B
    (Cybernetics), vol. 39, no. 6, pp. 1362-1381, Dec. 2009,
    doi: 10.1109/TSMCB.2009.2015956.
"""

import logging

import numpy as np

from vpso.jit import _int, jit
from vpso.math import batch_cdist, batch_pdist, batch_squareform
from vpso.typing import Array1d, Array2d, Array3d


@jit
def adaptation_strategy(f: Array1d) -> Array2d:
    """Picks the adaptation strategy for each problem based on the ratio of average
    distances between particles and to the best particle.

    Parameters
    ----------
    f : 1d array
        Array of shape `N`.

    Returns
    -------
    2d array
        Array of shape `(N, 2)`. Each row contains the adaptation strategy for the
        `i`-th problem. The first column contains the adaptation strategy for `c1` and
        the second column contains the adaptation strategy for `c2`.
    """
    f = f[:, np.newaxis]  # add a dimension for broadcasting

    # NOTE: fails in numba's npython mode
    # deltas = np.full((f.size, 2), (-1.0, 1.0), dtype=f.dtype)  # initialize all to S4
    deltas = np.full((f.size, 2), -1.0)  # initialize all to S4 (fill in two steps)
    deltas[:, 1].fill(1.0)

    deltas = np.where(f <= 23 / 30, [(1.0, -1.0)], deltas)  # S1
    deltas = np.where(f < 0.5, [(0.5, -0.5)], deltas)  # S2
    deltas = np.where(f < 7 / 30, [(0.5, 0.5)], deltas)  # S3
    return deltas


@jit
def perform_adaptation(
    nvec: int,
    w: Array3d,
    c1: Array3d,
    c2: Array3d,
    stage: Array1d,
    np_random: np.random.Generator,
) -> tuple[Array3d, Array3d, Array3d]:
    """Performs the adaptation of the parameters `w`, `c1` and `c2` based on the
    stage of the algorithm.

    Parameters
    ----------
    nvec : int
        Number of vectorized problems.
    w : 3d array
        Inertia weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c1 : 3d array
        Cognitive weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c2 : 3d array
        Social weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    stage : 1d array
        Current stage of the algorithm. An array of shape `(N,)`, where each element
        corresponds to a problem.
    np_random : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple of 3d arrays
        The newly adapted parameters `w`, `c1` and `c2`.
    """
    # adapt w
    w = (1 / (1 + 1.5 * np.exp(-2.6 * stage)))[:, np.newaxis, np.newaxis]

    # adapt c1 and c2
    deltas = adaptation_strategy(stage) * np_random.uniform(
        0.05, 0.1, size=(nvec, _int(1))
    )
    c1 = (c1 + deltas[:, 0, np.newaxis, np.newaxis]).clip(1.5, 2.5)
    c2 = (c2 + deltas[:, 1, np.newaxis, np.newaxis]).clip(1.5, 2.5)
    sum_c = c1 + c2
    mask = sum_c > 4
    c1 = np.where(mask, 4 * c1 / sum_c, c1)
    c2 = np.where(mask, 4 * c2 / sum_c, c2)
    return w, c1, c2


def adapt(
    px: Array3d,
    sx: Array3d,
    nvec: int,
    swarmsize: int,
    lb: Array3d,
    ub: Array3d,
    w: Array3d,
    c1: Array3d,
    c2: Array3d,
    np_random: np.random.Generator,
    logger: logging.Logger,
) -> tuple[Array3d, Array3d, Array3d]:
    """Adapts the parameters `w`, `c1` and `c2` on-line.

    Parameters
    ----------
    px : 3d array
        Best positions of the particles so far. An array of shape `(N, M, d)`.
    sx : 3d array
        Social best, i.e., the best particle so far. An array of shape `(N, 1, d)`.
    nvec : int
        Number of vectorized problems.
    swarmsize : int
        Number of particles in the swarm.
    lb : 3d array
        Lower bound of the search space. An array of shape `(N, 1, d)`.
    ub : 3d array
        Upper bound of the search space. An array of shape `(N, 1, d)`.
    w : 3d array
        Inertia weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c1 : 3d array
        Cognitive weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c2 : 3d array
        Social weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    np_random : np.random.Generator
        Random number generator.
    logger : logging.Logger
        Logger object.

    Returns
    -------
    tuple of 3d arrays
        The newly adapted parameters `w`, `c1` and `c2`.
    """
    domain = ub - lb
    px_normalized = px / domain
    sx_normalized = sx / domain
    D = batch_squareform(batch_pdist(px_normalized)).sum(2) / (swarmsize - 1)
    Dmin = D.min(1)
    Dmax = D.max(1)
    G = batch_cdist(px_normalized, sx_normalized).mean((1, 2))
    stage = (G - Dmin) / (Dmax - Dmin + 1e-32)
    w_new, c1_new, c2_new = perform_adaptation(nvec, w, c1, c2, stage, np_random)

    if logger.level <= logging.DEBUG:
        logger.debug(
            "adaptation: w ∈ [%e, %e], c1 ∈ [%e, %e], c2 ∈ [%e, %e]",
            w_new.min(),
            w_new.max(),
            c1_new.min(),
            c1_new.max(),
            c2_new.min(),
            c2_new.max(),
        )

    return w_new, c1_new, c2_new
