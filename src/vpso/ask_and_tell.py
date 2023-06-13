import logging

import numpy as np

from vpso.math import pso_equation
from vpso.mutation import mutate
from vpso.reparation import repair_out_of_bounds
from vpso.typing import Array1d, Array2d, Array3d


def generate_offsprings(
    x: Array3d,
    px: Array3d,
    pf: Array2d,
    sx: Array3d,
    v: Array3d,
    v_max: Array3d,
    lb: Array3d,
    ub: Array3d,
    nvec: int,
    dim: int,
    w: Array3d,
    c1: Array3d,
    c2: Array3d,
    repair_iters: int,
    perturb_best: bool,
    mutation_prob: float,
    np_random: np.random.Generator,
) -> tuple[Array3d, Array3d]:
    """Given the current swarm, generates the next generation of the particle swarm.

    Parameters
    ----------
    x : 3d array
        Current positions of the particles. An array of shape `(N, M, d)`, where `N` is
        the number of vectorized problems to solve simultaneously, `M` is the number of
        particles in the swarm, and `d` is the dimension of the search space.
    px : 3d array
        Best positions of the particles so far. An array of shape `(N, M, d)`.
    pf : 2d array
        Best values of the particles so far. An array of shape `(N, M)`.
    sx : 3d array
        Social best, i.e., the best particle so far. An array of shape `(N, 1, d)`.
    v : 3d array
        Current velocities of the particles. An array of shape `(N, M, d)`.
    v_max : 3d array
        Maximum velocities of the particles. An array of shape `(N, 1, d)`.
    lb : 3d array
        Lower bound of the search space. An array of shape `(N, 1, d)`.
    ub : 3d array
        Upper bound of the search space. An array of shape `(N, 1, d)`.
    nvec : int
        Number of vectorized problems.
    dim : int
        Dimensionality of the problem.
    w : 3d array
        Inertia weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c1 : 3d array
        Cognitive weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c2 : 3d array
        Social weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    repair_iters : int
        Number of iterations to try to repair the particles before random sampling.
    perturb_best : bool
        Whether to perturb the best particle or not.
    mutation_prob : float
        Probability of applying the mutation.
    np_random : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple of 3d arrays
        The positions and velocities of the particles in the next generation.
    """
    x_new, v_new = pso_equation(x, px, sx, v, v_max, w, c1, c2, np_random)
    x_new, v_new = repair_out_of_bounds(
        x, x_new, v_new, px, sx, v, v_max, lb, ub, w, c1, c2, repair_iters, np_random
    )
    if perturb_best:
        mutate(x_new, px, pf, lb, ub, nvec, dim, mutation_prob, np_random)
    return x_new, v_new


def advance_population(
    x: Array3d, f: Array2d, px: Array3d, pf: Array2d
) -> tuple[Array3d, Array2d]:
    """Advances the population by replacing the particles with better positions.

    Parameters
    ----------
    x : 3d array
        Current positions of the particles. An array of shape `(N, M, d)`, where `N` is
        the number of vectorized problems to solve simultaneously, `M` is the number of
        particles in the swarm, and `d` is the dimension of the search space.
    f : 2d array
        Current values of the particles. An array of shape `(N, M)`.
    px : 3d array
        Best positions of the particles so far. An array of shape `(N, M, d)`.
    pf : 2d array
        Best values of the particles so far. An array of shape `(N, M)`.

    Returns
    -------
    tuple of 3d and 2d arrays
        Returns the new positions and values of the best particles.
    """
    improvement_mask = f < pf
    px = np.where(improvement_mask[:, :, np.newaxis], x, px)
    pf = np.where(improvement_mask, f, pf)
    return px, pf


def get_best(
    px: Array3d, pf: Array2d, nvec: int, logger: logging.Logger, iter: int
) -> tuple[Array3d, Array1d]:
    """Returns the best particle and its value for each problem.

    Parameters
    ----------
    px : 3d array
        Best positions of the particles so far. An array of shape `(N, M, d)`, where `N`
        is the number of vectorized problems to solve simultaneously, `M` is the number
        of particles in the swarm, and `d` is the dimension of the search space.
    pf : 2d array
        Best values of the particles so far. An array of shape `(N, M)`.
    nvec : int
        Number of vectorized problems.
    logger : logging.Logger
        Logger object.
    iter : int
        Current iteration. Only used for logging.

    Returns
    -------
    tuple of 3d and 1d arrays
        The best particle and its value for each problem with shape `(N, 1, d)` and
        `(N,)`, respectively.
    """
    idx = np.arange(nvec)
    k = pf.argmin(1)
    sx = px[idx, np.newaxis, k]  # (social/global) best particle
    sf = pf[idx, k]  # (social/global) best value

    if logger.level <= logging.INFO:
        logger.info("best values at iteration %i ∈ [%e, %e]", iter, sf.min(), sf.max())
    return sx, sf
