import logging
from typing import Callable, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.qmc import LatinHypercube

from vpso.adaptation import adapt
from vpso.ask_and_tell import advance_population, generate_offsprings, get_best
from vpso.initialization import adjust_dimensions, initialize_particles
from vpso.typing import Array1d, Array2d, Array3d

logger = logging.getLogger(__name__)


def vpso(
    func: Callable[[Array3d], ArrayLike],
    lb: Array2d,
    ub: Array2d,
    #
    swarmsize: int = 25,
    max_velocity_rate: Union[float, Array1d] = 0.2,
    w: Union[float, Array1d] = 0.9,
    c1: Union[float, Array1d] = 2.0,
    c2: Union[float, Array1d] = 2.0,
    #
    repair_iters: int = 20,
    perturb_best: bool = True,
    mutation_prob: float = 0.9,
    adaptive: bool = True,
    #
    maxiter: int = 300,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    patience: int = 10,
    #
    seed: Optional[int] = None,
    verbosity: int = logging.WARNING,
) -> tuple[Array2d, Array1d, str]:
    """Vectorized Particle Swarm Optimization (VPSO). This implementation of PSO is able
    to solve multiple optimization problems simultaneously in a vectorized fashion.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must take a 3d array of shape
        `(N, M, d)` as input, and return an output that can be converted to an array of
        shape `(N, M)`, where `N` is the number of vectorized problems to solve, `M` is
        the `swarmsize`, and `d` is the dimension of the search space.
    lb : 2d array
        Lower bound of the search space. An array of shape `(N, d)`. Can be different
        in each of the `N` vectorized problems.
    ub : 2d array
        Upper bound of the search space. An array of shape `(N, d)`.
    swarmsize : int, optional
        Number of particles in the swarm to solve each problem. By default, `25`.
    max_velocity_rate : float or array, optional
        Maximum velocity rate used to initialize the particles. By default, `0.2`. Can
        also be an 1d array_like of shape `(N,)` to specify a different value for each
        of the `N` vectorized problems.
    w : float or 1d array, optional
        Inertia weight. By default, `0.9`. Can
        also be an 1d array_like of shape `(N,)` to specify a different value for each
        of the `N` vectorized problems.
    c1 : float or 1d array, optional
        Cognitive weight. By default, `2.0`. Can also be an 1d array_like of shape
        `(N,)` to specify a different value for each of the `N` vectorized problems.
    c2 : float or 1d array, optional
        Social weight. By default, `2.0`. Can also be an 1d array_like of shape `(N,)`
        to specify a different value for each of the `N` vectorized problems.
    repair_iters : int, optional
        Number of iterations to repair particles that are outside bounds. If this
        reparation fails, the particle is randomly re-sampled. By default, `20`.
    perturb_best : bool, optional
        Whether to perturb the best particle in the swarm at each iteration.
        By default, `True`.
    mutation_prob : float, optional
        Probability of mutating the best particle in the swarm at each iteration. By
        default, `0.9`. Only used if `perturb_best=True`.
    adaptive : bool, optional
        Whether to adapt the weights at each iteration. By default, `True`.
    maxiter : int, optional
        Maximum number of iterations to run the optimization for. By default, `300`.
    seed : int, optional
        Seed for the random number generator. By default, `None`.
    verbosity : int, optional
        Verbosity level of the solver. Levels higher than `INFO` log nothig. By default,
        `logging.WARNING` is used.

    Returns
    -------
    tuple of (2d array, 1d array, str)
        Returns a tuple containing
         - the best minimizer of each problem
         - the best minimum of each problem
         - the termination reason as a string
    """
    logger.setLevel(verbosity)

    # first, adjust some dimensions
    lb, ub, nvec, dim, max_velocity_rate, w, c1, c2 = adjust_dimensions(
        lb, ub, max_velocity_rate, w, c1, c2
    )

    # initialize particle positions and velocities
    lhs_sampler = LatinHypercube(d=nvec * dim, seed=seed)
    np_random = np.random.Generator(np.random.PCG64(seed))
    x, v, v_max = initialize_particles(
        nvec, swarmsize, dim, lb, ub, max_velocity_rate, lhs_sampler, np_random
    )

    # initialize particle's best pos/value and global best
    px = x  # particle's best position
    pf = np.reshape(func(x), (nvec, swarmsize))  # particle's best value
    sx, sf = get_best(px, pf, nvec, logger, 0)  # social/global best position/value

    # main optimization loop
    termination_reason = "maxiter"
    for i in range(1, maxiter + 1):
        x, v = generate_offsprings(
            x,
            px,
            pf,
            sx,
            v,
            v_max,
            lb,
            ub,
            nvec,
            dim,
            w,
            c1,
            c2,
            repair_iters,
            perturb_best,
            mutation_prob,
            np_random,
        )
        f = np.reshape(func(x), (nvec, swarmsize))  # evaluate particles (non-jittable)
        px, pf = advance_population(x, f, px, pf)
        sx, sf = get_best(px, pf, nvec, logger, i)
        if adaptive:
            w, c1, c2 = adapt(
                px,
                sx,
                swarmsize,
                lb,
                ub,
                w,
                c1,
                c2,
                np_random,
                logger,
            )

        # TODO: check termination conditions

    logger.info('termination due to "%s"', termination_reason)
    return sx[:, 0], sf, termination_reason
