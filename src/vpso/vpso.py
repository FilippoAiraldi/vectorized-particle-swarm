from typing import Callable, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.qmc import LatinHypercube

from vpso.ask_and_tell import generate_offsprings
from vpso.initialization import initialize_particles
from vpso.typing import Array1d, Array2d, Array3d

# def _improve_population(
#     x: Array, f: Array, px: Array, pf: Array, sx: Array, adaptive: bool
# ):
#     improvement_mask = f < pf
#     px = np.where(improvement_mask, x, px)
#     pf = np.where(improvement_mask, f, pf)


def vpso(
    func: Callable[[Array3d], ArrayLike],
    lb: Array2d,
    ub: Array2d,
    #
    swarmsize: int = 25,
    max_velocity_rate: float = 0.2,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
    #
    repair_iters: int = 20,
    perturb_best: bool = True,
    mutation_prob: float = 0.9,
    adaptive: bool = True,
    #
    maxiter: int = 300,
    #
    seed: Optional[int] = None,
) -> tuple[Array2d, Array1d]:
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
    max_velocity_rate : float, optional
        Maximum velocity rate used to initialize the particles. By default, `0.2`.
    w : float, optional
        Inertia weight. By default, `0.9`.
    c1 : float, optional
        Cognitive weight. By default, `2.0`.
    c2 : float, optional
        Social weight. By default, `2.0`.
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

    Returns
    -------
    tuple of (2d array, 1d array)
        Returns a tuple containing the best minimizer and minimum of each problem.
    """
    lb = np.expand_dims(lb, 1)  # add swarm dimension
    ub = np.expand_dims(ub, 1)  # add swarm dimension
    nvec, _, dim = lb.shape

    # initialize particle positions and velocities
    lhs_sampler = LatinHypercube(d=nvec * dim, seed=seed)
    np_random = np.random.Generator(np.random.PCG64(seed))
    x, v, v_max = initialize_particles(
        nvec, swarmsize, dim, lb, ub, max_velocity_rate, lhs_sampler, np_random
    )

    # initialize other quantities
    px = x  # particle's best position
    pf = np.reshape(func(x), (nvec, swarmsize))  # particle's best value
    sx = x[np.arange(nvec), np.newaxis, pf.argmin(1)]  # (social/global) best particle

    # main optimization loop
    for _ in range(maxiter):
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

        # evaluate new particles (cannot be jitted)
        f = np.reshape(func(x), (nvec, swarmsize))

        # TODO: implement advance

        # # improve population
        # improvement_mask = f < pf
        # px = np.where(improvement_mask, x, px)
        # pf = np.where(improvement_mask, f, pf)

        # update social best (here or before?)
        # sx

        # adapt

        # check termination conditions

    raise NotImplementedError
