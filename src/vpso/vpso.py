from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.qmc import LatinHypercube

from vpso.adaptation import adapt
from vpso.ask_and_tell import advance_population, generate_offsprings, get_best
from vpso.initialization import adjust_dimensions as adj_dim
from vpso.initialization import initialize_particles
from vpso.termination import termination
from vpso.typing import Array1d, Array1i, Array2d, Array3d


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
    maxiter: int = 400,
    maxevals: Union[int, float] = float("inf"),
    ftol: Union[float, Array1d] = 1e-9,
    xtol: Union[float, Array1d] = 1e-9,
    patience: Union[int, Array1i] = 30,
    #
    seed: Union[None, int, np.random.SeedSequence, np.random.Generator] = None,
) -> tuple[Array2d, Array1d, str]:
    """Vectorized Particle Swarm Optimization (VPSO). This implementation of PSO is able
    to solve multiple optimization problems simultaneously in a vectorized fashion. It
    runs the adaptive version of PSO based on [1].

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
    max_velocity_rate : float or array_like, optional
        Maximum velocity rate used to initialize the particles. By default, `0.2`. Can
        also be an 1d array_like of shape `(N,)` to specify a different value for each
        of the `N` vectorized problems.
    w : float or 1d array_like, optional
        Inertia weight. By default, `0.9`. Can
        also be an 1d array_like of shape `(N,)` to specify a different value for each
        of the `N` vectorized problems.
    c1 : float or 1d array_like, optional
        Cognitive weight. By default, `2.0`. Can also be an 1d array_like of shape
        `(N,)` to specify a different value for each of the `N` vectorized problems.
    c2 : float or 1d array_like, optional
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
    maxiter : int , optional
        Maximum number of iterations to run the optimization for. By default, `300`.
    maxevals : int or float(inf)
        Maximum number of evaluations of `f`, after which the algorithm is terminated.
        By default, `+inf`.
    ftol : float or 1d array_like of floats, optional
        Tolerance for changes in the objective function value before terminating the
        solver. Can also be an 1d array_like of shape `(N,)` to specify a different
        value for each of the `N` vectorized problems. By default, `1e-8`. Pass a
        negative value to disable this check.
    xtol : float or 1d array_like of floats, optional
        Tolerance for average changes in the objective minimizer before terminating the
        solver. Can also be an 1d array_like of shape `(N,)` to specify a different
        value for each of the `N` vectorized problems. By default, `1e-8`. Pass a
        negative value to disable this check.
    patience : int or 1d array_like of ints, optional
        Number of iterations to wait before terminating the solver if no improvement is
        witnessed. Can also be an 1d array_like of shape `(N,)` to specify a different
        value for each of the `N` vectorized problems. By default, `1`.
    seed : int or generator, optional
        Seed for the random number generator, or a generator. By default, `None`.

    Returns
    -------
    tuple of (2d array, 1d array, str)
        Returns a tuple containing
         - the best minimizer of each problem
         - the best minimum of each problem
         - the termination reason as a string

    References
    ----------
    [1] Z. H. Zhan, J. Zhang, Y. Li and H. S. H. Chung, "Adaptive Particle Swarm
        Optimization," in IEEE Transactions on Systems, Man, and Cybernetics, Part B
        (Cybernetics), vol. 39, no. 6, pp. 1362-1381, Dec. 2009,
        doi: 10.1109/TSMCB.2009.2015956.
    """
    # first, adjust some dimensions
    lb, ub, nvec, dim, max_velocity_rate, w, c1, c2, ftol, xtol, patience = adj_dim(
        lb, ub, max_velocity_rate, w, c1, c2, ftol, xtol, patience
    )

    # initialize particle positions and velocities
    np_random = np.random.default_rng(seed)
    lhs_sampler = LatinHypercube(d=nvec * dim, seed=np_random)
    x, v, v_max = initialize_particles(
        nvec, swarmsize, dim, lb, ub, max_velocity_rate, lhs_sampler, np_random
    )

    # initialize particle's best pos/value and global best
    px = x  # particle's best position
    pf = np.reshape(func(x), (nvec, swarmsize))  # particle's best value
    sx, sf = get_best(px, pf, nvec)  # social/global best position/value

    # main optimization loop
    patience_level = np.zeros((nvec, 2), dtype=np.int32)  # 2 level for x and for f
    evals = swarmsize
    termination_reason = "maxiter"
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
        f = np.reshape(func(x), (nvec, swarmsize))  # evaluate particles (non-jittable)
        px, pf = advance_population(x, f, px, pf)
        sx_new, sf_new = get_best(px, pf, nvec)

        # DEBUG
        # px.flags.writeable = (
        #     pf.flags.writeable
        # ) = sx.flags.writeable = sf.flags.writeable = False
        # assert (sf_new <= sf).all()

        if adaptive:
            w, c1, c2 = adapt(px, sx_new, nvec, swarmsize, lb, ub, w, c1, c2, np_random)

        # check termination conditions
        should_terminate, reason, range_min, range_max = termination(
            sx, sf, sx_new, sf_new, lb, ub, xtol, ftol, patience, patience_level
        )
        if should_terminate:
            termination_reason = f"{reason} ∈ [{range_min:e}, {range_max:e}]"
            break
        evals += swarmsize
        if evals >= maxevals:
            termination_reason = "maxevals"
            break

        # save new best
        sx = sx_new
        sf = sf_new
    return sx[:, 0], sf, termination_reason
