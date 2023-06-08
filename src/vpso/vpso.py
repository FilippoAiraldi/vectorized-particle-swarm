from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from scipy.stats.qmc import LatinHypercube
from typing_extensions import TypeAlias
from numba import njit


Array: TypeAlias = npt.NDArray[np.floating]


def _initialize_particles(
    swarmsize: int,
    nvec: int,
    dim: int,
    lb: Array,
    ub: Array,
    max_velocity_rate: float,
    seed: Optional[int],
) -> tuple[Array, Array, Array, np.random.Generator]:
    """Initializes particle positions and velocities."""
    lhs_sampler = LatinHypercube(d=nvec * dim, seed=seed)
    np_random = np.random.Generator(np.random.PCG64(seed))
    domain = ub - lb

    x = lb + domain * lhs_sampler.random(swarmsize).reshape(
        swarmsize, nvec, dim
    ).transpose((1, 0, 2))

    v_max = max_velocity_rate * domain
    v = np_random.uniform(0, v_max, (nvec, swarmsize, dim))
    return x, v, v_max, np_random


@njit
def _pso_equation(
    x: Array,
    px: Array,
    sx: Array,
    v: Array,
    v_max: Array,
    w: float,
    c1: float,
    c2: float,
    np_random: np.random.Generator,
):
    """Updates particle positions and velocities."""
    r1 = np_random.random(x.shape)
    r2 = np_random.random(x.shape)

    inerta = w * v
    cognitive = c1 * r1 * (px - x)
    social = c2 * r2 * (sx[:, np.newaxis] - x)

    v_new = np.clip(inerta + cognitive + social, -v_max, v_max)
    x_new = x + v_new
    return x_new, v_new


@njit
def _repair_out_of_bounds(
    x: Array,
    x_new: Array,
    v_new: Array,
    px: Array,
    sx: Array,
    v: Array,
    lb: Array,
    ub: Array,
    v_max: Array,
    w: float,
    c1: float,
    c2: float,
    np_random: np.random.Generator,
    iters: int,
) -> tuple[Array, Array]:
    """Repairs particles that have gone out of bounds with an iterative process and, if
    it failed, with random re-sampling."""
    lmask = x_new < lb
    umask = x_new > ub
    if not lmask.any() and not umask.any():
        return x_new, v_new

    for _ in range(iters):
        mask = lmask | umask
        x_new_, v_new_ = _pso_equation(x, px, sx, v, v_max, w, c1, c2, np_random)
        x_new = np.where(mask, x_new_, x_new)
        v_new = np.where(mask, v_new_, v_new)
        lmask = x_new < lb
        umask = x_new > ub
        any_lmask = lmask.any()
        any_umask = umask.any()
        if not any_lmask and not any_umask:
            return x_new, v_new

    if any_lmask:
        x_new = np.where(lmask, lb + np_random.random(x.shape) * (x - lb), x_new)
    if any_umask:
        x_new = np.where(umask, ub - np_random.random(x.shape) * (ub - x), x_new)
    return x_new, v_new


def _polynomial_mutation(
    x: Array,
    pf: Array,
    lb: Array,
    ub: Array,
    nvec: int,
    dim: int,
    mutation_prob: float,
    np_random: np.random.Generator,
) -> Array:
    if np_random.random() > mutation_prob:
        return x

    mutation_mask = np_random.random((nvec, dim)) <= min(0.5, 1 / dim)
    if not mutation_mask.any():
        return x

    # pick best particle for each problem
    k = pf.argmin(1)
    problems = np.arange(nvec)
    x_best = x[problems, np.newaxis, k]

    # compute mutations
    domain = ub - lb
    delta = np.asarray(((x_best - lb) / domain, (ub - x_best) / domain))
    eta = np_random.uniform(low=5, high=30, size=(1, nvec, 1, 1))
    xy = np.power(1 - delta, eta + 1)
    rand = np_random.random((nvec, 1, dim))
    d = np.power(
        np.asarray(
            (
                2 * rand + (1 - 2 * rand) * xy[0],
                2 * (1 - rand) + 2 * (rand - 0.5) * xy[1],
            )
        ),
        1.0 / (eta + 1.0),
    )
    deltaq = np.where(rand <= 0.5, d[0] - 1, 1 - d[1])
    x_mutated = np.clip(x_best + deltaq * domain, lb, ub)

    # apply mutations
    x[problems, k]

    pass


def vpso(
    func: Callable[[Array], Array],
    lb: Array,
    ub: Array,
    #
    swarmsize: int = 25,
    max_velocity_rate: float = 0.2,
    w: float = 0.9,
    c1: float = 2.0,
    c2: float = 2.0,
    #
    repair_iters: int = 20,
    mutation_prob: float = 0.9,
    #
    maxiter: int = 300,
    #
    seed: Optional[int] = None,
) -> Array:
    lb = np.expand_dims(lb, 1)  # add swarm dimension
    ub = np.expand_dims(ub, 1)  # add swarm dimension
    nvec, _, dim = lb.shape

    # initialize particle positions and velocities
    x, v, v_max, rng = _initialize_particles(
        swarmsize, nvec, dim, lb, ub, max_velocity_rate, seed
    )

    # initialize other quantities
    # TODO: to understand how these are saved after each iteration
    f = func(x).reshape(nvec, swarmsize)  # particle's current value
    px = x.copy()  # particle's best position
    pf = f.copy()  # particle's best value
    sx = x[np.arange(nvec), f.argmin(1)]  # (social/global) best particle

    # main optimization loop
    for _ in range(maxiter):
        x_new, v_new = _pso_equation(x, px, sx, v, v_max, w, c1, c2, rng)
        x_new, v_new = _repair_out_of_bounds(
            x, x_new, v_new, px, sx, v, lb, ub, v_max, w, c1, c2, rng, repair_iters
        )

        # perturb best solution
        x_new = _polynomial_mutation(
            x_new, pf, lb, ub, n_problems, n_vars, mutation_prob, np_random
        )

        # check termination conditions

    raise NotImplementedError
