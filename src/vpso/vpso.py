from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.stats.qmc import LatinHypercube
from typing_extensions import TypeAlias

from vpso.jit import _int, jit

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


@jit
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
) -> tuple[Array, Array]:
    """Updates particle positions and velocities."""
    r1 = np_random.random(x.shape)
    r2 = np_random.random(x.shape)

    inerta = w * v
    cognitive = c1 * r1 * (px - x)
    social = c2 * r2 * (sx[:, np.newaxis] - x)

    v_new = np.clip(inerta + cognitive + social, -v_max, v_max)
    x_new = x + v_new
    return x_new, v_new


@jit
def _repair_out_of_bounds(
    x: Array,
    x_new: Array,
    v_new: Array,
    px: Array,
    sx: Array,
    v: Array,
    v_max: Array,
    lb: Array,
    ub: Array,
    w: float,
    c1: float,
    c2: float,
    iters: int,
    np_random: np.random.Generator,
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


@jit
def _polynomial_mutation(
    x: Array,
    px: Array,
    pf: Array,
    lb: Array,
    ub: Array,
    nvec: int,
    dim: int,
    mutation_prob: float,
    np_random: np.random.Generator,
) -> Array:
    """Mutates the best particle of each problem with polynomial mutation."""
    # get the mutation mask for each vectorized problem, and for each dimension
    mutation_mask = np.logical_and(
        np_random.random((nvec, _int(1))) <= mutation_prob,
        np_random.random((nvec, dim)) <= min(0.5, 1 / dim),
    )
    if not mutation_mask.any():
        return x

    # pick the best particle from each problem as target for mutation
    k = pf.argmin(1)
    x_best = np.empty((nvec, dim))
    for i, j in enumerate(k):
        x_best[i] = px[i, j]
    # x_best = px[np.arange(nvec), k]  # cannot do this in njit

    # compute mutation magnitude
    lb, ub = lb[:, 0], ub[:, 0]  # remove swarm dimension
    domain = ub - lb
    eta = np_random.uniform(6.0, 31.0, (nvec, _int(1)))
    mut_pow = 1.0 / eta
    xy1 = np.power((ub - x_best) / domain, eta)
    xy2 = np.power((x_best - lb) / domain, eta)
    R = np_random.random((nvec, dim))
    val1 = np.power(2.0 * R + (1.0 - 2.0 * R) * xy1, mut_pow)
    val2 = np.power(2.0 * (1.0 - R) + 2.0 * (R - 0.5) * xy2, mut_pow)
    deltaq = np.where(R <= 0.5, val1 - 1.0, 1.0 - val2)

    # mutate the target best particle and apply it to the original swarm
    x_mutated = np.clip(
        np.where(mutation_mask, x_best + deltaq * domain, x_best), lb, ub
    )
    for i, j in enumerate(k):
        x[i, j] = x_mutated[i]
    return x


@jit
def _generate_offsprings(
    x: Array,
    px: Array,
    pf: Array,
    sx: Array,
    v: Array,
    v_max: Array,
    lb: Array,
    ub: Array,
    nvec: int,
    dim: int,
    w: float,
    c1: float,
    c2: float,
    repair_iters: int,
    perturb_best: bool,
    mutation_prob: float,
    rng: np.random.Generator,
):
    """Given the current swarm, generates the next generation of the particle swarm."""
    x_new, v_new = _pso_equation(x, px, sx, v, v_max, w, c1, c2, rng)
    x_new, v_new = _repair_out_of_bounds(
        x, x_new, v_new, px, sx, v, v_max, lb, ub, w, c1, c2, repair_iters, rng
    )
    if perturb_best:
        x_new = _polynomial_mutation(
            x_new, px, pf, lb, ub, nvec, dim, mutation_prob, rng
        )
    return x_new, v_new


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
    perturb_best: bool = True,
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
    f = np.reshape(func(x), (nvec, swarmsize))  # particle's current value
    px = x.copy()  # particle's best position
    pf = f.copy()  # particle's best value
    sx = x[np.arange(nvec), f.argmin(1)]  # (social/global) best particle

    # main optimization loop
    for _ in range(maxiter):
        x_new, v_new = _generate_offsprings(
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
            rng,
        )
        f_new = np.reshape(func(x_new), (nvec, swarmsize))

        # TODO: implement advance
        # infills: x_new, v_new, f_new
        # self.pop: ??
        f_new < f
        # assign improved particles to pop

        # update social best
        # sx

        # adapt

        # check termination conditions

    raise NotImplementedError
