from typing import Callable, Optional
import numpy as np
import numpy.typing as npt
from scipy.stats.qmc import LatinHypercube
from typing_extensions import TypeAlias


Array: TypeAlias = npt.NDArray[np.floating]


def _initialize_particles(
    swarmsize: int,
    n_problems: int,
    n_vars: int,
    lb: Array,
    ub: Array,
    max_velocity_rate: float,
    seed: Optional[int],
) -> tuple[Array, Array, Array, np.random.Generator]:
    """Initializes particle positions and velocities."""
    lhs_sampler = LatinHypercube(d=n_problems * n_vars, seed=seed)
    np_random = np.random.Generator(np.random.PCG64(seed))
    domain = ub - lb

    x = lb + domain * lhs_sampler.random(swarmsize).reshape(
        swarmsize, n_problems, n_vars
    ).transpose((1, 0, 2))

    v_max = max_velocity_rate * domain
    v = np_random.uniform(0, v_max, (n_problems, swarmsize, n_vars))
    return x, v, v_max, np_random


def _pso_equation(
    x: Array,
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
):
    """Updates particle positions and velocities."""
    r1 = np_random.random(x.shape)
    r2 = np_random.random(x.shape)

    inerta = w * v
    cognitive = c1 * r1 * (px - x)
    social = c2 * r2 * (sx - x)

    v_new = np.clip(inerta + cognitive + social, -v_max, v_max)

    x_new = x + v_new
    maskl = x_new < lb
    masku = x_new > ub
    if maskl.any():
        x_new[maskl] = (lb + np_random.random(x.shape) * (x - lb))[maskl]
    if masku.any():
        x_new[masku] = (ub + np_random.random(x.shape) * (ub - x))[masku]
    return x_new, v_new


def _polynomial_mutation(
    x: Array,
    pf: Array,
    lb: Array,
    ub: Array,
    n_problems: int,
    n_vars: int,
    mutation_prob: float,
    np_random: np.random.Generator,
) -> Array:
    if np_random.random() > mutation_prob:
        return x

    mutation_mask = np_random.random((n_problems, n_vars)) <= min(0.5, 1 / n_vars)
    if not mutation_mask.any():
        return x

    # pick best particle for each problem
    k = pf.argmin(1)
    problems = np.arange(n_problems)
    x_best = x[problems, np.newaxis, k]

    # compute mutations
    domain = ub - lb
    delta = np.asarray(((x_best - lb) / domain, (ub - x_best) / domain))
    eta = np_random.uniform(low=5, high=30, size=(1, n_problems, 1, 1))
    xy = np.power(1 - delta, eta + 1)
    rand = np_random.random((n_problems, 1, n_vars))
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
    mutation_prob: float = 0.9,
    #
    maxiter: int = 300,
    #
    seed: Optional[int] = None,
) -> Array:
    lb = np.expand_dims(lb, 1)  # add swarm dimension
    ub = np.expand_dims(ub, 1)  # add swarm dimension
    n_problems, _, n_vars = lb.shape

    # initialize particle positions and velocities, and best particles
    x, v, v_max, np_random = _initialize_particles(
        swarmsize, n_problems, n_vars, lb, ub, max_velocity_rate, seed
    )
    f = func(x).reshape(n_problems, swarmsize)  # current particles' values
    px = x  # best particles's positions
    pf = f  # best particles's values
    sx = x[np.arange(n_problems), f.argmin(1)]  # social best
    sx = sx[:, np.newaxis].repeat(swarmsize, axis=1)  # add swarm dimension

    # main optimization loop
    for _ in range(maxiter):
        x_new, v_new = _pso_equation(x, px, sx, v, lb, ub, v_max, w, c1, c2, np_random)

        # perturb best solution
        x_new = _polynomial_mutation(
            x_new, pf, lb, ub, n_problems, n_vars, mutation_prob, np_random
        )

        # check termination conditions

    raise NotImplementedError
