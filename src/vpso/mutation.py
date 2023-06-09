import numpy as np

from vpso.jit import _int, jit
from vpso.typing import Array2d, Array3d


@jit
def polynomial_mutation(
    x_best: Array2d,
    mutation_mask: Array2d,
    lb: Array2d,
    ub: Array2d,
    nvec: int,
    dim: int,
    np_random: np.random.Generator,
) -> Array2d:
    """Performs a polynomial mutation on the best particle of each problem.

    Parameters
    ----------
    x_best : 2d array
        Best particles eligible for mutation. An array of shape `(nvec, dim)`.
    mutation_mask : 2d array
        Boolean mask indicating which particles should be mutated. An array of shape
        `(nvec, dim)`.
    lb : 2d array
        Lower bound of the search space. An array of shape `(nvec, dim)`.
    ub : 2d array
        Upper bound of the search space. An array of shape `(nvec, dim)`.
    nvec : int
        Number of vectorized problems.
    dim : int
        Dimensionality of the problem.
    np_random : np.random.Generator
        Random number generator.

    Returns
    -------
    2d array
        Modified best particles. An array of shape `(nvec, dim)`.
    """
    domain = ub - lb
    eta = np_random.uniform(6.0, 31.0, (nvec, _int(1)))
    mut_pow = 1.0 / eta
    xy1 = np.power((ub - x_best) / domain, eta)
    xy2 = np.power((x_best - lb) / domain, eta)
    R = np_random.random((nvec, dim))
    val1 = np.power(2.0 * R + (1.0 - 2.0 * R) * xy1, mut_pow)
    val2 = np.power(2.0 * (1.0 - R) + 2.0 * (R - 0.5) * xy2, mut_pow)
    deltaq = np.where(R <= 0.5, val1 - 1.0, 1.0 - val2)

    return np.clip(np.where(mutation_mask, x_best + deltaq * domain, x_best), lb, ub)


@jit
def mutate(
    x: Array3d,
    px: Array3d,
    pf: Array2d,
    lb: Array3d,
    ub: Array3d,
    nvec: int,
    dim: int,
    mutation_prob: float,
    np_random: np.random.Generator,
) -> None:
    """Mutates in-place the best particle of each problem with a polynomial mutation.

    Parameters
    ----------
    x : 3d array
        Current positions of the particles. An array of shape `(N, M, d)`, where `N` is
        the number of vectorized problems to solve simultaneously, `M` is the number of
        particles in the swarm, and `d` is the dimension of the search space. This array
        is modified in-place.
    px : 3d array
        Best positions of the particles so far. An array of shape `(N, M, d)`.
    pf : 2d array
        Best values of the particles so far. An array of shape `(N, M)`.
    lb : 3d array
        Lower bound of the search space. An array of shape `(N, 1, d)`.
    ub : 3d array
        Upper bound of the search space. An array of shape `(N, 1, d)`.
    nvec : int
        Number of vectorized problems.
    dim : int
        Dimensionality of the problem.
    mutation_prob : float
        Probability of applying the mutation.
    np_random : np.random.Generator
        Random number generator.
    """
    # get the mutation mask for each vectorized problem, and for each dimension
    mutation_mask = np.logical_and(
        np_random.random((nvec, _int(1))) <= mutation_prob,
        np_random.random((nvec, dim)) <= min(0.5, 1 / dim),
    )
    if not mutation_mask.any():
        return

    # pick the best particle from each problem as target for mutation
    k = pf.argmin(1)
    # x_best = px[np.arange(nvec), k]  # cannot do this in njit
    x_best = np.empty((nvec, dim))
    for i, j in enumerate(k):
        x_best[i] = px[i, j]

    # compute mutation and apply it to the particle
    x_mutated = polynomial_mutation(
        x_best, mutation_mask, lb[:, 0], ub[:, 0], nvec, dim, np_random
    )
    for i, j in enumerate(k):
        x[i, j] = x_mutated[i]
