import numpy as np

from vpso.jit import jit
from vpso.math import pso_equation
from vpso.mutation import polynomial_mutation
from vpso.reparation import repair_out_of_bounds
from vpso.typing import Array2d, Array3d


@jit
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
    w: float,
    c1: float,
    c2: float,
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
    sx : 2d array
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
    w : float
        Inertia weight.
    c1 : float
        Cognitive weight.
    c2 : float
        Social weight.
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
        polynomial_mutation(
            x_new, px, pf, lb, ub, nvec, dim, mutation_prob, np_random
        )
    return x_new, v_new
