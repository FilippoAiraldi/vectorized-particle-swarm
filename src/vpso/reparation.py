import numpy as np

from vpso.jit import jit
from vpso.math import pso_equation
from vpso.typing import Array3d


@jit
def repair_out_of_bounds(
    x: Array3d,
    x_new: Array3d,
    v_new: Array3d,
    px: Array3d,
    sx: Array3d,
    v: Array3d,
    v_max: Array3d,
    lb: Array3d,
    ub: Array3d,
    w: Array3d,
    c1: Array3d,
    c2: Array3d,
    iters: int,
    np_random: np.random.Generator,
) -> tuple[Array3d, Array3d]:
    """Repairs particles that have gone out of bounds with an iterative process and, if
    it failed, with random re-sampling.

    Parameters
    ----------
    x : 3d array
        Current positions of the particles. An array of shape `(N, M, d)`, where `N` is
        the number of vectorized problems to solve simultaneously, `M` is the number of
        particles in the swarm, and `d` is the dimension of the search space.
    x_new : 3d array
        New candidate positions of the particles. An array of shape `(N, M, d)`.
    v_new : 3d array
        new candidate velocities of the particles. An array of shape `(N, M, d)`.
    px : 3d array
        Best positions of the particles so far. An array of shape `(N, M, d)`.
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
    w : 3d array
        Inertia weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c1 : 3d array
        Cognitive weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c2 : 3d array
        Social weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    iters : int
        Number of iterations to try to repair the particles before random sampling.
    np_random : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple of 3d arrays
        Repaired positions and velocities of the particles.
    """
    lmask = x_new < lb
    umask = x_new > ub
    any_lmask = lmask.any()
    any_umask = umask.any()
    if not any_lmask and not any_umask:
        return x_new, v_new

    for _ in range(iters):
        mask = np.logical_or(lmask, umask)
        x_new_, v_new_ = pso_equation(x, px, sx, v, v_max, w, c1, c2, np_random)
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
