from numbers import Number
from typing import Union

import numpy as np
from scipy.stats.qmc import LatinHypercube

from vpso.typing import Array1d, Array1i, Array2d, Array3d, Array3i


def _as3darray(val, nvec, dtype):
    if isinstance(val, Number):
        val = np.full(nvec, val, dtype=dtype)
    return np.reshape(val, (nvec, 1, 1)).astype(dtype, copy=False)


def adjust_dimensions(
    lb: Array2d,
    ub: Array2d,
    max_velocity_rate: Union[float, Array1d],
    w: Union[float, Array1d],
    c1: Union[float, Array1d],
    c2: Union[float, Array1d],
    ftol: Union[float, Array1d],
    xtol: Union[float, Array1d],
    patience: Union[int, Array1i],
) -> tuple[
    Array3d,
    Array3d,
    int,
    int,
    Array3d,
    Array3d,
    Array3d,
    Array3d,
    Array3d,
    Array3d,
    Array3i,
]:
    """Adjusts the dimensions of the input arrays to be compatible with the vectorized
    algorithm, i.e., adds dimensions when necessary or converts to array.

    lb : 2d array
        Lower bound of the search space. An array of shape `(N, d)`.
    ub : 2d array
        Upper bound of the search space. An array of shape `(N, d)`.
    max_velocity_rate : float or array, optional
        Maximum velocity rate used to initialize the particles. By default, `0.2`. Can
        also be an 1d array_like of shape `(N,)` to specify a different value for each
        of the `N` vectorized problems.
    w : float, optional
        Inertia weight. By default, `0.9`. Can
        also be an 1d array_like of shape `(N,)` to specify a different value for each
        of the `N` vectorized problems.
    c1 : float, optional
        Cognitive weight. By default, `2.0`. Can also be an 1d array_like of shape
        `(N,)` to specify a different value for each of the `N` vectorized problems.
    c2 : float, optional
        Social weight. By default, `2.0`. Can also be an 1d array_like of shape `(N,)`
        to specify a different value for each of the `N` vectorized problems.
    ftol : float or 1d array_like of floats, optional
        Tolerance for changes in the objective function value before terminating the
        solver. Can also be an 1d array_like of shape `(N,)` to specify a different
        value for each of the `N` vectorized problems. By default, `1e-8`.
    xtol : float or 1d array_like of floats, optional
        Tolerance for average changes in the objective minimizer before terminating the
        solver. Can also be an 1d array_like of shape `(N,)` to specify a different
        value for each of the `N` vectorized problems. By default, `1e-8`.
    patience : int or 1d array_like of ints, optional
        Number of iterations to wait before terminating the solver if no improvement is
        witnessed. Can also be an 1d array_like of shape `(N,)` to specify a different
        value for each of the `N` vectorized problems. By default, `1`.

    Returns
    -------
    tuple of (3d array, 3d array, int, int, 3d array, 3d array, 3d array, 3d array)
        Returns the `lb`, `ub`, `nvec`, `dim`, `max_velocity_rate`, `w`, `c1`, and `c2`.
    """
    lb = np.expand_dims(lb, 1)  # add swarm dimension
    ub = np.expand_dims(ub, 1)  # add swarm dimension
    nvec, _, dim = lb.shape
    return (
        lb,
        ub,
        nvec,
        dim,
        _as3darray(max_velocity_rate, nvec, float),
        _as3darray(w, nvec, float),
        _as3darray(c1, nvec, float),
        _as3darray(c2, nvec, float),
        _as3darray(ftol, nvec, float),
        _as3darray(xtol, nvec, float),
        _as3darray(patience, nvec, int),
    )


def initialize_particles(
    nvec: int,
    swarmsize: int,
    dim: int,
    lb: Array3d,
    ub: Array3d,
    max_velocity_rate: Array3d,
    lhs_sampler: LatinHypercube,
    np_random: np.random.Generator,
) -> tuple[Array3d, Array3d, Array3d]:
    """Initializes particle positions and velocities.

    Parameters
    ----------
    nvec : int
        Number of vectorized problems.
    swarmsize : int
        Size of the swarm for each problem.
    dim : int
        Dimensionality of the problem.
    lb : 3d array
        Lower bound of the search space. An array of shape `(N, 1, d)`.
    ub : 3d array
        Upper bound of the search space. An array of shape `(N, 1, d)`.
    max_velocity_rate : 3d array
        The maximum velocity rate (proportional to the search domain size) for each
        problem. An array of shape `(N, 1, 1)`.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple of 3d arrays
        Returns a tuple containing the particle positions, velocities, and maximum
        velocities.
    """
    domain = ub - lb

    x = lb + domain * lhs_sampler.random(swarmsize).reshape(
        swarmsize, nvec, dim
    ).transpose((1, 0, 2))

    v_max = max_velocity_rate * domain
    v = np_random.uniform(0, v_max, (nvec, swarmsize, dim))
    return x, v, v_max
