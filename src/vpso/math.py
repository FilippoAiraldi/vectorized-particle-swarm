from typing import Literal

import numpy as np
from numba import prange

from vpso.jit import jit
from vpso.typing import Array2d, Array3d


@jit()
def cdist_func(
    x: Array2d, y: Array2d, type: Literal["euclidean", "sqeuclidean"]
) -> Array2d:
    """Computes the distance matrix for 2D arrays `x` and `y`.

    Parameters
    ----------
    x : 2d array
        A 2D array of shape `(N, d)`.
    y : 2d array
        A 2D array of shape `(M, d)`.
    type : {"euclidean", "sqeuclidean"}
        Type of distance to compute.

    Returns
    -------
    2d array
        Distance matrix of shape `(N, M)` between each pair of entries in `x` and `y`.
    """
    x2_sum = np.square(x).sum(axis=1)
    y2_sum = np.square(y).sum(axis=1)
    xy_dot = np.dot(x, y.T)
    D = np.maximum(x2_sum[:, np.newaxis] + y2_sum - 2 * xy_dot, 0.0)
    if type == "euclidean":
        return np.sqrt(D)
    if type == "sqeuclidean":
        return D
    raise ValueError(f"unknown cdist type: {type}")


@jit()
def pdist_func(x: Array2d, type: Literal["euclidean", "sqeuclidean"]) -> Array2d:
    """Computes the pairwise distance matrix of the elements in the 2D array `x`.

    Parameters
    ----------
    x : 2d array
        A 2D array of shape `(N, d)`.
    type : {"euclidean", "sqeuclidean"}
        Type of distance to compute.

    Returns
    -------
    2d array
        Symmetric distance matrix of shape `(N, N)` between each pair of entries in `x`.
    """
    x2_sum = np.square(x).sum(axis=1)
    xx_dot = np.dot(x, x.T)
    D = np.maximum(x2_sum[:, np.newaxis] + x2_sum - 2 * xx_dot, 0.0)
    np.fill_diagonal(D, 0.0)
    if type == "euclidean":
        return np.sqrt(D)
    if type == "sqeuclidean":
        return D
    raise ValueError(f"unknown pdist type: {type}")


@jit(parallel=True)
def batch_cdist(
    X: Array3d, Y: Array3d, type: Literal["euclidean", "sqeuclidean"]
) -> Array3d:
    """Computes the distance matrices for 3D arrays.

    Parameters
    ----------
    X : 3d array
        An array of shape `(N, M, d)`.
    Y : 3d array
        An array of shape `(N, K, d)`.
    dist_func : callable, optional
        Distance function to use. By default, `scipy.spatial.distance.cdist_euclidean`
        is used. It must support the `out` argument.

    Returns
    -------
    3d array
        Distance matrices between each of the `(M, d)` and `(K, d)` matrices, where `d`
        is assumed to be the axis over which the distance is computed. The output has
        thus shape (N, M, K).
    """
    B, N, _ = X.shape
    out = np.empty((B, N, Y.shape[1]), dtype=X.dtype)
    for i in prange(B):
        out[i] = cdist_func(X[i], Y[i], type)
    return out


@jit(parallel=True)
def batch_pdist(X: Array3d, type: Literal["euclidean", "sqeuclidean"]) -> Array3d:
    """Computes the pairwise distance matrices for the entries of a 3D array.

    Parameters
    ----------
    X : 3d array
        An array of shape `(N, M, d)`.
    dist_func : callable, optional
        Distance function to use. By default, `scipy.spatial.distance.pdist_euclidean`
        is used. It must support the `out` argument.

    Returns
    -------
    3d array
        Distance matrix of shape `(N, M, M)` between each pair of entries of
        the `(M, d)` matrices, where `d` is assumed to be the axis over which the
        distance is computed.
    """
    B, N, _ = X.shape
    out = np.empty((B, N, N), dtype=X.dtype)
    for i in prange(B):
        out[i] = pdist_func(X[i], type)
    return out


@jit(parallel=True)
def batch_cdist_and_pdist(
    X: Array3d, Y: Array3d, type: Literal["euclidean", "sqeuclidean"]
) -> tuple[Array3d, Array3d]:
    """Computes the distance matrices between the 3D arrays `X` and `Y`, as well as the
    pairwise distance matrices for the entries of `X`.

    Parameters
    ----------
    X : 3d array
        An array of shape `(N, M, d)`.
    Y : 3d array
        An array of shape `(N, K, d)`.
    dist_func : callable, optional
        Distance function to use. By default, `scipy.spatial.distance.cdist_euclidean`
        is used. It must support the `out` argument.

    Returns
    -------
    3d array
        Distance matrices between each of the `(M, d)` and `(K, d)` matrices, where `d`
        is assumed to be the axis over which the distance is computed. The output has
        thus shape (N, M, K).
    """
    B, N, _ = X.shape
    out = np.empty((B, N, N), dtype=X.dtype)
    for i in prange(B):
        out[i] = pdist_func(X[i], type)
    return out


@jit()
def pso_equation(
    x: Array3d,
    px: Array3d,
    sx: Array3d,
    v: Array3d,
    v_max: Array3d,
    w: Array3d,
    c1: Array3d,
    c2: Array3d,
    np_random: np.random.Generator,
) -> tuple[Array3d, Array3d]:
    """Computes the new positions and velocities of particles in a PSO algorithm.

    Parameters
    ----------
    x : 3d array
        Current positions of the particles. An array of shape `(N, M, d)`, where `N` is
        the number of vectorized problems to solve simultaneously, `M` is the number of
        particles in the swarm, and `d` is the dimension of the search space.
    px : 3d array
        Best positions of the particles so far. An array of shape `(N, M, d)`.
    sx : 3d array
        Social best, i.e., the best particle so far. An array of shape `(N, 1, d)`.
    v : 3d array
        Current velocities of the particles. An array of shape `(N, M, d)`.
    v_max : 3d array
        Maximum velocities of the particles. An array of shape `(N, 1, d)`.
    w : 3d array
        Inertia weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c1 : 3d array
        Cognitive weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    c2 : 3d array
        Social weight. An array of shape `(N, 1, 1)`, where each element is used for
        the corresponding problem.
    np_random : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple of 3d arrays
        New positions and velocities of the particles.
    """
    r1 = np_random.random(x.shape)
    r2 = np_random.random(x.shape)

    inerta = w * v
    cognitive = c1 * r1 * (px - x)
    social = c2 * r2 * (sx - x)

    v_new = (inerta + cognitive + social).clip(-v_max, v_max)
    x_new = x + v_new
    return x_new, v_new  # type: ignore[return-value]  # thinks it's a bool array...
