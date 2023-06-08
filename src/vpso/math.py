import numpy as np
from scipy.spatial.distance import (
    _copy_array_if_base_present,
    _distance_pybind,
    _distance_wrap,
)

from vpso.jit import jit
from vpso.typing import Array2d, Array3d


def batch_cdist(
    X: Array3d, Y: Array3d, dist_func=_distance_pybind.cdist_euclidean
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
    N = X.shape[0]
    out = np.empty((N, X.shape[1], Y.shape[1]), dtype=X.dtype)
    for i in range(N):
        dist_func(X[i], Y[i], out=out[i])
    return out


def batch_pdist(X: Array3d, dist_func=_distance_pybind.pdist_euclidean) -> Array2d:
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
    2d array
        Distance matrix of shape `(N, M * (M - 1) / 2)` between each pair of entries of
        the `(M, d)` matrices, where `d` is assumed to be the axis over which the
        distance is computed.
    """
    N, M = X.shape[:2]
    out = np.empty((N, (M - 1) * M // 2), dtype=X.dtype)
    for i in range(N):
        dist_func(X[i], out=out[i])
    return out


def batch_squareform(D: Array2d) -> Array2d:
    """Converts a batch of pairwise distance matrices to distance matrices.

    Parameters
    ----------
    D : 2d array
        Pairwise distance matrices of shape `(N, M * (M - 1) / 2)`, i.e., as returned by
        `batch_pdist` and not in square form.

    Returns
    -------
    3d array
        The same distance matrices but in square form, i.e., of shape `(N, M, M)`.
    """
    D = _copy_array_if_base_present(D)
    N, M = D.shape
    d = int(0.5 * (np.sqrt(8 * M + 1) + 1))
    out = np.zeros((N, d, d), dtype=D.dtype)
    for i in range(N):
        _distance_wrap.to_squareform_from_vector_wrap(out[i], D[i])
    return out


@jit
def pso_equation(
    x: Array3d,
    px: Array3d,
    sx: Array2d,
    v: Array3d,
    v_max: Array3d,
    w: float,
    c1: float,
    c2: float,
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
    sx : 2d array
        Social best, i.e., the best particle so far. An array of shape `(N, d)`.
    v : 3d array
        Current velocities of the particles. An array of shape `(N, M, d)`.
    v_max : 3d array
        Maximum velocities of the particles. An array of shape `(N, 1, d)`.
    w : float
        Inertia weight.
    c1 : float
        Cognitive weight.
    c2 : float
        Social weight.
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
    social = c2 * r2 * (sx[:, np.newaxis] - x)

    v_new = np.clip(inerta + cognitive + social, -v_max, v_max)
    x_new = x + v_new
    return x_new, v_new
