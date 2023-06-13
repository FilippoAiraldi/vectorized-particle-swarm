# cython: infer_types=True
cimport cython
import numpy as np
from scipy.spatial.distance import _distance_pybind


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def batch_cdist2(double[:, :, :] X, double[:, :, :] Y):
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
    out = np.empty((N, X.shape[1], Y.shape[1]), dtype=np.double)
    for i in range(N):
        out[i] = _distance_pybind.cdist_euclidean(X[i], Y[i])
    return out
