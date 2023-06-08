from typing import Callable

import numba as nb

_float = nb.float64
_int = nb.int32

_signatures = {
    "_pso_equation": nb.types.UniTuple(_float[:, :, :], 2)(
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :],
        _float[:, :, :],
        _float[:, :, :],
        _float,
        _float,
        _float,
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "_repair_out_of_bounds": nb.types.UniTuple(_float[:, :, :], 2)(
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :, :],
        _float,
        _float,
        _float,
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
        _int,
    ),
    "_polynomial_mutation": _float[:, :, :](
        _float[:, :, :],
        _float[:, :, :],
        _float[:, :],
        _float[:, :, :],
        _float[:, :, :],
        _int,
        _int,
        _float,
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
}


def jit(func: Callable) -> Callable:
    """Assigns a jit decorator to the given function with the correct signature."""
    return nb.njit(_signatures[func.__name__])(func)
