from typing import Callable

import numba as nb

_float = nb.float64
_int = nb.int32
_array2d = _float[:, :]
_array3d = _float[:, :, :]

_signatures = {
    "_pso_equation": nb.types.UniTuple(_array3d, 2)(
        _array3d,  # x
        _array3d,  # px
        _array2d,  # sx
        _array3d,  # v
        _array3d,  # v_max
        _float,  # w
        _float,  # c1
        _float,  # c2
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "_repair_out_of_bounds": nb.types.UniTuple(_array3d, 2)(
        _array3d,  # x
        _array3d,  # x_new
        _array3d,  # v_new
        _array3d,  # px
        _array2d,  # sx
        _array3d,  # v
        _array3d,  # v_max
        _array3d,  # lb
        _array3d,  # ub
        _float,  # w
        _float,  # c1
        _float,  # c2
        _int,  # iters
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "_polynomial_mutation": _array3d(
        _array3d,  # x
        _array3d,  # px
        _array2d,  # pf
        _array3d,  # lb
        _array3d,  # ub
        _int,  # nvec
        _int,  # dim
        _float,  # mutation_prob
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "_generate_offsprings": nb.types.UniTuple(_array3d, 2)(
        _array3d,  # x
        _array3d,  # px
        _array2d,  # pf
        _array2d,  # sx
        _array3d,  # v
        _array3d,  # v_max
        _array3d,  # lb
        _array3d,  # ub
        _int,  # nvec
        _int,  # dim
        _float,  # w
        _float,  # c1
        _float,  # c2
        _int,  # iters
        nb.bool_,  # perturb_best
        _float,  # mutation_prob
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
}


def jit(func: Callable) -> Callable:
    """Assigns a jit decorator to the given function with the correct signature."""
    return nb.njit(_signatures[func.__name__])(func)
