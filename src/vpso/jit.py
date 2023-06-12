from typing import Callable

import numba as nb

_float = nb.float64
_int = nb.int32

_signatures = {
    "pso_equation": nb.types.UniTuple(_float[:, :, :], 2)(
        _float[:, :, :],  # x
        _float[:, :, :],  # px
        _float[:, :, :],  # sx
        _float[:, :, :],  # v
        _float[:, :, :],  # v_max
        _float[:, :, :],  # w
        _float[:, :, :],  # c1
        _float[:, :, :],  # c2
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "repair_out_of_bounds": nb.types.UniTuple(_float[:, :, :], 2)(
        _float[:, :, :],  # x
        _float[:, :, :],  # x_new
        _float[:, :, :],  # v_new
        _float[:, :, :],  # px
        _float[:, :, :],  # sx
        _float[:, :, :],  # v
        _float[:, :, :],  # v_max
        _float[:, :, :],  # lb
        _float[:, :, :],  # ub
        _float[:, :, :],  # w
        _float[:, :, :],  # c1
        _float[:, :, :],  # c2
        _int,  # iters
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "polynomial_mutation": _float[:, :](
        _float[:, :],  # x_best
        nb.bool_[:, :],  # mutation_mask
        _float[:, :],  # lb
        _float[:, :],  # ub
        _int,  # nvec
        _int,  # dim
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "mutate": nb.types.void(
        _float[:, :, :],  # x
        _float[:, :, :],  # px
        _float[:, :],  # pf
        _float[:, :, :],  # lb
        _float[:, :, :],  # ub
        _int,  # nvec
        _int,  # dim
        _float,  # mutation_prob
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "advance_population": nb.types.Tuple((_float[:, :, :], _float[:, :]))(
        _float[:, :, :],  # x
        _float[:, :],  # f
        _float[:, :, :],  # px
        _float[:, :],  # pf
    ),
    "generate_offsprings": nb.types.UniTuple(_float[:, :, :], 2)(
        _float[:, :, :],  # x
        _float[:, :, :],  # px
        _float[:, :],  # pf
        _float[:, :, :],  # sx
        _float[:, :, :],  # v
        _float[:, :, :],  # v_max
        _float[:, :, :],  # lb
        _float[:, :, :],  # ub
        _int,  # nvec
        _int,  # dim
        _float[:, :, :],  # w
        _float[:, :, :],  # c1
        _float[:, :, :],  # c2
        _int,  # iters
        nb.bool_,  # perturb_best
        _float,  # mutation_prob
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "adaptation_strategy": _float[:, :](_float[:]),
    "perform_adaptation": nb.types.UniTuple(_float[:, :, :], 3)(
        _float[:, :, :],  # w
        _float[:, :, :],  # c1
        _float[:, :, :],  # c2
        _float[:],  # stage
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
}


def jit(func: Callable, cache: bool = True) -> Callable:
    """Assigns a jit decorator to the given function with the correct signature."""
    return nb.njit(_signatures[func.__name__], cache=cache)(func)
