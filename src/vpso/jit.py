from typing import Callable

import numba as nb

_float = nb.float64
_int = nb.int32

_signatures = {
    "cdist_func": nb.float64[:, :](
        nb.float64[:, :], nb.float64[:, :], nb.types.unicode_type
    ),
    "pdist_func": nb.float64[:, :](nb.float64[:, :], nb.types.unicode_type),
    "batch_cdist": nb.float64[:, :, :](
        nb.float64[:, :, :], nb.float64[:, :, :], nb.types.unicode_type
    ),
    "batch_pdist": nb.float64[:, :, :](nb.float64[:, :, :], nb.types.unicode_type),
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
        _float[:, :, :],  # px
        _float[:, :, :],  # sx
        _int,  # nvec
        _int,  # swarmize
        _float[:, :, :],  # lb
        _float[:, :, :],  # ub
        _float[:, :, :],  # w
        _float[:, :, :],  # c1
        _float[:, :, :],  # c2
        nb.types.NumPyRandomGeneratorType("NumPyRandomGeneratorType"),
    ),
    "update_patience": nb.types.UniTuple(_float[:], 2)(
        _float[:, :, :],  # sx
        _float[:],  # sf
        _float[:, :, :],  # sx_new
        _float[:],  # sf_new
        _float[:, :, :],  # lb
        _float[:, :, :],  # ub
        _float[:],  # xtol
        _float[:],  # ftol
        _int[:, :],  # current_patience_level
    ),
}


def jit(signature=None, cache: bool = True, parallel: bool = False) -> Callable:
    """Assigns a jit decorator to the given function with the correct signature."""
    _signature = signature

    def _decorator(func: Callable) -> Callable:
        signature = _signatures[func.__name__] if _signature is None else _signature
        return nb.njit(signature, cache=cache, parallel=parallel)(func)

    return _decorator
