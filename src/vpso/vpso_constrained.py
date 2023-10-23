from typing import Any, Callable

import numba as nb
import numpy as np
from numpy.typing import ArrayLike

from vpso import vpso
from vpso.typing import Array1d, Array2d, Array3d


@nb.njit(
    nb.float64[:, :](nb.int32, nb.float64[:, :, :], nb.boolean), cache=True, nogil=True
)
def penalty(k: int, g: Array3d, rapid: bool) -> Array2d:
    """Takes the current iteration and the constraints values at the swarm positions as
    input (N, M, d), and returns the penalty factor for each particle in each swarm,
    i.e., (N, M)."""
    q = np.maximum(0, g)
    theta = np.full(q.shape, 300.0)
    theta = np.where(q <= 1.0, 100.0, theta)
    theta = np.where(q <= 0.1, 20.0, theta)
    theta = np.where(q <= 0.001, 10.0, theta)
    gamma = np.where(q < 1, 1, 2)
    multiplier = np.sqrt(k)
    if rapid:
        multiplier *= k
    return multiplier * (theta * np.power(q, gamma)).sum(2)


class FuncWithPenalty:
    def __init__(
        self,
        f: Callable[[Array3d], ArrayLike],
        g: Callable[[Array3d], ArrayLike],
        rapid_penalty: bool = True,
    ) -> None:
        self.iteration = 0
        self.f = f
        self.g = g
        self.rapid = rapid_penalty

    def __call__(self, x: Array3d) -> Array2d:
        self.iteration += 1
        f = self.f(x)
        g = self.g(x)
        return f + penalty(self.iteration, g, self.rapid)


def vpso_constrained(
    func: Callable[[Array3d], ArrayLike],
    constraints: Callable[[Array3d], ArrayLike],
    *args: Any,
    rapid_penalty: bool = True,
    **kwargs: Any,
) -> tuple[Array2d, Array1d, str]:
    """Constrained variant of Vectorized Particle Swarm Optimization (VPSO), which can
    handle inequality constraints as in [1].

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Same as in `vpso`.
    constraints : callable
        The inequality constraints to be satisfied, in the form `g(x) <= 0`. Must take a
        3d array of shape `(N, M, d)` as input, and return an output that can be
        converted to an array of shape `(N, M, g)`, where `N` is the number of
        vectorized problems to solve, `M` is the `swarmsize`, `d` is the dimension of
        the search space, and `g` is the  number of inequality constraints.
    rapid_penalty : bool, optional
        If `True`, the penalty factor is multiplied by the iteration number, increasing
        the speed at which the constraint penalty increases. Default is `True`.
    *args, **kwargs
        All other arguments are passed to `vpso`.

    Returns
    -------
    tuple of (2d array, 1d array, str)
        Returns a tuple containing
         - the best minimizer of each problem
         - the best minimum of each problem (including constraint penalty)
         - the termination reason as a string

    References
    ----------
    [1] Parsopoulos, K.E. and Vrahatis, M.N., 2002. Particle swarm optimization method
        for constrained optimization problems. Intelligent technologies–theory and
        application: New trends in intelligent technologies, 76(1), pp.214-220.
    """
    return vpso(FuncWithPenalty(func, constraints, rapid_penalty), *args, **kwargs)
