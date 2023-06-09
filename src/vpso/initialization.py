from typing import Union
import numpy as np
from scipy.stats.qmc import LatinHypercube

from vpso.typing import Array1d, Array3d


def initialize_particles(
    nvec: int,
    swarmsize: int,
    dim: int,
    lb: Array3d,
    ub: Array3d,
    max_velocity_rate: Union[float, Array1d],
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
    max_velocity_rate : float or 1d array
        The maximum velocity rate (proportional to the search domain size).
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

    if not isinstance(max_velocity_rate, np.ndarray):
        max_velocity_rate = np.full(nvec, max_velocity_rate)
    v_max = max_velocity_rate[:, np.newaxis, np.newaxis] * domain
    v = np_random.uniform(0, v_max, (nvec, swarmsize, dim))
    return x, v, v_max
