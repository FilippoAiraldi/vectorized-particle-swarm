from typing import Optional

import numpy as np
from scipy.stats.qmc import LatinHypercube

from vpso.typing import Array3d


def initialize_particles(
    nvec: int,
    swarmsize: int,
    dim: int,
    lb: Array3d,
    ub: Array3d,
    max_velocity_rate: float,
    seed: Optional[int],
) -> tuple[Array3d, Array3d, Array3d, np.random.Generator]:
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
    max_velocity_rate : float
        The maximum velocity rate (proportional to the search domain size).
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple of 3d arrays and np.random.Generator

    """
    lhs_sampler = LatinHypercube(d=nvec * dim, seed=seed)
    np_random = np.random.Generator(np.random.PCG64(seed))
    domain = ub - lb

    x = lb + domain * lhs_sampler.random(swarmsize).reshape(
        swarmsize, nvec, dim
    ).transpose((1, 0, 2))

    v_max = max_velocity_rate * domain
    v = np_random.uniform(0, v_max, (nvec, swarmsize, dim))
    return x, v, v_max, np_random
