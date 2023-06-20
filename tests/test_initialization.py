import unittest

import numpy as np
from scipy.stats.qmc import LatinHypercube

from vpso.initialization import initialize_particles


class TestInitialization(unittest.TestCase):
    def test_initialize_particles(self):
        nvec, dim = np.random.randint(3, 10, size=2)
        swarmsize = np.random.randint(1000, 2000)
        ub = np.abs(np.random.randn(nvec, 1, dim))
        lb = -np.abs(np.random.randn(nvec, 1, dim))
        max_velocity_rate = np.random.uniform(0.1, 0.5)
        lhs_sampler = LatinHypercube(d=nvec * dim, scramble=False)
        np_random = np.random.default_rng()

        x, v, v_max = initialize_particles(
            nvec, swarmsize, dim, lb, ub, max_velocity_rate, lhs_sampler, np_random
        )
        self.assertTupleEqual(x.shape, (nvec, swarmsize, dim))
        self.assertTupleEqual(v.shape, (nvec, swarmsize, dim))
        self.assertTupleEqual(v_max.shape, (nvec, 1, dim))
        np.testing.assert_allclose(x.mean(1, keepdims=True), (ub + lb) / 2)
        self.assertTrue(((x >= lb) & (x <= ub)).all())
        self.assertTrue(((v >= 0) & (v <= v_max)).all())


if __name__ == "__main__":
    unittest.main()
