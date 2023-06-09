from itertools import product
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import unittest

import numpy as np

from vpso.mutation import polynomial_mutation


class TestMutation(unittest.TestCase):
    def test_polynomial_mutation__if_mutation_prob_is_zero__returns_immediately(self):
        np_random = np.random.Generator(np.random.PCG64())
        nvec, dim = np_random.integers(3, 10, size=2)
        swarmsize = np_random.integers(1000, 2000)
        ub = np.abs(np_random.normal(size=(nvec, 1, dim))) + 10
        lb = -np.abs(np_random.normal(size=(nvec, 1, dim))) - 10
        x_mutated = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        x_original = x_mutated.copy()

        polynomial_mutation(
            x_mutated, None, None, None, None, nvec, dim, 0.0, np_random
        )
        np.testing.assert_array_equal(x_original, x_mutated)

    def test_polynomial_mutation(self):
        np_random = np.random.Generator(np.random.PCG64(17))
        nvec, dim = np_random.integers(3, 10, size=2)
        swarmsize = np_random.integers(1000, 2000)
        ub = np.abs(np_random.normal(size=(nvec, 1, dim))) + 10
        lb = -np.abs(np_random.normal(size=(nvec, 1, dim))) - 10
        x_mutated = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        x_original = x_mutated.copy()
        px = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        pf = np_random.uniform(size=(nvec, swarmsize))
        mutation_prob = 1.0

        polynomial_mutation(
            x_mutated, px, pf, lb, ub, nvec, dim, mutation_prob, np_random
        )
        self.assertFalse(np.allclose(x_original, x_mutated))

        bests = set(pf.argmin(1))
        for i, j in product(range(nvec), range(swarmsize)):
            if j not in bests:
                np.testing.assert_array_equal(x_original[i, j], x_mutated[i, j])


if __name__ == "__main__":
    unittest.main()
