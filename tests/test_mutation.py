import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import unittest
from itertools import product

import numpy as np

from vpso.mutation import mutate, polynomial_mutation


class TestMutation(unittest.TestCase):
    def test_mutate__if_mutation_prob_is_zero__returns_immediately(self):
        np_random = np.random.Generator(np.random.PCG64())
        nvec, dim = np_random.integers(3, 10, size=2)
        swarmsize = np_random.integers(1000, 2000)
        ub = np.abs(np_random.normal(size=(nvec, 1, dim))) + 10
        lb = -np.abs(np_random.normal(size=(nvec, 1, dim))) - 10
        x_mutated = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        x_original = x_mutated.copy()
        px = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        pf = np_random.uniform(size=(nvec, swarmsize))
        mutation_prob = 0.0

        mutate(x_mutated, px, pf, lb, ub, nvec, dim, mutation_prob, np_random)
        np.testing.assert_array_equal(x_original, x_mutated)

    def test_mutate__performs_mutation(self):
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

        mutate(x_mutated, px, pf, lb, ub, nvec, dim, mutation_prob, np_random)
        self.assertFalse(np.allclose(x_original, x_mutated))

        bests = set(pf.argmin(1))
        for i, j in product(range(nvec), range(swarmsize)):
            if j not in bests:
                np.testing.assert_array_equal(x_original[i, j], x_mutated[i, j])

    def test_polynomial_mutation(self):
        seed = np.random.randint(0, 1000)
        np_random = np.random.Generator(np.random.PCG64(seed))
        nvec, dim = np_random.integers(3, 10, size=2)
        ub = np.abs(np_random.normal(size=(nvec, dim))) + 10
        lb = -np.abs(np_random.normal(size=(nvec, dim))) - 10
        x = np_random.uniform(lb, ub, (nvec, dim))

        def original_implementation(X, eta, lb, ub, rand):
            delta1 = (X - lb) / (ub - lb)
            delta2 = (ub - X) / (ub - lb)
            mut_pow = 1.0 / (eta + 1.0)
            mask = rand <= 0.5
            mask_not = np.logical_not(mask)
            deltaq = np.zeros(X.shape)
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (eta + 1.0)))
            d = np.power(val, mut_pow) - 1.0
            deltaq[mask] = d[mask]
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (eta + 1.0)))
            d = 1.0 - (np.power(val, mut_pow))
            deltaq[mask_not] = d[mask_not]
            return (X + deltaq * (ub - lb)).clip(lb, ub)

        np_random = np.random.Generator(np.random.PCG64(seed))
        eta = np_random.uniform(5, 30, size=nvec)
        rand = np_random.uniform(size=(nvec, dim))
        expected = [
            original_implementation(x[i], eta[i], lb[i], ub[i], rand[i])
            for i in range(nvec)
        ]

        np_random = np.random.Generator(np.random.PCG64(seed))
        actual = polynomial_mutation(
            x, np.full((nvec, dim), True), lb, ub, nvec, dim, np_random
        )
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
