import unittest

import numpy as np

from vpso import vpso_constrained


class TestNumericalConstrained(unittest.TestCase):
    def test_vpo_constrained(self):
        def func(x):
            return np.square(x[..., 0] - 2) + np.square(x[..., 1] - 1)

        def constraints(x):
            x_sq = np.square(x)
            g1a = x[..., 0] - 2 * x[..., 1] + 1
            g1b = -g1a
            g2 = x_sq[..., 0] / 4 + x_sq[..., 1] - 1
            return np.stack((g1a, g1b, g2), axis=-1)

        bounds = np.full((1, 2), 100.0)
        x_opt, _, _ = vpso_constrained(
            func=func,
            constraints=constraints,
            lb=-bounds,
            ub=+bounds,
            swarmsize=100,
            maxiter=1000,
            ftol=-1,  # disable ftol termination
            xtol=-1,  # disable xtol termination
            seed=1909,
            rapid_penalty=False,  # this function prefers a slower penalty increase
        )

        np.testing.assert_allclose(func(x_opt).item(), 1.393465)


if __name__ == "__main__":
    unittest.main()
