import unittest

import numpy as np
from parameterized import parameterized

from vpso.reparation import repair_out_of_bounds


class TestReparation(unittest.TestCase):
    def test_repair_out_of_bounds__if_in_bounds__returns_immediately(self):
        np_random = np.random.Generator(np.random.PCG64())
        nvec, dim = np_random.integers(3, 10, size=2)
        swarmsize = np_random.integers(1000, 2000)
        ub = np.abs(np_random.normal(size=(nvec, 1, dim))) + 10
        lb = -np.abs(np_random.normal(size=(nvec, 1, dim))) - 10
        x = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        x_new = np_random.uniform(lb + 0.1, ub - 0.1, (nvec, swarmsize, dim))
        v_new = np_random.uniform(size=(nvec, swarmsize, dim))
        px = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        sx = np_random.uniform(lb, ub, (nvec, 1, dim))
        v = np_random.uniform(size=(nvec, swarmsize, dim))
        v_max = np_random.uniform(size=(nvec, 1, dim))
        w = np_random.uniform(size=(nvec, 1, 1))
        c1 = np_random.uniform(size=(nvec, 1, 1))
        c2 = np_random.uniform(size=(nvec, 1, 1))

        x_new_, v_new_ = repair_out_of_bounds(
            x,
            x_new,
            v_new,
            px,
            sx,
            v,
            v_max,
            lb,
            ub,
            w,
            c1,
            c2,
            200,
            np_random,
        )

        np.testing.assert_array_equal(x_new_, x_new)
        np.testing.assert_array_equal(v_new_, v_new)

    @parameterized.expand([(0,), (200,)])
    def test_repair_out_of_bounds__with_resamplimg_repair(self, iters: int):
        np_random = np.random.Generator(np.random.PCG64())
        nvec, dim = np_random.integers(3, 10, size=2)
        swarmsize = np_random.integers(1000, 2000)
        ub = np.abs(np_random.normal(size=(nvec, 1, dim))) + 10
        lb = -np.abs(np_random.normal(size=(nvec, 1, dim))) - 10
        x = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        x_new = np_random.uniform(lb - 10, ub + 10, (nvec, swarmsize, dim))
        v_new = np_random.uniform(size=(nvec, swarmsize, dim))
        px = np_random.uniform(lb, ub, (nvec, swarmsize, dim))
        sx = np_random.uniform(lb, ub, (nvec, 1, dim))
        v = np_random.uniform(size=(nvec, swarmsize, dim))
        v_max = np_random.uniform(size=(nvec, 1, dim))
        w = np_random.uniform(size=(nvec, 1, 1))
        c1 = np_random.uniform(size=(nvec, 1, 1))
        c2 = np_random.uniform(size=(nvec, 1, 1))

        x_new_, v_new_ = repair_out_of_bounds(
            x,
            x_new,
            v_new,
            px,
            sx,
            v,
            v_max,
            lb,
            ub,
            w,
            c1,
            c2,
            iters,
            np_random,
        )

        self.assertTrue(((x_new_ >= lb) & (x_new_ <= ub)).all())
        if iters > 0:
            mask = (x_new < lb) | (x_new > ub)  # only these velocities are modified
            self.assertTrue(((v_new_ >= -v_max) & (v_new_ <= v_max))[mask].all())


if __name__ == "__main__":
    unittest.main()
