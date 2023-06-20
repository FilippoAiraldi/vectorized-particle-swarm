import unittest

import numpy as np
from pymoo.problems.functional import FunctionalProblem
from pymoo.util.misc import norm_eucl_dist

from vpso.adaptation import adapt, adaptation_strategy


def original_implementation1(f):
    def S1_explore(f):
        if f <= 0.4:
            return 0
        elif 0.4 < f <= 0.6:
            return 5 * f - 2
        elif 0.6 < f <= 0.7:
            return 1
        elif 0.7 < f <= 0.8:
            return -10 * f + 8
        elif 0.8 < f:
            return 0

    def S2_exploit(f):
        if f <= 0.2:
            return 0
        elif 0.2 < f <= 0.3:
            return 10 * f - 2
        elif 0.3 < f <= 0.4:
            return 1
        elif 0.4 < f <= 0.6:
            return -5 * f + 3
        elif 0.6 < f:
            return 0

    def S3_converge(f):
        if f <= 0.1:
            return 1
        elif 0.1 < f <= 0.3:
            return -5 * f + 1.5
        elif 0.3 < f:
            return 0

    def S4_jump_out(f):
        if f <= 0.7:
            return 0
        elif 0.7 < f <= 0.9:
            return 5 * f - 3.5
        elif 0.9 < f:
            return 1

    S = [S1_explore(f), S2_exploit(f), S3_converge(f), S4_jump_out(f)]
    strategy = np.argmax(S) + 1
    if strategy == 1:
        return 1, -1
    elif strategy == 2:
        return 0.5, -0.5
    elif strategy == 3:
        return 0.5, 0.5
    else:
        return -1, 1


def original_implementation2(problem, x, sx, w, c1, c2, delta):
    D = norm_eucl_dist(problem, x, x)
    mD = D.sum(axis=1) / (x.shape[0] - 1)
    _min, _max = mD.min(), mD.max()
    g_D = norm_eucl_dist(problem, sx, x).mean()
    f = (g_D - _min) / (_max - _min + 1e-32)
    strategy = original_implementation1(f)
    c1 = max(1.5, min(2.5, c1 + strategy[0] * delta))
    c2 = max(1.5, min(2.5, c2 + strategy[1] * delta))
    if c1 + c2 > 4.0:
        c1, c2 = 4.0 * (c1 / (c1 + c2)), 4.0 * (c2 / (c1 + c2))
    w = 1 / (1 + 1.5 * np.exp(-2.6 * f))
    return w, c1, c2


class TestAdaptation(unittest.TestCase):
    def test_adaptation_strategy(self):
        f = np.concatenate((np.arange(0, 1, 0.001), [0.23333, 0.5, 0.76666]))
        expected = [original_implementation1(f_) for f_ in f]
        actual = adaptation_strategy(f)
        np.testing.assert_array_equal(actual, expected)

    def test_adapt(self):
        nvec, dim = np.random.randint(3, 10, size=2)
        swarmsize = np.random.randint(100, 1000)
        ub = np.abs(np.random.rand(nvec, dim)) * 5 + 10
        lb = -np.abs(np.random.rand(nvec, dim)) * 5 - 10
        px = np.random.uniform(
            lb[:, np.newaxis], ub[:, np.newaxis], (nvec, swarmsize, dim)
        )
        sx = np.random.uniform(lb, ub, (nvec, dim))
        seed = np.random.randint(0, 1000)
        w = np.random.rand(nvec)
        c1 = np.random.rand(nvec) + 2
        c2 = np.random.rand(nvec) + 2

        np_random = np.random.default_rng(seed)
        deltas = 0.05 + np_random.random(size=nvec) * 0.05
        w_new, c1_new, c2_new = [], [], []
        for i in range(nvec):
            problem = FunctionalProblem(
                dim,
                lambda x: np.square(x).sum(),
                xl=lb[i],
                xu=ub[i],
            )
            o = original_implementation2(
                problem, px[i], sx[i], w[i], c1[i], c2[i], deltas[i]
            )
            w_new.append(o[0])
            c1_new.append(o[1])
            c2_new.append(o[2])

        np_random = np.random.default_rng(seed)
        w_new_, c1_new_, c2_new_ = adapt(
            px,
            sx[:, np.newaxis],
            nvec,
            swarmsize,
            lb[:, np.newaxis],
            ub[:, np.newaxis],
            w[:, np.newaxis, np.newaxis],
            c1[:, np.newaxis, np.newaxis],
            c2[:, np.newaxis, np.newaxis],
            np_random,
        )
        np.testing.assert_allclose(w_new_.squeeze(), w_new, err_msg="w")
        np.testing.assert_allclose(c1_new_.squeeze(), c1_new, err_msg="c1")
        np.testing.assert_allclose(c2_new_.squeeze(), c2_new, err_msg="c2")


if __name__ == "__main__":
    unittest.main()
