import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
import unittest

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from vpso.math import batch_cdist, batch_pdist, batch_squareform, pso_equation


class TestMath(unittest.TestCase):
    def test_batch_cdist(self):
        N, M1, M2, d = np.random.randint(10, 50, size=4)
        X = np.random.randn(N, M1, d)
        Y = np.random.randn(N, M2, d)
        actual = batch_cdist(X, Y)
        expected = np.asarray([cdist(x, y) for x, y in zip(X, Y)])
        np.testing.assert_allclose(actual, expected)

    def test_batch_pdist(self):
        N, M, d = np.random.randint(10, 50, size=3)
        X = np.random.randn(N, M, d)
        actual = batch_pdist(X)
        expected = np.asarray([pdist(x) for x in X])
        np.testing.assert_allclose(actual, expected)

    def test_batch_squareform(self):
        N, d = np.random.randint(10, 50, size=2)
        X = np.random.randn(N, d * (d - 1) // 2)
        actual = batch_squareform(X)
        expected = np.asarray([squareform(x) for x in X])
        np.testing.assert_allclose(actual, expected)

    def test_pso_equation(self):
        nvec, dim = np.random.randint(3, 10, size=2)
        swarmsize = np.random.randint(1000, 2000)
        x = np.random.randn(nvec, swarmsize, dim)
        px = np.random.randn(nvec, swarmsize, dim)
        sx = np.random.randn(nvec, 1, dim)
        v = np.random.randn(nvec, swarmsize, dim)
        v_max = np.random.randn(nvec, 1, dim)
        w = np.random.rand()
        c1 = np.random.rand()
        c2 = np.random.rand()
        r1 = np.random.rand(nvec, swarmsize, dim)
        r2 = np.random.rand(nvec, swarmsize, dim)

        def original_pso_eq(X, P_X, S_X, V, V_max, w, c1, c2, r1, r2):
            inerta = w * V
            cognitive = c1 * r1 * (P_X - X)
            social = c2 * r2 * (S_X - X)
            Vp = inerta + cognitive + social
            Vp = np.clip(Vp, -V_max, V_max)
            Xp = X + Vp
            return Xp, Vp

        x_new_, v_new_ = [], []
        for i in range(nvec):
            o = original_pso_eq(
                x[i], px[i], sx[i], v[i], v_max[i], w, c1, c2, r1[i], r2[i]
            )
            x_new_.append(o[0])
            v_new_.append(o[1])
        x_new_, v_new_ = np.asarray(x_new_), np.asarray(v_new_)

        x_new, v_new = pso_equation(x, px, sx, v, v_max, w, c1, c2, None, r1, r2)

        np.testing.assert_allclose(x_new, x_new_)
        np.testing.assert_allclose(v_new, v_new_)


if __name__ == "__main__":
    unittest.main()
