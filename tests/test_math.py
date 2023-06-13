import unittest

import numpy as np
from parameterized import parameterized
from scipy.spatial.distance import cdist, pdist, squareform

from vpso.math import batch_cdist, batch_pdist, pso_equation


class TestMath(unittest.TestCase):
    @parameterized.expand((("euclidean",), ("sqeuclidean",)))
    def test_batch_cdist(self, dist_type: str):
        N, M1, M2, d = np.random.randint(10, 50, size=4)
        X = np.random.randn(N, M1, d)
        Y = np.random.randn(N, M2, d)
        actual = batch_cdist(X, Y, dist_type)
        expected = np.asarray([cdist(x, y, dist_type) for x, y in zip(X, Y)])
        np.testing.assert_allclose(actual, expected)

    @parameterized.expand((("euclidean",), ("sqeuclidean",)))
    def test_batch_pdist(self, dist_type: str):
        np.random.seed(0)
        N, M, d = np.random.randint(10, 50, size=3)
        X = np.random.randn(N, M, d)
        actual = batch_pdist(X, dist_type)
        expected = np.asarray([squareform(pdist(x, dist_type)) for x in X])
        np.testing.assert_allclose(actual, expected)

    def test_pso_equation(self):
        seed = np.random.randint(0, 1000)
        nvec, dim = np.random.randint(3, 10, size=2)
        swarmsize = np.random.randint(1000, 2000)
        np_random = np.random.Generator(np.random.PCG64(seed))
        r1 = np_random.uniform(size=(nvec, swarmsize, dim))
        r2 = np_random.uniform(size=(nvec, swarmsize, dim))
        x = np_random.normal(size=(nvec, swarmsize, dim))
        px = np_random.normal(size=(nvec, swarmsize, dim))
        sx = np_random.normal(size=(nvec, 1, dim))
        v = np_random.normal(size=(nvec, swarmsize, dim))
        v_max = np_random.normal(size=(nvec, 1, dim))
        w = np_random.uniform(size=(nvec, 1, 1))
        c1 = np_random.uniform(size=(nvec, 1, 1))
        c2 = np_random.uniform(size=(nvec, 1, 1))

        def original_implementation(X, P_X, S_X, V, V_max, w, c1, c2, r1, r2):
            inerta = w * V
            cognitive = c1 * r1 * (P_X - X)
            social = c2 * r2 * (S_X - X)
            Vp = inerta + cognitive + social
            Vp = Vp.clip(-V_max, V_max)
            Xp = X + Vp
            return Xp, Vp

        x_new_, v_new_ = [], []
        for i in range(nvec):
            o = original_implementation(
                x[i], px[i], sx[i], v[i], v_max[i], w[i], c1[i], c2[i], r1[i], r2[i]
            )
            x_new_.append(o[0])
            v_new_.append(o[1])
        x_new_, v_new_ = np.asarray(x_new_), np.asarray(v_new_)

        np_random = np.random.Generator(np.random.PCG64(seed))
        x_new, v_new = pso_equation(x, px, sx, v, v_max, w, c1, c2, np_random)

        np.testing.assert_allclose(x_new, x_new_)
        np.testing.assert_allclose(v_new, v_new_)


if __name__ == "__main__":
    unittest.main()
