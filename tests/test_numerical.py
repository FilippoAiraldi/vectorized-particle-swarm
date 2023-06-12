import unittest
from typing import Type

import numpy as np
from parameterized import parameterized
from pymoo.core.problem import Problem
from pymoo.problems.single import (
    Ackley,
    Griewank,
    Himmelblau,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Zakharov,
)

from vpso import vpso

CLS = [Ackley, Griewank, Himmelblau, Rastrigin, Rosenbrock, Schwefel, Sphere, Zakharov]
np.set_printoptions(precision=3, suppress=True)


class TestNumerical(unittest.TestCase):
    @parameterized.expand([(cls,) for cls in CLS])
    def test_vpo(self, cls: Type[Problem]):
        np.random.seed(1909)
        nvec = np.random.randint(2, 10)
        swarmsize = 300
        # ftol = 1e-8
        maxiter = 500
        # period = 10
        problem = cls()

        def objs(x: np.ndarray) -> np.ndarray:
            return np.asarray(
                [problem.evaluate(x_, return_values_of=["F"]) for x_ in x]
            )

        actual_x, actual_f, _ = vpso(
            objs,
            np.tile(problem.xl, (nvec, 1)),
            np.tile(problem.xu, (nvec, 1)),
            #
            swarmsize=swarmsize,
            #
            maxiter=maxiter,
            #
            xtol=-1,
            ftol=-1,
            seed=np.random.randint(0, 1000),
        )

        pf = problem.pareto_front().item()
        ps = problem.pareto_set().squeeze()
        np.testing.assert_allclose(
            actual_f, pf, atol=1e-3, rtol=1e-3, err_msg=f"f {cls.__name__}"
        )
        if ps.ndim == 1:
            np.testing.assert_allclose(
                *np.broadcast_arrays(actual_x, ps),
                atol=1e-3,
                rtol=1e-3,
                err_msg=f"x {cls.__name__}",
            )
        else:
            for i in range(nvec):
                x_opt = actual_x[i]
                self.assertTrue(
                    any(np.allclose(x_opt, ps_, atol=1e-3, rtol=1e-3) for ps_ in ps),
                    msg=f"x {i} {cls.__name__}",
                )


if __name__ == "__main__":
    unittest.main()
