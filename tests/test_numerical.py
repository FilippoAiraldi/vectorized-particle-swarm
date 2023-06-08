import unittest
from pymoo.problems.single import Ackley
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
import numpy as np
from vpso import vpso


class TestNumerical(unittest.TestCase):
    def test_against_pymoo(self):
        np.random.seed(17)

        pop_size = np.random.randint(10, 50)
        ftol = np.random.uniform(1e-8, 1e-2)
        n_max_gen = np.random.randint(100, 1000)
        period = np.random.randint(1, 30)
        problem = Ackley()
        res = minimize(
            problem,
            PSO(pop_size=pop_size),
            termination=DefaultSingleObjectiveTermination(
                ftol=ftol, n_max_gen=n_max_gen, period=period
            ),
            verbose=True,
            seed=49,
        )

        res_ = vpso(
            lambda x: np.asarray([problem.evaluate(x[0])] * 2),
            problem.xl[np.newaxis],
            problem.xu[np.newaxis],
            swarmsize=pop_size,
            seed=49,
        )


if __name__ == "__main__":
    unittest.main()
