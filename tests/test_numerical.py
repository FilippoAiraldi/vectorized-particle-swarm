# import unittest

# import numpy as np
# from pymoo.problems.single import Ackley

# from vpso import vpso


# class TestNumerical(unittest.TestCase):
#     def test_against_pymoo(self):
#         np.random.seed(17)

#         pop_size = np.random.randint(10, 50)
#         np.random.uniform(1e-8, 1e-2)
#         np.random.randint(100, 1000)
#         np.random.randint(1, 30)
#         problem = Ackley()
#         # res = minimize(
#         #     problem,
#         #     PSO(pop_size=pop_size),
#         #     termination=DefaultSingleObjectiveTermination(
#         #         ftol=ftol, n_max_gen=n_max_gen, period=period
#         #     ),
#         #     verbose=False,
#         #     seed=49,
#         # )

#         nvec = 3
#         res_ = vpso(
#             lambda x: np.asarray([problem.evaluate(x_) for x_ in x]),
#             np.tile(problem.xl, (nvec, 1)),
#             np.tile(problem.xu, (nvec, 1)),
#             swarmsize=pop_size,
#             seed=49,
#         )


# if __name__ == "__main__":
#     unittest.main()
