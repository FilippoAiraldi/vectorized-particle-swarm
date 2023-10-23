"""
This simple example demonstrates how the constrained version of VPSO can be used to
tackle two academic examples from [1], at the same time thanks to its vectorization.
The functions are taken from test problem 4 and 5 of the reference.

References
----------
[2] Parsopoulos, K.E. and Vrahatis, M.N., 2002. Particle swarm optimization method for
    constrained optimization problems. Intelligent technologies–theory and application:
    New trends in intelligent technologies, 76(1), pp.214-220.
"""


import numpy as np

from vpso import vpso_constrained


def func(x):
    x_sq = np.square(x)
    return (
        5.3578547 * x_sq[..., 2]
        + 0.8356891 * x[..., 0] * x[..., 4]
        + 37.293239 * x[..., 0]
        - 40792.141
    )


T1_func = lambda x: np.asarray((x[0, :, 1] * x[0, :, 4], x[0, :, 1] * x[0, :, 2]))
T2 = np.reshape([0.0006262, 0.00026], (2, 1))


def constraints(x):
    T1 = T1_func(x)
    g1 = (
        85.334407
        + 0.0056858 * T1
        + T2 * x[..., 0] * x[..., 3]
        - 0.0022053 * x[..., 2] * x[..., 4]
    )
    g2 = (
        80.51249
        + 0.0071317 * x[..., 1] * x[..., 4]
        + 0.0029955 * x[..., 0] * x[..., 1]
        + 0.0021813 * x[..., 2] * x[..., 2]
    )
    g3 = (
        9.300961
        + 0.0047026 * x[..., 2] * x[..., 4]
        + 0.0012547 * x[..., 0] * x[..., 2]
        + 0.0019085 * x[..., 2] * x[..., 3]
    )
    return np.stack((-g1, g1 - 92, 90 - g2, g2 - 110, 20 - g3, g3 - 25), axis=-1)


# create the bounds for the variables of each function
bounds = np.asarray(
    [
        [78, 102],
        [33, 45],
        [27, 45],
        [27, 45],
        [27, 45],
    ]
)
n_problems = 2
lb = np.tile(bounds[:, 0], (n_problems, 1))
ub = np.tile(bounds[:, 1], (n_problems, 1))

# run the optimization
x_opt, f_opt, msg = vpso_constrained(
    func=func,
    constraints=constraints,
    lb=lb,
    ub=ub,
    swarmsize=1000,
    maxiter=1000,
    ftol=-1,  # disable ftol termination
    xtol=-1,  # disable xtol termination
    seed=1909,
    rapid_penalty=False,  # this function prefers a slower penalty increase
)

print("termination reason:", msg)

f_opt_no_penalty = func(x_opt)
g_opt = constraints(x_opt[:, None, :])
for i in range(n_problems):
    print(f"Test Function {4 + i}:")
    print(f"    x_opt = ({x_opt[i][0]:.6f}, {x_opt[i][1]:.6f})")
    print(f"    f_opt = {f_opt[i]:.6f}")
    print(f"    f_opt_no_penalty = {f_opt_no_penalty[i]:.6f}")
    print(f"    g_opt = {g_opt[i].flatten()}")
