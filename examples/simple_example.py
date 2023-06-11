"""
This simple example demonstrates how two functions can be optimized at once using
`vpso`. The functions are the Ackley and Himmelblau functions [1], both of which are
defined in a 2-dimensional space. The global minima are
    - Ackley: global minimum of 0 at (0, 0)
    - Himmelblau: global minima of 0 at (3, 2),
      (-2.805118, 3.131312), (-3.779310, -3.283186), and (3.584428, -1.848126)

References
----------
[1] Surjanovic, S. & Bingham, D. (2013). Virtual Library of Simulation Experiments: Test
    Functions and Datasets. Retrieved June 11, 2023, from
    http://www.sfu.ca.tudelft.idm.oclc.org/~ssurjano.
"""


import logging

import numpy as np

from vpso import vpso

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s (%(levelname)s): %(message)s.", "%Y-%m-%d,%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def ackley(x):
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(x * x, axis=1)))
    part2 = -np.exp(0.5 * np.sum(np.cos(2 * np.pi * x), axis=1))
    return part1 + part2 + 20 + np.exp(1)


def himmelblau(x):
    return (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2


# create the bounds for the variables of each function
n_vars = 2
bounds = np.tile([32.768, 6], (n_vars, 1))


# run the optimization
x_opt, f_opt, _ = vpso(
    func=lambda x: [ackley(x[0]), himmelblau(x[1])],
    lb=-bounds,
    ub=+bounds,
    verbosity=logging.DEBUG,
)


for i, fun in enumerate([ackley, himmelblau]):
    print(f"Function {i + 1} ({fun.__name__}):")
    print(f"    x_opt = ({x_opt[i][0]:.6f}, {x_opt[i][1]:.6f})")
    print(f"    f_opt = {f_opt[i]:.6f}")
