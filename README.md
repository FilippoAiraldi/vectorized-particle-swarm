# Vectorized Particle Swarm Optimization

**vpso** is a Python package for adaptive Particle Swarm Optimization (PSO) [[1]](#1) that allows to run multiple similar optimization problems in a vectorized fashion.

[![PyPI version](https://badge.fury.io/py/vpso.svg)](https://badge.fury.io/py/vpso)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/multi-pso/blob/master/LICENSE)
![Python 3.9](https://img.shields.io/badge/python->=3.9-green.svg)

[![Tests](https://github.com/FilippoAiraldi/multi-pso/actions/workflows/ci.yml/badge.svg)](https://github.com/FilippoAiraldi/multi-pso/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/vpso)](https://www.pepy.tech/projects/vpso)
[![Maintainability](https://api.codeclimate.com/v1/badges/746f504e874cffeedae2/maintainability)](https://codeclimate.com/github/FilippoAiraldi/vectorized-particle-swarm/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/746f504e874cffeedae2/test_coverage)](https://codeclimate.com/github/FilippoAiraldi/vectorized-particle-swarm/test_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Installation

To install the package, run

```bash
pip install vpso
```

**vpso** has the following dependencies

- [Numba](https://numba.pydata.org/)
- [SciPy](https://scipy.org/)
- [typing_extensions](https://pypi.org/project/typing-extensions/).

For playing around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/vectorized-particle-swarm.git
```

---

## Usage


---

## Examples

Our [examples](https://github.com/FilippoAiraldi/vectorized-particle-swarm/tree/master/examples) subdirectory contains a simple use-case to get started with.

---

## License

The repository is provided under the MIT License. See the LICENSE file included with this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “csnn” (Nueral Networks with CasADi) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.

---

## References

<a id="1">[1]</a>
Z. H. Zhan, J. Zhang, Y. Li and H. S. H. Chung, "Adaptive Particle Swarm Optimization," in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 6, pp. 1362-1381, Dec. 2009, doi: 10.1109/TSMCB.2009.2015956.
