# soft-brownian-offset
Soft Brownian Offset (SBO) defines an iterative approach to translate points by a most likely distance from a given dataset.
It can be used for generating out-of-distribution samples.

## Installation

This project is hosted on [PyPI](https://pypi.org/project/sbo/) and can therefore be installed easily through `pip`:

```
pip install sbo
```

Dependending on your setup you may need to add `--user` after the `install`.

## Usage

For brevity's sake here's a short introduction to the library's usage:

```python
from sklearn.datasets import make_moons
from sbo import soft_brownian_offset

X, _ = make_moons(n_samples=60, noise=.08)
X_ood = soft_brownian_offset(X, d_min=.35, d_off=.24, n_samples=120, softness=0)
```

For more details please see the [documentation](https://soft-brownian-offset.readthedocs.io/en/latest/).

## Background

The technique is described in detail within the paper **TBA**. For citations please see [*cite*](#cite).

## Demonstration

See the following plot to gain intuition on the approach's results:

![demonstration](docs/img/sbo-demo.svg)

Please see the [documentation](https://soft-brownian-offset.readthedocs.io/en/latest/#demonstration) for the source code to recreate the plot.

## Cite

Please cite SBO in your paper if it helps your research **TBA**:

```
@article{name2020sbo,
  Author = {TBA},
  Journal = {arXiv preprint arXiv:TBA},
  Title = {TBA},
  Year = {2020}
}
```


