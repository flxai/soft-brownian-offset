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

The technique allows for trivial OOD generation -- as shown above -- or more complex schemes that apply the transformation of learned representations.
For an in-depth look at the latter please refer to the paper that is also [as open access from the CVF](https://openaccess.thecvf.com/content/CVPR2021W/SAIAD/papers/Moller_Out-of-Distribution_Detection_and_Generation_Using_Soft_Brownian_Offset_Sampling_and_CVPRW_2021_paper.pdf).
For citations please see [*cite*](#cite).

## Demonstration

See the following plot to gain intuition on the approach's results:

![demonstration](docs/img/sbo-demo.svg)

Please see the [documentation](https://soft-brownian-offset.readthedocs.io/en/latest/#demonstration) for the source code to recreate the plot.

## Cite

Please cite SBO in your paper if it helps your research:

```bibtex
@inproceedings{MBH21,
  author    = {Möller, Felix and Botache, Diego and Huseljic, Denis and Heidecker, Florian and Bieshaar, Maarten and Sick, Bernhard},
  booktitle = {{Proc. of CVPR SAIAD Workshop}},
  title     = {{Out-of-distribution Detection and Generation using Soft Brownian Offset Sampling and Autoencoders}},
  year      = 2021
}
```

