.. Soft Brownian Offset documentation master file, created by
   sphinx-quickstart on Thu Dec 12 01:26:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

********************
Soft Brownian Offset
********************

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Introduction
============

Soft Brownian Offset (SBO) defines an iterative approach to translate points by a most likely distance from a given dataset.
It can be used for generating out-of-distribution (OOD) samples.
It is based on Gaussian Hyperspheric Offset (GHO), which is also included in this package (see :ref:`below <Gaussian Hyperspheric Offset>`).


Installation
============

This project is hosted on :ref:`PyPI <https://pypi.org/project/sbo/>` and can therefore be installed easily through ``pip``:

.. code-block::

  pip install sbo

Dependending on your setup you may need to add ``--user`` after the install.


Usage
=====

For brevity's sake here's a short introduction to the library's usage:

.. code-block:: python
  :linenos:

  from sklearn.datasets import make_moons
  from sbo import soft_brownian_offset

  X, _ = make_moons(n_samples=60, noise=.08)
  X_ood = soft_brownian_offset(X, d_min=.35, d_off=.24, n_samples=120, softness=0)


Parameter overview
==================

The following plot gives an overview of possible choices for ``d_min`` (:math:`d^-`), ``d_off`` (:math:`d^+`) and ``softness`` (:math:`\sigma`):

.. image:: img/sbo-demo.svg
   :width: 600
   :alt: Plot of parameter overview

It was created using the following Python code:

.. code-block:: python
  :linenos:

  #!/usr/bin/env python3
  # Creates a plot for Soft Brownian Offset (SBO)

  import numpy as np
  import pylab as plt
  import itertools
  import sys

  from matplotlib import cm
  from sklearn.datasets import make_moons

  from sbo import soft_brownian_offset

  plt.rc('text', usetex=True)

  c = cm.tab10.colors

  def plot_data(X, y, ax=plt):
      ax.scatter(X[:, 0], X[:, 1], marker='x', s=20, label='ID', alpha=alpha, c=[c[-1]])
      ax.scatter(y[:, 0], y[:, 1], marker='+', label='SBO', alpha=alpha, c=[c[-6]])
      
  def plot_mindist(X, y, ax=plt):
      if len(X.shape) == 1:
          X = X[:, None]
      if len(y.shape) == 1:
          y = y[:, None]
      ax.hist(pairwise_distances(y, X).min(axis=1), bins=len(y) // 10)
      ax.set_xlabel("Minimum distance from ood to id")
      ax.set_ylabel("Count")
      
  def plot_data_mindist(X, y):
      fig, ax = plt.subplots(1, 2)
      plot_data(X, y, ax=ax[0])
      plot_mindist(X, y, ax=ax[1])
      plt.show()


  n_samples_id = 60
  n_samples_ood = 150
  noise = .08
  show_progress = False
  alpha = .6

  n_colrow = 3
  d_min = np.linspace(.25, .45, n_colrow)
  softness = np.linspace(0, 1, n_colrow)
  fig, ax = plt.subplots(n_colrow, n_colrow, sharex=True, sharey=True, figsize=(8.5, 9))

  X, _ = make_moons(n_samples=n_samples_id, noise=noise)
  for i, (d_min_, softness_) in enumerate(itertools.product(d_min, softness)):
      xy = i // n_colrow, i % n_colrow
      d_off_ = d_min_ * .7
      ax[xy].set_title(f"$d^- = {d_min_:.2f}\ d^+ = {d_off_:.2f}\ \sigma = {softness_}$")
      if softness_ == 0:
          softness_ = False
      y = soft_brownian_offset(X, d_min_, d_off_, n_samples=n_samples_ood, softness=softness_, show_progress=show_progress)
      plot_data(X, y, ax=ax[xy])
      if i // n_colrow == len(d_min) - 1:
          ax[xy].set_xlabel("$x_1$")
      if i % n_colrow == 0:
          ax[xy].set_ylabel("$x_2$")
  ax[0, n_colrow - 1].legend(loc='upper right')

  plt.tight_layout()
  plt.show()


Gaussian Hyperspheric Offset
============================

GHO is the basis for SBO and assumes :math:`\pmb{X}\sim\mathcal{N}`.
The following code's result displays the shortcomings if the assumption does not hold:

.. code-block:: python
  :linenos:

  from sklearn.datasets import make_moons
  from sbo import soft_brownian_offset, gaussian_hyperspheric_offset

  X, _ = make_moons(n_samples=60, noise=.08)
  X_ood = (gaussian_hyperspheric_offset(n_samples=220, mu=2, std=.3, n_dim=X.ndim) + X.mean()) * X.std()

