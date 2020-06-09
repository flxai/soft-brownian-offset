import pylab as plt

from sklearn.datasets import make_moons
from sklearn.metrics import pairwise_distances
from sbo import soft_brownian_offset, gaussian_hyperspheric_offset

def plot_data(X, y, y_hard=None, ax=plt):
    ax.scatter(X[:, 0], X[:, 1], label='id')
    ax.scatter(y[:, 0], y[:, 1], s=8, label='ood')
    if y_hard is not None:
        ax.scatter(y_hard[:, 0], y_hard[:, 1], s=1, label='ood_hard')
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend()

def plot_mindist(X, y, ax=plt):
    if len(X.shape) == 1:
        X = X[:, None]
    if len(y.shape) == 1:
        y = y[:, None]
    ax.hist(pairwise_distances(y, X).min(axis=1), bins=len(y) // 10)
    ax.set_xlabel("Minimum distance from ood to id")
    ax.set_ylabel("Count")

def plot_data_mindist(X, y, y_hard=None):
    fig, ax = plt.subplots(1, 2)
    plot_data(X, y, y_hard, ax=ax[0])
    plot_mindist(X, y, ax=ax[1])
    plt.show()

n_samples_id = 48
n_samples_ood = 256
noise = .1
d_min = .3
d_off = .2
softness = 1
show_progress = False

X, _ = make_moons(n_samples=n_samples_id, noise=noise)
y = soft_brownian_offset(X, d_min, d_off, n_samples=n_samples_ood, softness=softness, show_progress=show_progress)
plot_data_mindist(X, y)
