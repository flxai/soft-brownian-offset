# Soft Brownian Offset
# (c) 2020 Felix Möller

import numpy as np

from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def soft_brownian_offset(X, d_min, d_off, n_samples=1, show_progress=False, softness=False, hs_scale=None,
                         random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n_dim = X.shape[1]
    ys = []
    iterator = range(n_samples)
    if show_progress:
        iterator = tqdm(iterator)
    for i in iterator:
        # Sample uniformly from X
        y = np.random.choice(X)
        # Move out of reach of other points
        skip = False
        while True:
            dist = pairwise_distances(y[:, None].T, X)[0]
            if dist.min() > 0:
                if not softness and dist.min() > d_min:
                    skip = True
                elif softness > 0:
                    p = 1 / (1 + np.exp((-dist.min() + d_min) / softness / d_min * 7))
                    if np.random.uniform() < p:
                        skip = True
                elif not isinstance(softness, bool):
                    raise ValueError("Softness should be float greater zero")
            if skip:
                break
            y += gaussian_hyperspheric_offset(1, n_dim=n_dim, hs_scale=hs_scale)[0] * d_off
        ys.append(np.array(y))
    return np.array(ys)


# Inspired by https://stackoverflow.com/a/33977530/10484131
def gaussian_hyperspheric_offset(n_samples, mu=4, std=.7, n_dim=3, hs_scale=None):
    vec = np.random.randn(n_dim, n_samples)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= np.random.normal(loc=mu, scale=std, size=n_samples)
    vec = vec.T
    if hs_scale is not None:
        return vec * hs_scale.std(axis=0) + hs_scale.mean(axis=0)
    return vec
