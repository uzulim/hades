# HADES: Fast Singularity Detection with Kernel

by Uzu Lim, Harald Oberhauser, and Vidit Nanda

Hades detects singularities in data, where the Manifold Hypothesis fails. Singularities include branching points, self-intersections, boundaries, and kinks in data distribution. Hades works by:

1. Locally performing dimensionality reduction, and then
2. Performing a kernel goodness-of-fit test against the uniform distribution over a disk. 

## Installation.

To install, use `poetry` or `pip`:
```
$ poetry install
```
or
```
$ pip install .
```

## Usage
Hades automatically searches the optimal hyperparameters and selects the threshold for the binary label. It only needs the input of data points and outputs a binary label on whether each data point is a singularity. Given a `numpy` array `X`, run the following:
```python
from hades import judge
verdict = judge(X)
```
The following code generates sample data, detects singularities, and plots it:
```python
from hades import judge
from hades.misc import plot, plot_filt
from hades.gen import two_circles

X = two_circles(5000, noise=0.01)
verdict = judge(X)

plot(X, c=verdict['score'], show=True)
plot_filt(X, verdict['label'], show=True)
```
By default, judge only runs hyperparameter search over the radius parameter. The other hyperparameters are:

- `r`: radius
- `k`: k nearest neighbors
- `t`: threshold for PCA ($0 < t < 1$)
- `a`: kernel parameter ($0 < a < 1$)

The following are 3 other modes of performing hyperparameter search.
```python
# Fully automatic search
verdict = judge(X, search_auto=['r', 't'], 
                   search_res={'r': 5, 't': 3})

# Search over a specified grid of hyperparameters
verdict = judge(X, search_range={'r': (0.05, 0.15), 't': (0.7, 0.9)}, 
                   search_res={'r': 5, 't': 3})

# Search over a specified list of hyperparameters
verdict = judge(X, search_list = [{'a': 0.1, 'k': 50, 't': 0.9}, {'a': 0.5, 'k': 50, 't': 0.9}, {'a': 0.9, 'k': 50, 't': 0.9}])
```

