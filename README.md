# HADES: Fast Singularity Detection with Kernel

<img width="1008" alt="banner" src="https://github.com/uzulim/hades/assets/56029596/dc1956c7-da1c-4591-8118-014e690f5bc4">

by Uzu Lim, Harald Oberhauser, and Vidit Nanda

HADES is a fast singularity detection algorithm. Singularities are points in data where the Manifold Hypothesis fails, such as  cusps and self-intersections. HADES does *not* use topological methods, and instead works by:

1. Locally performing dimensionality reduction, and then
2. Performing a kernel goodness-of-fit test against the uniform distribution over a disk.

HADES stands for *Hypothesis-testing Algorithm for Detection and Exploration of Singularities*.

## Installation.

To install, clone the repository and use `Poetry`:
```
$ poetry install
```

## Usage
Given a Numpy array X where each row represents a data point, run the following:
```python
from hades import judge
verdict = judge(X)
```
Below is a convenient starting point for generating sample data, detecting singularities, and plotting them:
```python
from hades import judge
from hades.misc import plot, plot_filt
from hades.gen import two_circles

X = two_circles(5000, noise=0.01)
verdict = judge(X)

plot(X, c=verdict['score'], show=True)
plot_filt(X, verdict['label'], show=True)
```
The following are hyperparameters used by `hades` :
- `r`: radius
- `k`: k nearest neighbors
- `t`: threshold for PCA ($0 < t < 1$)
- `a`: kernel parameter ($0 < a < 1$)

(only one of `r` and `k` are used at each time, so that only 3 hyperparameters are relevant in each run)

The following are 3 modes of performing hyperparameter search. The default run is the fully automatic search, only over the radius parameter.
```python
# Mode 1. Fully automatic search
verdict = judge(X, search_auto=['r', 't'], 
                   search_res={'r': 5, 't': 3})

# Mode 2. Search over a specified grid of hyperparameters
verdict = judge(X, search_range={'r': (0.05, 0.15), 't': (0.7, 0.9)}, 
                   search_res={'r': 5, 't': 3})

# Mode 3. Search over a specified list of hyperparameters
verdict = judge(X, search_list = [{'a': 0.1, 'k': 50, 't': 0.9}, {'a': 0.5, 'k': 50, 't': 0.9}, {'a': 0.9, 'k': 50, 't': 0.9}])
```

## Notebooks
There are `Jupyter` notebooks in the `notebooks` folder that reproduce computational experiments in the paper. 
