import numpy as np
import json
import os
import time
import math
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import cdist

import gen


def unif_mmd(X, ker_alpha, N_cutoff=10):
    """
    Squared MMD of an empirical measure vs. the uniform distribution over the unit disk of same dimension.
    The geometric series kernel K(x,y) = 1/(1-a * <x,y>) is used, where a = ker_alpha.
    :param X: Input d-dimensional sample
    :param ker_alpha: Value a in K(x,y) = 1/(1-a * <x,y>). Default = 0.5
    :param N_cutoff: Accuracy parameter. If large, slower and more accurate. Default = 10
    :return:
    """
    (N, d) = X.shape
    # Geometric series kernel
    # K(x,y) = \sum_k a^k <x,y>^k, a = ker_alpha
    Gram = X.dot(X.T)
    X_norms = np.diag(Gram)
    A_mat = 1 / (1 - ker_alpha * Gram)
    A = np.sum(A_mat) / (N ** 2)
    BC = 0
    BC_const = math.gamma(1 + (d / 2)) / math.sqrt(math.pi)
    for k in np.arange(0, N_cutoff):
        summand = d / (d + 2 * k) - 2 * np.mean(np.power(X_norms, k))
        summand *= (ker_alpha ** (2 * k))
        summand *= (math.gamma(k + 0.5) / math.gamma(k + (d / 2) + 1))
        BC += summand
    BC *= BC_const
    return A + BC


def gen_mmd_data(alpha_arr, d_arr, N_arr, N_iter, N_cutoff=10, x_res=1000, y_cutoff=0.9, save=False, save_filename='mmd_data'):
    """
    Return a dict
    Index format = (alpha, d)
    Item format = [N * MMD, x_cut, y_cut, reg_coef]

    N_cutoff: Cutoff for unif_mmd evaluation
    x_res: Grid resolution for ECDF evaluation, used for tail regression
    y_cutoff: Cutoff point for taking tail of the ECDF
    """
    output = {}
    for alpha in alpha_arr:
        for d in d_arr:
            key = (alpha, d)
            print(f'Working on (alpha, d) = {key}...')
            # Rescaled MMD data; collection of N * MMD_squared
            mmd_item = []
            for N in N_arr:
                mmd_item += [N * unif_mmd(gen.disk(N, d,
                                          force_scale=False),  # Radius 1
                                          ker_alpha=alpha,
                                          N_cutoff=N_cutoff)
                             for _ in range(N_iter)]
            mmd_item = np.sort(mmd_item)

            ecdf = ECDF(mmd_item)
            x_max = np.max(mmd_item)

            # Evaluate ECDF on a grid of values
            x_grid = np.linspace(0, x_max, x_res, endpoint=False)
            y_grid = ecdf(x_grid)
            assert np.sum(y_grid == 1) == 0, "y values should be < 1"
            z_grid = -np.log(1 - y_grid)

            # Snip x_grid at where y value > y_cutoff, to feed to regression.
            snip_ind = np.min(np.nonzero(y_grid > y_cutoff))
            x_cut = x_grid[snip_ind]
            y_cut = ecdf(x_cut)

            # Linear regression of tail, starting from x_snip
            x_snipped = x_grid[snip_ind:].reshape(-1, 1)
            z_snipped = z_grid[snip_ind:]
            reg = LinearRegression().fit(x_snipped, z_snipped)
            reg_score = reg.score(x_snipped, z_snipped)

            # Save item
            item = {}
            item['mmd'] = list(mmd_item)
            item['x_cut'] = float(x_cut)
            item['y_cut'] = float(y_cut)
            item['reg_coef'] = float(reg.coef_[0])

            # Not used currently, but saved anyway
            item['reg_intercept'] = float(reg.intercept_)
            item['reg_score'] = float(reg_score)

            output[key] = item
    if save:
        mmd_json_save(output, save_filename)
    return output


def mmd_pval_fun(t, alpha, d, mmd_data):
    """
    Function that evaluates p-value based on the provided MMD data
    """

    # Find nearest key value
    key = np.array([alpha, d])  # Initial key
    keys_all = [np.array(key_each) for key_each in mmd_data.keys()]
    key_dists = [np.linalg.norm(key - key_each) for key_each in keys_all]
    key = tuple(keys_all[np.argmin(key_dists)])  # Updated key

    # mmd_data is a dict with keys (alpha, d),
    # with each item being dict with keys ecdf, x_cut, y_cut, reg_coef
    ecdf = mmd_data[key]['ecdf']
    x_cut = mmd_data[key]['x_cut']
    y_cut = mmd_data[key]['y_cut']
    reg_coef = mmd_data[key]['reg_coef']

    if t < x_cut:
        return 1 - ecdf(t)  # This is p-value
    else:
        tail_len = t - x_cut
        damper = np.exp(- reg_coef * tail_len)
        return damper * (1 - y_cut)


def string_to_tuple(stringin):
    # Get [1: -1] to remove braces
    return tuple(map(float, stringin[1: -1].split(', ')))


def mmd_prepare_json(dictin):
    """
    Input: dict of numpy arrays indexed by tuples of numbers
    Tuple keys are changed into strings, Array entries are changed into lists
    Output: dict of lists indexed by strings
    """
    dictout = {str(key): dictin[key] for key in dictin.keys()}
    return dictout


def mmd_unprepare_json(dictin):
    """
    Input: dict of lists indexed by strings
    Reverses mmd_prepare_json
    Output: dict of numpy arrays indexed by tuples of numbers
    """
    dictout = {string_to_tuple(key): dictin[key] for key in dictin.keys()}
    return dictout


def mmd_make_ecdf(dictin):
    dictout = {}
    for key in dictin.keys():
        item = dictin[key]
        item['ecdf'] = ECDF(item['mmd'])
        dictout[key] = item
    return dictout


def mmd_json_save(dictin, filename):
    with open(f'{filename}.json', 'w') as f:
        json.dump(mmd_prepare_json(dictin), f)


def mmd_json_load(filename):
    if os.path.isfile(f'{filename}.json'):
        with open(f'{filename}.json', 'r') as f:
            dict_load = mmd_unprepare_json(json.load(f))
        return dict_load
    else:
        raise Exception(f'MMD Data not found in {filename}.json')


if __name__ == '__main__':
    # Main function generates and saves MMD Data

    params = {'alpha_arr': [0.1, 0.3, 0.5, 0.7, 0.9],
              'd_arr': np.arange(1, 51),
              'N_arr': [10],
              'N_iter': 10000,
              'N_cutoff': 10,
              'x_res': 1000,
              'y_cutoff': 0.9,
              'save': True,
              'save_filename': 'mmd_data'}

    print(f'Starting MMD Data Generation')
    gen_mmd_data(**params)
    print(f'Done')

