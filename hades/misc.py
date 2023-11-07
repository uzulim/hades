import os
import sys
import math
import json
import time
import random
import datetime
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
from matplotlib import cm
from copy import deepcopy
from scipy.linalg import svd
from sklearn.neighbors import BallTree
from statsmodels.distributions.empirical_distribution import ECDF


def timestamp():
    return str(datetime.datetime.now())[:-7]


def char_replace(text, x, y):
    text2 = ''
    for char in text:
        if char == x:
            text2 += y
        else:
            text2 += char
    return text2


def range_select(arr, a, b, mode='[]'):
    if mode == '()':
        y = np.select([(a < arr) & (arr < b)], [arr], default=None)
    elif mode == '(]':
        y = np.select([(a < arr) & (arr <= b)], [arr], default=None)
    elif mode == '[)':
        y = np.select([(a <= arr) & (arr < b)], [arr], default=None)
    elif mode == '[]':
        y = np.select([(a <= arr) & (arr <= b)], [arr], default=None)
    else:
        assert False, 'Invalid input'
    y = y[y != None]
    return y


def power_add(a, b, p, mean=False):
    assert p > 0
    if mean:
        return ((a ** p + b ** p)/2) ** (1/p)
    else:
        return (a ** p + b ** p) ** (1 / p)


def power_subtract(a, b, p, mean=False):
    assert p > 0
    if mean:
        return ((a ** p - b ** p) / 2) ** (1 / p)
    else:
        return (a ** p - b ** p) ** (1 / p)


def pairwise_distance_sample(X, N_probe=1000):
    N, D = X.shape
    inds_probe1 = np.random.choice(N, N_probe)
    inds_probe2 = np.random.choice(N, N_probe)
    X1 = X[inds_probe1]
    X2 = X[inds_probe2]
    dists = np.linalg.norm(X1 - X2, axis=1)
    return dists


def dim_est_eig(eigs, pca_thr):
    eigs_total = np.sum(eigs)
    filt = np.cumsum(eigs) < (pca_thr * eigs_total)
    dim_est = 1 + np.sum(filt)
    return dim_est


def dim_est_local(Y, pca_thr):
    """
    Dimension estimation using PCA
    """
    svd_out = svd(Y)
    pcomp = svd_out[0]
    psv = svd_out[1]
    eigs = np.power(psv, 2)
    return dim_est_eig(eigs, pca_thr)


def dim_est_global(X, N_dp=None, k_mult=3, pca_thr=0.95, round_output=True):
    """
    Local-Global dimension estimation using PCA
    """
    N = X.shape[0]
    D = X.shape[1]
    if N_dp is None:
        N_dp = N

    # Subsample
    N_dp = min(N_dp, N)  # Can't sample more than the sample itself
    k_dp = k_mult * D  # Size of neighbor to examine for dimension estimate
    inds_dp = np.random.choice(N, N_dp, replace=False)
    X_dp = X[inds_dp]
    bt = BallTree(X)
    bt_dist, bt_ind = bt.query(X_dp, k_dp + 1)  # dist, ind

    # Average local dimension estimates
    d_ests = []
    for i in range(N_dp):
        X_near_now = X[bt_ind[i]]
        Y = X_near_now - X[inds_dp[i]]
        d_est_now = dim_est_local(Y, pca_thr)
        d_ests.append(d_est_now)
    d_est = np.mean(d_ests)

    # Return, round if asked
    if round_output:
        d_est = round(d_est)
        d_est = np.max([2, d_est]).astype(int)
        d_est = np.min([D, d_est]).astype(int)
    return d_est


def unif_eig_vec(d, D):
    v1 = (1/(d+2)) * np.ones(d)
    v2 = np.zeros(D-d)
    v = np.hstack([v1, v2])
    return v


def pca_reduce(Y, pca_thr):
    """
    Dimension reduction using PCA
    """
    N = Y.shape[0]
    svd_out = svd(Y)
    pcomp = svd_out[0]
    psv = svd_out[1]
    eigs = np.power(psv, 2)
    dim_est = dim_est_eig(eigs, pca_thr)
    assert dim_est > 0
    pcomp_thr = pcomp[:, :dim_est]  # principal components
    psv_thr = psv[:dim_est]  # principal singular values
    Y_proj = pcomp_thr @ np.diag(psv_thr)  # projected Y

    return Y_proj


def eig_thr_from_dim(eigs, d):
    eigs = np.array(eigs)
    D = eigs.size
    tails = np.cumsum(eigs)/np.sum(eigs)
    tails = np.hstack([0, tails])
    d0 = min(D, max(0, int(d)))  # clip between [0, D-1]
    d1 = min(D, max(0, int(d+1)))  # clip between [0, D-1]
    a = d - d0
    T1, T2 = tails[d0], tails[d1]  # minus 1 for array access
    T = (1-a) * T1 + a * T2  # linear interpolation
    return T


def print_toggle(choice, stdout_stream=None):
    if choice:
        if stdout_stream is None:
            stdout_stream = sys.__stdout__
        sys.stdout = stdout_stream
    else:
        sys.stdout = open(os.devnull, 'w')


def nn_aug(X, k, n_iter):
    """
    Nearest neighbor augmentation
    """
    X2 = X.copy()
    for i in range(n_iter):
        balltree = BallTree(X2)
        X_nd, X_nind = balltree.query(X2, k)
        X_nind = X_nind[:, 1:]
        X3 = []
        for i in range(X2.shape[0]):
            new_pt = np.mean(X2[X_nind[i]], axis=0)
            X3.append(new_pt)
        X3 = np.array(X3)
        X2 = np.vstack([X2, X3])
    return X2


def bounding_box_size(X):
    N, D = X.shape
    axis_minmax = [np.max(X.T[i]) - np.min(X.T[i]) for i in range(D)]
    total_minmax = np.max(axis_minmax)
    return total_minmax


def rotate_3d(X, t_x=0, t_y=0, t_z=0):
    assert X.shape[1] == 3

    t_y = -t_y  # Convention

    t_x *= (np.pi / 180)
    t_y *= (np.pi / 180)
    t_z *= (np.pi / 180)

    c_x, s_x = np.cos(t_x), np.sin(t_x)
    c_y, s_y = np.cos(t_y), np.sin(t_y)
    c_z, s_z = np.cos(t_z), np.sin(t_z)

    mat_x = np.array([[1, 0, 0], [0, c_x, s_x], [0, -s_x, c_x]])
    mat_y = np.array([[c_y, 0, -s_y], [0, 1, 0], [s_y, 0, c_y]])
    mat_z = np.array([[c_z, s_z, 0], [-s_z, c_z, 0], [0, 0, 1]])

    return X @ mat_x @ mat_y @ mat_z


def print_progress_bar(tick, max_tick, message="", percent_decimal=2, eta_est=None):
    proportion = (tick + 1) / max_tick
    proportion_prev = tick / max_tick
    tick_d = math.floor((proportion) * 100 * (10 ** percent_decimal)) / (10 ** percent_decimal)
    tick_d_prev = math.floor((proportion_prev) * 100 * (10 ** percent_decimal)) / (10 ** percent_decimal)
    if tick_d > tick_d_prev:
        tick_d_str = str(tick_d)
        tick_d_str = ("0" * (3 - (tick_d_str.find(".")))) + tick_d_str
        tick_d_str = tick_d_str + ("0" * (3 - (len(tick_d_str) - tick_d_str.find("."))))
        if eta_est != None:
            sys.stdout.write("\r" + message + " " + tick_d_str + "%")
            sys.stdout.write(("|" + "█" * int(tick_d / 5)) + (" " * (20 - int(tick_d / 5))) + "| "
                             + "ETA ≈ " + str(datetime.timedelta(seconds=int((max_tick - tick - 1) * eta_est))))
        else:
            sys.stdout.write("\r" + message + tick_d_str + "%")
            sys.stdout.write(("|" + "█" * int(tick_d / 5)) + (" " * (20 - int(tick_d / 5))) + "|")
    if max_tick == (tick + 1):
        sys.stdout.write("\r" + message + tick_d_str + "%")
        sys.stdout.write("|" + "█" * 20 + "|")
        sys.stdout.write('')


def plot(X, ax=None, show_grid=False, reposition=True, show=False, **kwargs):
    dim = X.shape[1]
    assert dim in {2, 3}, "Input array must be either 2D or 3D."

    # Set axis object
    if ax is None:
        if dim == 2:
            ax = plt.subplot(111)
        if dim == 3:
            ax = plt.subplot(111, projection='3d', proj_type='ortho')

    if reposition:
        # Bounds
        x_min = np.min(X[:, 0])
        x_max = np.max(X[:, 0])
        x_rad = (x_max - x_min) / 2
        x_cen = (x_max + x_min) / 2

        y_min = np.min(X[:, 1])
        y_max = np.max(X[:, 1])
        y_rad = (y_max - y_min) / 2
        y_cen = (y_max + y_min) / 2

        padscale2d = 1.1
        padscale3d = 0.9
        padextra3d = 0.1
        xyz_rad = max(x_rad, y_rad)
        if dim == 3:
            z_min = np.min(X[:, 2])
            z_max = np.max(X[:, 2])
            z_rad = (z_max - z_min) / 2
            z_cen = (z_max + z_min) / 2
            xyz_rad = max(x_rad, y_rad, z_rad)
        if dim == 2:
            x_min = x_cen - padscale2d * xyz_rad
            x_max = x_cen + padscale2d * xyz_rad
            y_min = y_cen - padscale2d * xyz_rad
            y_max = y_cen + padscale2d * xyz_rad
        if dim == 3:
            x_min = x_cen - padscale3d * xyz_rad - padextra3d
            x_max = x_cen + padscale3d * xyz_rad + padextra3d
            y_min = y_cen - padscale3d * xyz_rad - padextra3d
            y_max = y_cen + padscale3d * xyz_rad + padextra3d
            z_min = z_cen - padscale3d * xyz_rad - padextra3d
            z_max = z_cen + padscale3d * xyz_rad + padextra3d

        if dim == 2:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect(1)
        if dim == 3:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

    # Grid
    if not show_grid:
        ax.set_xticks([])
        ax.set_yticks([])
        if dim == 3:
            ax.set_zticks([])

    # Plot
    if dim == 2:
        ax.scatter(X.T[0], X.T[1], **kwargs)
    elif dim == 3:
        ax.scatter(X.T[0], X.T[1], X.T[2], **kwargs)

    if show:
        plt.show()

    return ax


def plot_filt(X, filt, c1='gray', c2='blue', a1=0.1, a2=1, s1=5, s2=5, show=False, **kwargs):
    nfilt = np.logical_not(filt)
    Xf = X[filt]
    Xnf = X[nfilt]

    ax = None
    if np.any(nfilt):
        ax = plot(Xnf, ax=ax, c=c1, alpha=a1, s=s1, **kwargs)
    if np.any(filt):
        ax = plot(Xf, ax=ax, c=c2, alpha=a2, s=s2, reposition=False, **kwargs)

    if show:
        plt.show()

    return ax


def gallery(X_all, names, outputs, fig_scale=2, ncols=3, save_fig=True, picture_format='png',
            mode = 'filt', title='', title_y=1.0, title_fontsize=20,
            cbar_dim = [-0.05, 0.15, 0.05, 1], cbar_ticks = 5,
            v_minmax = (0, 15), **kwargs):
    assert mode in {'filt', 'pval'}
    timestamp_now = timestamp()

    assert len(X_all) == len(outputs)
    N_output = len(X_all)
    nrows = np.ceil(N_output / ncols).astype(int)
    # (row, col) = (i,j) --> output_ind = 3*i + j
    # output_ind = k --> (row, col) = (np.floor(k/3), k%3)

    # Matplotlib setup
    fig, ax = plt.subplots(ncols=ncols,
                           nrows=nrows,
                           figsize=(fig_scale * nrows, fig_scale * ncols))
    fig.set_size_inches(fig_scale * ncols, fig_scale * nrows)
    plt.subplots_adjust(wspace=0.03, hspace=0.15)
    plt.suptitle(title, fontsize=title_fontsize, y=title_y)

    for i in range(nrows):
        for j in range(ncols):
            if (nrows == 1) and (ncols == 1):
                ax_now = ax
            elif nrows == 1:
                ax_now = ax[j]
            elif ncols == 1:
                ax_now = ax[i]
            else:
                ax_now = ax[i, j]
            ax_now.set_xticks([])  # No tick labels
            ax_now.set_yticks([])  # No tick labels
            ax_now.spines['top'].set_visible(True)
            ax_now.spines['right'].set_visible(True)
            ax_now.spines['bottom'].set_visible(True)
            ax_now.spines['left'].set_visible(True)
            ax_now.set_facecolor('#FFFFFF00')  # RGBA code, last two 00 = Transparent
            if ncols * i + j < N_output:
                ax_now.set_xlabel(names[ncols * i + j], fontsize=15)

    # Draw main objects
    for ind, output in enumerate(outputs):
        X = X_all[ind]
        X_dim = X.shape[1]
        assert (X_dim in {2, 3}), 'Only 2 or 3-dimensional examples allowed.'

        x_min = np.min(X[:, 0])
        x_max = np.max(X[:, 0])
        x_rad = (x_max - x_min) / 2
        x_cen = (x_max + x_min) / 2

        y_min = np.min(X[:, 1])
        y_max = np.max(X[:, 1])
        y_rad = (y_max - y_min) / 2
        y_cen = (y_max + y_min) / 2

        padscale2d = 1.1
        padscale3d = 0.9
        padextra3d = 0.1
        xyz_rad = max(x_rad, y_rad)
        if X_dim == 3:
            z_min = np.min(X[:, 2])
            z_max = np.max(X[:, 2])
            z_rad = (z_max - z_min) / 2
            z_cen = (z_max + z_min) / 2
            xyz_rad = max(x_rad, y_rad, z_rad)
        if X_dim == 2:
            x_min = x_cen - padscale2d * xyz_rad
            x_max = x_cen + padscale2d * xyz_rad
            y_min = y_cen - padscale2d * xyz_rad
            y_max = y_cen + padscale2d * xyz_rad
        if X_dim == 3:
            x_min = x_cen - padscale3d * xyz_rad - padextra3d
            x_max = x_cen + padscale3d * xyz_rad + padextra3d
            y_min = y_cen - padscale3d * xyz_rad - padextra3d
            y_max = y_cen + padscale3d * xyz_rad + padextra3d
            z_min = z_cen - padscale3d * xyz_rad - padextra3d
            z_max = z_cen + padscale3d * xyz_rad + padextra3d

        # Thresholded
        filt = output[0]
        # filt = output
        Xf = X[filt]
        Xnf = X[np.logical_not(filt)]
        if X_dim == 2:
            ax = fig.add_subplot(nrows, ncols, ind + 1, frame_on=False)
            if mode == 'filt':
                ax.scatter(Xnf.T[0], Xnf.T[1], c='gray', alpha=0.3, **kwargs)
                ax.scatter(Xf.T[0], Xf.T[1], c='blue', **kwargs)
            elif mode == 'pval':
                ax.scatter(X.T[0], X.T[1], c=output[1]/np.log(10), vmin=v_minmax[0], vmax=v_minmax[1], alpha=1.0, **kwargs)
        elif X_dim == 3:
            gray_alpha = 0.05
            ax = fig.add_subplot(nrows, ncols, ind + 1, projection='3d', proj_type='ortho', frame_on=False)
            # ax.view_init(elev=30, azim=45)
            ax.view_init(elev=20, azim=125)
            if mode == 'filt':
                ax.scatter(Xnf.T[0], Xnf.T[1], Xnf.T[2], c='gray', alpha=gray_alpha, zorder=0, **kwargs)
                ax.scatter(Xf.T[0], Xf.T[1], Xf.T[2], c='blue', alpha=1, zorder=100, **kwargs)
            elif mode == 'pval':
                ax.scatter(X.T[0], X.T[1], X.T[2], c=output[1]/np.log(10), vmin=v_minmax[0], vmax=v_minmax[1], alpha=1.0, **kwargs)
        if X_dim == 2:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect(1)
        if X_dim == 3:
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])
            ax.set_zlim([z_min, z_max])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        ax.tick_params(left=False, bottom=False)  # No tickmarks
        ax.set_facecolor('#FFFFFF00')  # RGBA code, last two 00 = Transparent

    # Add colorbar for pval mode
    if mode == 'pval':
        cbar = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=v_minmax[0], vmax=v_minmax[1]), cmap='viridis')
        cbax = fig.add_axes(cbar_dim)  # Location of colorbar

        cbar2 = plt.colorbar(cbar, cax=cbax)
        cbar2.locator = ticker.MaxNLocator(nbins=cbar_ticks)
        cbar2.update_ticks()

    if save_fig:
        filename = f'Gallery {timestamp_now}.{picture_format}'
        output_dir = 'output'
        plt.savefig(output_dir + '/' + filename)
