import time
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from kneed import KneeLocator
from skdim.id import MLE as skdim_mle
from sklearn.metrics import roc_auc_score
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import ParameterGrid
from scipy.stats import gaussian_kde

from . import gen, mmd, misc
from .misc import pca_reduce, print_toggle, power_add, power_subtract
from .mmd import mmd_json_load, mmd_make_ecdf, mmd_pval_fun


def uniformity(Y, pca_thr=0.95, ker_alpha=0.5, mmd_expr_cutoff=10, mmd_data=None, reduce_dim=True, max_mult=np.inf):
    """
    Uniformity test for a neighborhood of a point.
    p-value computed from dimensionality reduction, MMD computation and p-value lookup.
    """
    if mmd_data is None:
        mmd_filename = 'mmd_data'
        mmd_data = mmd_json_load(mmd_filename)
        mmd_data = mmd_make_ecdf(mmd_data)
    N, D = Y.shape

    # Subsample if N_max < N and get Y1
    N_max = D * max_mult
    if N > N_max:
        inds1 = np.random.choice(N, N_max, replace=False)
        Y1 = Y[inds1]
    else:
        Y1 = Y

    # Reduce dimension of Y1 to get Y2
    if reduce_dim:
        Y2 = pca_reduce(Y1, pca_thr)
    else:
        Y2 = Y1
    d2 = Y2.shape[1]

    # Calculate MMD and the p-value.
    mmd_sq = mmd.unif_mmd(Y2, ker_alpha=ker_alpha, N_cutoff=mmd_expr_cutoff)
    pval = mmd_pval_fun(N * mmd_sq, ker_alpha, d2, mmd_data)
    return {'pval': pval, 'mmd': mmd_sq, 'dim': d2}


def hp_grid(r=None, k=None, pca_thr=0.95, ker_alpha=0.5):
    """
    Make a grid of hyperparameters
    """
    in_r = r is not None  # Check if r_all is inputted
    in_k = k is not None  # Check if k_all is inputted
    assert in_r + in_k == 1, 'Exactly one of r_all and k_all should be specified.'

    if not (isinstance(r, list) or isinstance(r, np.ndarray)):
        r = [r]
    if not (isinstance(k, list) or isinstance(k, np.ndarray)):
        k = [k]
    if not (isinstance(pca_thr, list) or isinstance(pca_thr, np.ndarray)):
        pca_thr = [pca_thr]
    if not (isinstance(ker_alpha, list) or isinstance(ker_alpha, np.ndarray)):
        ker_alpha = [ker_alpha]

    if in_r:
        grid = list(ParameterGrid({'r': r, 'pca_thr': pca_thr, 'ker_alpha': ker_alpha}))
    elif in_k:
        grid = list(ParameterGrid({'k': k, 'pca_thr': pca_thr, 'ker_alpha': ker_alpha}))
    return grid


def damp_fn(t, a, b=5):
    """
    Auxiliary damping function for scoring Hades output
    """
    assert (0 <= a) and (a <= 1)
    assert (0 <= t) and (t <= 1)
    if t <= a:
        return 0
    out = np.power((t - a) / (1 - a), b)
    return out


def score_fn(dauc, pur, prop, reg_coef):
    """
    Scoring function, based on DAUC, purity, and proportion
    High score = bad
    """
    dpur = (dauc + pur) / 2
    ddpur = sum(damp_fn(1 - t, 0.5, 5) for t in dpur)
    score = ddpur + reg_coef * damp_fn(prop, 0, 2)
    return score


def compute_kde(arr, n_steps):
    """
    Compute KDE smoothing of an array
    """
    kde_func = gaussian_kde(arr, bw_method=0.2)
    x_min, x_max = np.min(arr), np.max(arr)
    h_x = np.linspace(x_min, x_max, n_steps)
    h_y = kde_func(h_x)
    h_y = h_y / np.sum(h_y)  # normalise
    return (h_x, h_y)


def compute_pdf_knee(g_x, g_y, S=1.0, alert_none=True):
    """
    Get knee from an empirical PDF input
    """

    ind_decr = np.min(np.nonzero(g_y[1:] < g_y[:-1]))
    g_x = g_x[ind_decr:]
    g_y = g_y[ind_decr:]
    print(f'ind_decr = {ind_decr}')

    knee_val = KneeLocator(g_x, g_y, S=S, curve="convex", direction="decreasing").knee

    if (knee_val is None) and alert_none:
        print(f'Knee = None, set by default to 0')
        knee_val = 0

    comp_arr = g_x <= knee_val
    if np.any(comp_arr):
        knee_ind = int(np.max(np.nonzero(g_x <= knee_val)))
    else:
        knee_ind = 0

    print(f'knee = {knee_val}, knee_ind = {knee_ind}')

    output = {'knee_val': knee_val, 'knee_ind': knee_ind}
    return output


def compute_local_scale(X, n_probe=100, res_inter=100, k_extent=50, k_mult=3, thr_S=1.0):
    """
    Compute local scale parameters
    """
    
    print(f'Computing scale parameters...')
    # Detect scale at which data points show stable manifold-like structure
    N, D = X.shape
    k_all = k_mult * D * np.arange(1, k_extent + 1)

    # Nearest neighbors
    k_max = np.max(k_all)
    inds_probe = np.random.choice(N, n_probe, replace=False)
    X_probe = X[inds_probe]
    bt_dist, bt_ind = BallTree(X).query(X_probe, k_max + 1)  # Balltree

    # Neighborhoods and Local dimensions.
    # Both are List of dicts. List index = probe points, Dict index = k.
    # nbds[i][k] is a numpy array. dims[i][k] is a float.
    nbds = [{k: X[bt_ind[i][:k]] - X_probe[i] for k in k_all} for i in range(n_probe)]
    dim_fun = lambda t: skdim_mle().fit_predict(t)
    dims = [{k: dim_fun(nbds[i][k]) for k in k_all} for i in range(n_probe)]

    dims_reformat = [np.array(list(item.values())) for item in dims]
    dim_diffs = [np.abs(item[1:] - item[:-1]) for item in dims_reformat]

    # Average and estimate knee
    dim_fns = [(bt_dist[i, k_all][:-1], dim_diffs[i]) for i in range(n_probe)]
    dist_max = np.max(bt_dist)
    x_inter = np.linspace(0, dist_max, res_inter)
    y_inter_all = np.vstack([np.interp(x_inter, dim_fns[i][0], dim_fns[i][1]) for i in range(n_probe)])
    y_inter = np.mean(y_inter_all, axis=0)

    # Enforce decreasing
    y_inter = np.abs(y_inter - y_inter[-1])
    knee_r = -KneeLocator(-x_inter, y_inter, S=thr_S, curve="convex", direction="increasing").knee
    knee_k_pre = [np.argmin(np.abs(item - knee_r)) for item in bt_dist]
    knee_k = int(np.median(knee_k_pre))

    print(f'Radius scale = {knee_r}, kNN scale = {knee_k}')
    return {'r': knee_r, 'k': knee_k}


def compute_supc(pvals, n_steps=10, thr=10000, prop_min=0.01, prop_max=0.5):
    """
    SUPC = Small Uniformity P-value Concentration.
    Calculates normalised proportion of small uniformity p-values.
    """
    N = pvals.size

    logp_max = -np.log(np.quantile(pvals, prop_min))
    logp_min = -np.log(np.quantile(pvals, prop_max))

    s_arr = np.linspace(logp_min, logp_max, n_steps)
    thrs = np.exp(-s_arr)

    nm_props = [np.sum(pvals < t) / (N * t) for t in thrs]
    peak = np.max(nm_props)
    verdict = peak < thr
    output = {'verdict': verdict, 'peak': peak, 'nm_props': nm_props, 'logp_max': logp_max, 'logp_min': logp_min}
    return output


class Hades:
    """Classifier for singularity detection."""

    def __init__(self):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings = {'bt_leaf': 40,
                         'mmd_expr_cutoff': 10,
                         'max_mult': 100,
                         'multithread': True,
                         'n_cores': -1,
                         'src_dir': src_dir,
                         'mmd_filename': f'{src_dir}/mmd_data'}
        
        # hhp = Hyper-hyperparameter
        # hhp is used to run hyperparameter search
        self.hhp = {'expand_hps': False,
                    'dauc_r_mult': 0.5,
                    'filt_knee_S': 0.5,
                    'filt_n_steps': 100,
                    'score_regcoef': 30,
                    'r_mult_initial': (1.5, 5.0),
                    'r_mult_final': (0.5, 8.0),
                    'bounds_initial': {'pca_thr': (0.7, 0.9), 'ker_alpha': (0.3, 0.7)},
                    'bounds_final': {'pca_thr': (0.5, 0.95), 'ker_alpha': (0.05, 0.95)},
                    'grid_resolution': {'r': 5, 'pca_thr': 3, 'ker_alpha': 3}}

        # Load MMD data
        mmd_data = mmd_json_load(self.settings['mmd_filename'])
        mmd_data = mmd_make_ecdf(mmd_data)
        self.mmd_data = mmd_data

    """
    Automatic Hyperparameter Search
    """

    def compute_dim_est(self, round_output=False):
        """
        Dimension estimate is used to run compute_k_bounds and compute_r_bounds
        """
        X = self.X
        X_dim_est = skdim_mle().fit_transform(X)
        if round_output:
            X_dim_est = int(X_dim_est)
        self.X_dim_est = X_dim_est
        return X_dim_est

    def compute_hp_bounds(self, neighbor):
        """
        Updates self.hhp['bounds_initial'] and self.hhp['bounds_final']
        by adding keys of 'r' or 'k', with values being pairs specifying initial and final bounds for hyperparameter search
        """
        assert neighbor in {'r', 'k'}
        self.local_scale = compute_local_scale(self.X)
        r_scale, k_scale = self.local_scale['r'], self.local_scale['k']

        if neighbor == 'r':
            self.hhp['bounds_initial']['r'] = r_scale * np.array(self.hhp['r_mult_initial'])
            self.hhp['bounds_final']['r'] = r_scale * np.array(self.hhp['r_mult_final'])
        elif neighbor == 'k':
            self.hhp['bounds_initial']['k'] = np.round(k_scale * np.array(self.hhp['r_mult_initial'])).astype(int)
            self.hhp['bounds_final']['k'] = np.round(k_scale * np.array(self.hhp['r_mult_final'])).astype(int)

    def make_hpgrid(self, res_all, power_all, lower_all, upper_all, freeze, antifreeze, frozen_value, endpoint=False):
        """
        Make a grid of hyperparameters
        If res_all = [r1, ... rk], make a dict indexed by r1 x ... x rk grid points
        each indexing key is a tuple of integers, and item is a power-scaled grid point.
        """

        # Step 1
        # pre_output is a dict of pairs (X, L)
        # where X = coordinate in a hyperparameter grid
        # and L = list of hyperparameter values in antifreeze
        k = len(res_all)
        assert len(power_all) == k and len(lower_all) == k and len(upper_all) == k

        # Coordinates for indexing hyperparameters
        coords = [np.arange(item) for item in res_all]
        coords = np.meshgrid(*coords)
        coords = [item.reshape(-1) for item in coords]
        coords = np.vstack(coords).T

        # Hyperparameter values
        hpvals = [np.power(np.linspace(lower ** d, upper ** d, res, endpoint=endpoint), 1 / d)
                  for res, d, lower, upper in zip(res_all, power_all, lower_all, upper_all)]
        # Change entries of 'k' entry into integer and remove duplicates
        if 'k' in antifreeze:
            ind_k = antifreeze.index('k')
            list_k = hpvals[ind_k]
            list_k = np.round(list_k).astype(int)  # Round to integer values
            hpvals[ind_k] = list(set(list_k))  # Remove duplicates
        hpvals = np.meshgrid(*hpvals)
        hpvals = [item.reshape(-1) for item in hpvals]
        hpvals = np.vstack(hpvals).T
        pre_output = {tuple(coord): hpval for coord, hpval in zip(coords, hpvals)}

        # Step 2
        # output is a dict of pairs (X, L')
        # where X = coordinate in a hyperparameter grid
        # and L' = dict indexed by hyperparameter names in both freeze and antifreeze.
        # This is the point where we insert frozen hyperparameter values from the variable frozen_values
        output = {}
        for key, antifreeze_value in pre_output.items():
            item = {}
            for i, hp_name in enumerate(antifreeze):
                item[hp_name] = antifreeze_value[i]
            for i, hp_name in enumerate(freeze):
                item[hp_name] = frozen_value[i]
            output[key] = item
        return output

    """
    Balltree
    """

    def store_balltree(self, local_rk, mode, target):
        """
        If target == 'all', then balltree examines all points X
        If target == 'probe', then balltree examines only the probe points (from the probes)
        """
        X = self.X
        inds_probe = self.inds_probe
        Xp = X[inds_probe]
        self.mode_rk = mode
        assert mode in {'r', 'k'}, 'mode should be either "r" or "k".'
        in_r = mode == 'r'
        in_k = mode == 'k'
        assert target in {'all', 'probe'}, 'target should be either "all" or "probe"'
        tar_all = target == 'all'
        tar_pro = target == 'probe'

        # Set the target point cloud to be queried by balltree
        if tar_all:
            X_tar = X
        if tar_pro:
            X_tar = Xp

        # Check if balltree has been pre-computed, dividing tar_all and tar_pro cases.
        # Have self.bt_querytype be shared over tar_all and tar_pro cases
        precomputed_tar_all = ('bt_out' in self.__dict__ and tar_all)
        precomputed_tar_pro = ('pbt_out' in self.__dict__ and tar_pro)
        if precomputed_tar_all or precomputed_tar_pro:
            if in_r:
                assert self.bt_querytype == 'r'
            if in_k:
                assert self.bt_querytype == 'k'
            if local_rk <= self.bt_rk_max:
                print('Unneccesary to recompute')
                # Return if unnecessary to re-compute
                return

        # bt = balltree (tar_all case), pbt = probe balltree (tar_pro case).
        if tar_all:
            self.bt_object = BallTree(X_tar, leaf_size=self.settings['bt_leaf'])
            self.bt_rk_max = local_rk
            bt_object = self.bt_object
        if tar_pro:
            self.pbt_object = BallTree(X_tar, leaf_size=self.settings['bt_leaf'])
            self.pbt_rk_max = local_rk  # Different from self.bt_rk_max
            bt_object = self.pbt_object

        # Compute balltree by querying
        if in_r:
            # output is (ind, dist)
            self.bt_querytype = 'r'
            out_ind, out_dist = bt_object.query_radius(X_tar, local_rk,
                                                       return_distance=True)
        if in_k:
            # output is (dist, ind)
            self.bt_querytype = 'k'
            out_dist, out_ind = bt_object.query(X_tar, local_rk)

        # Store output
        if tar_all:
            self.bt_out = (out_ind, out_dist)
        if tar_pro:
            self.pbt_out = (out_ind, out_dist)

        # Record minmax;
        # k_minmax = smallest of local query index sizes
        # r_minmax = smallest of max local query distance
        all_k_max = [inds.size for inds in out_ind]
        all_r_max = [np.max(dists) for dists in out_dist]
        if tar_all:
            self.bt_k_minmax = np.min(all_k_max).astype(int)
            self.bt_r_minmax = np.min(all_r_max)
        if tar_pro:
            self.pbt_k_minmax = np.min(all_k_max).astype(int)
            self.pbt_r_minmax = np.min(all_r_max)
        # print(f'done.')

    def get_balltree(self, local_rk, mode, target, recompute=True):
        # Assertions
        assert mode in {'r', 'k'}, 'mode should be either "r" or "k".'
        in_r = mode == 'r'
        in_k = mode == 'k'
        assert target in {'all', 'probe'}, 'target should be either "all" or "probe"'
        tar_all = target == 'all'
        tar_pro = target == 'probe'
        bt_stored = 'bt_out' in self.__dict__
        pbt_stored = 'pbt_out' in self.__dict__
        assert (tar_all and bt_stored) or (tar_pro and pbt_stored), 'Call store_balltree() first.'
        if mode == 'k':
            assert isinstance(local_rk, int) or isinstance(local_rk, np.int64), 'Input local_k must be of int type'

        # Set minmax parameter based on mode and target (2x2 = 4 cases)
        if in_r:
            if tar_all:
                r_minmax = self.bt_r_minmax
            if tar_pro:
                r_minmax = self.pbt_r_minmax
        if in_k:
            if tar_all:
                k_minmax = self.bt_k_minmax
            if tar_pro:
                k_minmax = self.pbt_k_minmax

        # Check if stored balltree needs recomputing due to larger query radius or k
        if in_r and (local_rk > r_minmax):
            if recompute:
                print(f'Recomputing balltree...')
                self.store_balltree(local_rk, mode='r', target=target)
                # print(f'done computing balltree')
            else:
                raise Exception('Query radius bigger than stored. Call compute_balltree or get_balltree with recompute = True')
        if in_k and (local_rk > k_minmax):
            if recompute:
                print(f'Recomputing balltree...')
                self.store_balltree(local_rk, mode='k', target=target)
                # print(f'done computing balltree')
            else:
                raise Exception('Query k bigger than stored. Call compute_balltree or get_balltree with recompute = True')

        # Fetch fully computed balltree
        if tar_all:
            inds_full, dists_full = self.bt_out
        if tar_pro:
            inds_full, dists_full = self.pbt_out
        inds, dists = [], []
        for i in range(dists_full.shape[0]):
            # Depending on mode, form a different filter
            if in_r:
                filt = dists_full[i] <= local_rk
                dists_now = dists_full[i][filt]
                inds_now = inds_full[i][filt]
            if in_k:
                # bt_filt = bt_inds_full[i] <= local_rk
                dists_now = dists_full[i][:local_rk]
                inds_now = inds_full[i][:local_rk]

            dists.append(dists_now)
            inds.append(inds_now)
        if in_r:
            inds = np.array(inds, dtype='object')
            dists = np.array(dists, dtype='object')
        if in_k:
            inds = np.array(inds)
            dists = np.array(dists)
        return (inds, dists)

    """
    Fit without scoring
    """

    def fit_one(self, hp, compute_balltree=False):
        """
        A global runthrough of the sample, with one set of hyperparameters
        """

        # Input processing
        X = self.X
        inds_probe = self.inds_probe
        pca_thr = hp['pca_thr']
        ker_alpha = hp['ker_alpha']
        assert 'mmd_data' in self.__dict__, 'MMD data not computed, call gen_mmd_data().'
        # Accept exactly one of the r and k inputs.
        in_r = 'r' in hp.keys()
        in_k = 'k' in hp.keys()
        assert in_r + in_k == 1, "Exactly one of (k, r) should be specified"

        hpname_string = ', '.join(hp.keys())  # e.g. "r, pca_thr"
        hpval_string = ', '.join(list(np.round(list(hp.values()), 3).astype(str)))  # e.g. "0.15, 0.95". Rounded to 3 digits
        print(f'Running ({hpname_string}) = ({hpval_string})')

        if in_r:
            local_r = hp['r']
        if in_k:
            local_k = hp['k']

        # Balltree
        if in_r:
            if compute_balltree:
                self.store_balltree(local_r, mode='r', target='all')
                self.store_balltree(local_r, mode='r', target='probe')
            near_inds, near_dists = self.get_balltree(local_r, mode='r', target='all')
        if in_k:
            if compute_balltree:
                self.store_balltree(local_k, mode='k', target='all')
                self.store_balltree(local_k, mode='k', target='probe')
            near_inds, near_dists = self.get_balltree(local_k, mode='k', target='all')

        # Restrict to inds_probe
        near_inds_probe = near_inds[inds_probe]
        near_dists_probe = near_dists[inds_probe]

        # Main loop
        pvals = np.zeros(inds_probe.size)
        mmds = np.zeros(inds_probe.size)
        dims = np.zeros(inds_probe.size)
        for i, inds in enumerate(near_inds_probe):
            inds0 = inds.astype(int)  # Same values, making sure integer indices will be used
            if in_k:
                # For neighborhood rescaling
                local_r = near_dists_probe[i][local_k - 1]

            # Extract local neighborhood
            ii = inds_probe[i]
            Y = (X[inds0] - X[ii]) / local_r  # Rescaled neighborhood of X[ii]
            if Y.shape[0] > 1:
                jloc_out = uniformity(Y, pca_thr, ker_alpha,
                                      self.settings['mmd_expr_cutoff'], self.mmd_data, max_mult=self.settings['max_mult'])
            else:  # Trivial output
                jloc_out = {"pval": -1, "mmd": -1, "dim": -1}
            # Record outputs
            pvals[i], mmds[i], dims[i] = jloc_out['pval'], jloc_out['mmd'], jloc_out['dim']
        output = {'pval': pvals, 'mmd': mmds, 'dim': dims}

        hpname_string = ', '.join(hp.keys())  # e.g. "r, pca_thr"
        hpval_string = ', '.join(list(np.round(list(hp.values()), 3).astype(str)))  # e.g. "0.15, 0.95". Rounded to 3 digits
        print(f'done with ({hpname_string}) = ({hpval_string})')
        return output

    def fit_many(self, hp):
        """
        Multiple hyperparameters
        """
        in_r = np.all([('r' in hp_now.keys()) for hp_now in hp])
        in_k = np.all([('k' in hp_now.keys()) for hp_now in hp])
        assert in_r + in_k == 1, 'Input hyperparameters should all either specify r or k'

        # Call store_balltree once for largest of r or k parameters.
        print(f'Computing balltree for the current batch of hyperparameters...')
        if in_r:
            max_r = np.max([hp_now['r'] for hp_now in hp])
            self.store_balltree(max_r, mode='r', target='all')
            self.store_balltree(max_r, mode='r', target='probe')
        if in_k:
            max_k = np.max([hp_now['k'] for hp_now in hp])
            self.store_balltree(max_k, mode='k', target='all')
            self.store_balltree(max_k, mode='k', target='probe')

        if self.settings['multithread']:
            print('Starting a multi-thread run.')
            n_cores = self.settings['n_cores']
            if n_cores == -1:
                n_cores = os.cpu_count()
            pool = ThreadPool(n_cores)
            output = pool.map(self.fit_one, hp)
            pool.close()
        else:
            print('Starting a single-thread run.')
            output = [self.fit_one(hp_now) for hp_now in hp]
        return output

    """
    Score parameters obtained from fitting
    """

    def filt_pval(self, pvals, n_bins):
        """
        Find a cutoff point for the input p-values after taking -log(p)
        Return two things: binary filters for singularity and the knee values
        """
        logp = -np.log(pvals[pvals > 0])
        epdf = compute_kde(logp, n_bins)
        knee = compute_pdf_knee(*epdf, S=self.hhp['filt_knee_S'])['knee_val']

        filt = pvals < np.exp(-knee)
        output = {'filt': filt, 'knee': knee}
        return output

    def filt_output(self, output_noscore):
        """
        Run filt_pval as a loop over the output dict variable
        Return two things: binary filters for singularity and the knee values
        """
        filt_all = []
        knee_all = []
        for output_now in output_noscore:
            pval_now = output_now['pval']
            output_now = self.filt_pval(pval_now, n_bins=self.hhp['filt_n_steps'])
            filt_all.append(output_now['filt'])
            knee_all.append(output_now['knee'])
        output = {'filt': filt_all, 'knee': knee_all}
        return output

    def dauc(self, filt, local_rk, mode):
        """
        Compute directional AUC (DAUC) using the sample,
        filtered location of singularities, and another radius parameter.
        """
        assert mode in {'r', 'k'}
        X = self.X
        inds_probe = self.inds_probe
        Xp = X[inds_probe]
        if not np.any(filt):
            return -1

        # Retrieve balltree and filter
        bt_inds, bt_dists = self.get_balltree(local_rk=local_rk, mode=mode, target='probe')
        inds_filt = np.nonzero(filt)[0]

        auc_all = np.zeros(inds_filt.size)
        for ii, i in enumerate(inds_filt):
            inds_now = bt_inds[i].astype(int)  # Nearby indices
            filt_now = filt[inds_now]  # Binary label on singular or not, on nearby indices
            Y = Xp[inds_now] - Xp[i]  # X[i] is centre

            # Direction towards singularities
            q = filt_now.reshape(1, -1) @ Y  # Direction to singularities
            q = q.reshape(-1)

            # If nearby points are all smooth
            if (not np.any(q)) or (not np.any(filt_now)):
                auc = 0
            # If nearby points are all singular
            elif not np.any(1 - filt_now):
                auc = 1
            else:
                q = q / np.linalg.norm(q)  # Normalise
                y = Y @ q.reshape(-1, 1)
                y = y.reshape(-1)  # Y projected to q
                auc = roc_auc_score(filt_now, y)
            auc_all[ii] = auc
        output = auc_all
        return output

    def score_output(self, output_noscore, hp, test_manifold):
        """
        Run dauc as a loop over the output dict variable
        """
        # assert neighbor in {'r', 'k'}
        print(f'Filtering and scoring output...')
        # 1. Filter singular points
        filt_out = self.filt_output(output_noscore)
        filts = filt_out['filt']
        knees = filt_out['knee']
        output = []
        for i, output_now in enumerate(output_noscore):
            neighbor = list(set(hp[i].keys()).intersection({'r', 'k'}))
            assert len(neighbor) == 1
            neighbor = neighbor[0]

            # 2. Small-Unifornity P-value Concentration (SUPC)
            smt_out = compute_supc(output_now['pval'])
            mh_verdict = smt_out['verdict']  # Manifold hypothesis
            print(f'SUPC Peak value = {smt_out["peak"]}')
            if mh_verdict:
                print(f'MH True at hp={hp[i]}')

            # 3. Directional AUC
            if neighbor == 'r':
                local_rk = hp[i]['r']
                local_rk2 = local_rk * self.hhp['dauc_r_mult']
            if neighbor == 'k':
                local_rk = hp[i]['k']
                local_rk2 = local_rk * self.hhp['dauc_r_mult']
                local_rk2 = np.round(local_rk2).astype(int)
            filt_now = filts[i]
            dauc_now = self.dauc(filt_now, local_rk2, mode=neighbor)
            prop_now = np.mean(filt_now)

            # 4. Purity
            bt_inds2, bt_dists2 = self.get_balltree(local_rk=local_rk2, mode=neighbor, target='probe')
            pur_now = [np.mean(filt_now[item.astype(int)]) for item in bt_inds2]
            pur_now = np.array(pur_now)[filt_now]

            # 5. Score
            score_now = score_fn(dauc_now, pur_now, prop_now, self.hhp['score_regcoef'])
            if test_manifold:
                score_now *= (1 - mh_verdict)  # If manifold hypothesis holds, set score = 0 (=best score).

            output_per = {'dauc': dauc_now,
                          'score': score_now,
                          'filt': filts[i],
                          'knee': knees[i],
                          'purity': pur_now,
                          'mh_verdict': mh_verdict,
                          'prop': prop_now}
            output.append(output_per)
        print('done.')
        return output

    """
    Main functions for users. fit, predict, etc.
    """

    def fit(self, X, probe_prop=None, probe_inds=None, hp=None,
            neighbor='r', freeze=[], test_manifold=False, verbose=False):
        """
        Main fit method

        Internal variables:
        hps_bounds_initial: dict with key = hp names, value = tuple (lower bound, upper bound)
        hps_bounds_final: dict, same format as hps_bounds_initial. Specifies final bounds
        antifreeze: list of string. entry = hp names that are not frozen
        hps_lower: list of float. entry = lower bound for searched hp grid in current iteration, ordered same as antifreeze
        hps_upper: list of float. same as hps_lower

        Example:
        hps_bounds_initial = {'r': (0.1, 0.15), ...}
        hps_bounds_final = {'r': (0.03, 0.4), ...}
        antifreeze = ['r', 'pca_thr']
        hps_lower = [0.1, 0.8]
        hps_upper = [0.2, 0.95]
        """

        # Assertions
        self.X = X  # Sample
        N = X.shape[0]
        assert neighbor in {'r', 'k'}
        in_inds = probe_inds is not None
        in_prop = probe_prop is not None
        assert in_inds + in_prop < 2, 'Only one of inds_probe and prop should be specified'
        self.inds_probe_isall = False
        if (not in_inds) and (not in_prop):  # Default if neither are supplied
            inds_probe = np.arange(N)
            self.inds_probe_isall = True
        elif in_prop:
            assert isinstance(probe_prop, float)
            assert (0 < probe_prop) and (probe_prop <= 1)
            if probe_prop == 1:
                inds_probe = np.arange(N)
                self.inds_probe_isall = True
            else:
                N2 = int(N * probe_prop)
                inds_probe = np.random.choice(N, N2, replace=False)
        self.inds_probe = inds_probe
        # Print toggle
        old_stdout = sys.stdout
        print_toggle(verbose, old_stdout)
        tic = time.monotonic()

        # Case 1. If hyperparameters are supplied
        if hp is not None:
            output_now = self.fit_many(hp)
            score_now = self.score_output(output_now, hp, test_manifold)
            best_ind = np.argmin([item['score'] for item in score_now])
            best_hp = hp[best_ind]
            best_output = output_now[best_ind]
            best_score_dict = score_now[best_ind]

        # Case 2. Automatic hyperparameter search when hyperparameters are not supplied
        else:
            # Hyperparameter bounds. Partition hps_names = freeze U antifreeze
            hps_names = ['pca_thr', 'ker_alpha']
            hps_names.insert(0, neighbor)  # e.g. hps_names = ['r', 'pca_thr', 'ker_alpha']
            antifreeze = list(set(hps_names).difference(set(freeze)))
            self.compute_dim_est()
            dim_est = self.X_dim_est  # Crudely estimated intrinsic dimension
            # Update values of self.hhp indexed by 'bounds_initial', 'bounds_final'
            # by adding tuples indexed by 'r' or 'k' into self.hhp['bounds_initial'] and self.hhp['bounds_final']
            self.compute_hp_bounds(neighbor)
            # Frozen hyperparameters specified in the freeze variable
            # dict indexed by strings in freeze, each value is a real number
            frozen_value_all = []
            for hp_name in freeze:
                item = self.hhp['bounds_initial'][hp_name]  # item = (lower_bound, upper_bound)
                if hp_name == 'r':
                    frozen_value = power_add(item[0], item[1], p=dim_est, mean=True)  # Power mean, power = dim_est
                elif hp_name == 'k':
                    frozen_value = int(np.mean(item))
                else:
                    frozen_value = np.mean(item)
                frozen_value_all.append(frozen_value)

            # Case 2-1. If all hyperparameter values are frozen, execute directly
            if len(antifreeze) == 0:
                hp = [{freeze[i]: frozen_value_all[i] for i in range(3)}]
                output_now = self.fit_many(hp)  # Using fit_many for uniformity of execution through method
                # score_now = self.score_output(output_now, hp, neighbor, test_manifold)
                score_now = self.score_output(output_now, hp, test_manifold)
                best_ind = 0  # best_ind = np.argmin([item['score'] for item in score_now])
                best_hp = hp[best_ind]
                best_output = output_now[best_ind]
                best_score_dict = score_now[best_ind]

            # Case 2-2. When there is at least one non-frozen hyperparameter, run a hyperparameter search loop.
            else:
                # Initial set of hyperparameters
                # Everything except hps_lower, hps_upper are re-used
                res_all = [self.hhp['grid_resolution'][key] for key in antifreeze]
                power_all = [1 + (dim_est - 1) * (key == 'r') for key in antifreeze]  # Dimension scaling for r
                hps_lower = [self.hhp['bounds_initial'][key][0] for key in antifreeze]
                hps_upper = [self.hhp['bounds_initial'][key][1] for key in antifreeze]
                hpgrid_dict = self.make_hpgrid(res_all, power_all, hps_lower, hps_upper, freeze, antifreeze, frozen_value_all)

                # Hyperparameter search loop
                # List of pairs (hps_lower, hps_upper) examined so far, to prevent re-visiting
                breadcrumb = (hps_lower.copy(), hps_upper.copy())
                breadcrumb_trail = [breadcrumb]
                while True:
                    # Step 1. Main calculation
                    hpgrid_keys = list(hpgrid_dict.keys())  # Integer grid coordinates for hyperparameter tuples
                    hpgrid_values = list(hpgrid_dict.values())  # List of dicts
                    output_now = self.fit_many(hpgrid_values)
                    # score_now = self.score_output(output_now, hpgrid_values, neighbor, test_manifold)
                    score_now = self.score_output(output_now, hpgrid_values, test_manifold)

                    # Step 2. Retrieve best output
                    best_ind = np.argmin([item['score'] for item in score_now])
                    best_key = hpgrid_keys[best_ind]  # Internal usage within this WHILE loop
                    best_hp = hpgrid_values[best_ind]
                    best_output = output_now[best_ind]
                    best_score_dict = score_now[best_ind]

                    # Step 3. Identify whether best set of hyperparameter was found at an extreme end
                    # BREAK when the best set of hyperparameter was found at the interior of the grid.
                    # This For loop creates priority of hyperparameters by design
                    expand_direction = 0  # Element of {-1, 0, +1}
                    for i, ii in enumerate(best_key):
                        ii_max = res_all[i] - 1
                        if ii == 0:
                            expand_direction = -1
                            break
                        elif ii == ii_max:
                            expand_direction = +1
                            break
                    if expand_direction == 0:
                        break

                    # Step 4. Else expand search range and return to the While loop.
                    # BREAK if reached end of search range.
                    # At this point, i is a fixed index determined from Step 3.
                    hp_name_changed = antifreeze[i]  # Which hp_name to change the search range for
                    final_lower, final_upper = self.hhp['bounds_final'][hp_name_changed]
                    if hp_name_changed == 'r':
                        pwr = dim_est
                    else:
                        pwr = 1
                    leap_mult = 1  # Fixed at 1 for now
                    leap_length = power_subtract(hps_upper[i], hps_lower[i], p=pwr)
                    if expand_direction == +1:
                        print(f'Optimal hyperparameters found at the largest {hp_name_changed}. Searching larger {hp_name_changed}.')
                        if hps_upper[i] == final_upper:
                            print(f'HPS stops: Reached end of search range.')
                            break  # End of search range
                        hps_lower[i] = hps_upper[i]
                        hps_upper[i] = power_add(hps_upper[i], leap_mult * leap_length, p=pwr)  # Power scaling
                        hps_upper[i] = min(final_upper, hps_upper[i])
                    elif expand_direction == -1:
                        print(f'Optimal hyperparameters found at the smallest {hp_name_changed}. Searching smaller {hp_name_changed}.')
                        if hps_lower[i] == final_lower:
                            print(f'HPS stops: Reached end of search range.')
                            break  # End of search range
                        hps_upper[i] = hps_lower[i]
                        hps_lower[i] = power_subtract(hps_lower[i], leap_mult * leap_length, p=pwr)  # Power scaling
                        hps_lower[i] = max(final_lower, hps_lower[i])

                    # Step 5. Set next set of hyperparameters (hpgrid_dict)
                    # Hyperparameter value for 'k' should be integer
                    if hp_name_changed == 'k':
                        hps_lower[i] = int(hps_lower[i])
                        hps_upper[i] = int(hps_upper[i])
                    # res_all, power_all are re-used from the initial hyperparameter grid
                    hpgrid_dict = self.make_hpgrid(res_all, power_all, hps_lower, hps_upper, freeze, antifreeze, frozen_value_all)
                    breadcrumb = (hps_lower.copy(), hps_upper.copy())
                    for i in range(len(antifreeze)):
                        assert breadcrumb[0][i] <= breadcrumb[1][i]
                        if breadcrumb[0][i] == breadcrumb[1][i]:
                            print(f'HPS stops: Next search range is trivial.')
                            break

                    # Step 6. Check for whether the next hyperparameter set overlaps as before (breadcrumb_trail)
                    # Check by looking for all parameter ranges having an overlap
                    if not self.hhp['expand_hps']:
                        # Option to not expand the range of hyperparameter search
                        print(f'Not expanding hyperparameter search loop.')
                        break
                    any_trail_overlap = False
                    for breadcrumb_before in breadcrumb_trail:
                        trail_overlap = True
                        for i in range(len(antifreeze)):
                            assert breadcrumb[0][i] < breadcrumb[1][i]
                            assert breadcrumb_before[0][i] < breadcrumb_before[1][i]

                            cond1 = breadcrumb[1][i] <= breadcrumb_before[0][i]
                            cond2 = breadcrumb_before[1][i] <= breadcrumb[0][i]
                            if cond1 or cond2:  # No intersection
                                trail_overlap = False
                        if trail_overlap:
                            any_trail_overlap = True
                            break
                    if any_trail_overlap:
                        print(f'HPS stops: Next search range was searched before.')
                        break
                    else:
                        print(f'Expanding hyperparameter search range.\n')
                        breadcrumb_trail.append(breadcrumb)

        # Save best learned parameters obtained from the above main WHILE loop
        self.params_hp = best_hp
        self.params = {**best_output, **best_score_dict}  # Combine the output itself and the scores

        # Output of manifold hypothesis testing
        smt_out = compute_supc(self.params['pval'])
        self.supc_output = smt_out
        if smt_out['verdict']:
            print(f'The data seems to be sampled from a manifold!')

        # Time taken, Toggle print, Return
        toc = time.monotonic()
        time_taken = toc - tic
        self.fit_time = time_taken
        print(f'Time taken = {round(time_taken, 2)}')
        print_toggle(True, old_stdout)
        # Return the classifier object itself
        return self

    def manifold_hypothesis(self, return_peak=False):
        smt_out = self.supc_output
        if not return_peak:
            return smt_out['verdict']
        else:
            output = {'verdict': smt_out['verdict'], 'peak': smt_out['peak']}
            return output

    def predict(self, Y, smooth_nbd=5, use_manifold_hypothesis=True):
        """
        Predict labels based on nearest-neighbor search of the computed p-values.
        We take the geometric mean of the p-values and check if exceedds the threshold (knee)
        """
        if use_manifold_hypothesis and self.manifold_hypothesis():
            filt = np.full(Y.shape[0], False)
        else:
            bt_dists, bt_inds = self.pbt_object.query(Y, smooth_nbd)
            pval = self.params['pval']
            logp = -np.log(pval)
            logp_query = logp[bt_inds]
            logp_query = np.mean(logp_query, axis=1)
            knee = self.params['knee']
            filt = logp_query > knee
        return filt

    def fit_predict(self, X, *args, **kwargs):
        self.fit(X, *args, **kwargs)
        return self.predict(X)

    def score_samples(self, Y, smooth_nbd=5):
        bt_dists, bt_inds = self.pbt_object.query(Y, smooth_nbd)
        pval = self.params['pval']
        logp = -np.log(pval)
        logp_query = logp[bt_inds]
        logp_query = np.mean(logp_query, axis=1)
        return logp_query

    def get_params(self):
        return self.params
    

def judge(X, search_auto=['r'], search_range=None, search_list=None, search_res=None, witness_prop=1.0, verbose=False):
    """
    One function for easy usage

    Order of precedence: search_auto < search_range < search_list
    search_auto: a list of strings, a subset of ['r', 'k', 't', 'a'].
    search_range: a dict with keys = ['r', 'k', 't', 'a']. The values are tuples (lower, upper).
    search_list: a list of dicts, where each dict has keys = ['r', 'k', 't', 'a'].
    Each time, only one of 'r' and 'k' may be specified.
    witness_prop: float between 0 and 1. Proportion of points to be used as witnesses.
    verbose: bool. Whether to print progress.
    """

    def hpname_convert(hpname):
        # Convert hyperparameter names to internal names
        if hpname == 'r':
            return 'r'
        elif hpname == 'k':
            return 'k'
        elif hpname == 't':
            return 'pca_thr'
        elif hpname == 'a':
            return 'ker_alpha'
        
    def hplist_convert(hp_list):
        # Convert hyperparameter names to internal names
        hp_out = [hpname_convert(item) for item in hp_list]
        return hp_out

    def hpdict_convert(hp_dict):
        # Convert hyperparameter names to internal names
        hp_out = {hpname_convert(key): hp_dict[key] for key in hp_dict.keys()}
        return hp_out

    if search_res is None:
        # Default search grid resolution for each hyperparameter
        search_res = {'r': 5, 'k': 5, 't': 3, 'a': 3}

    clf = Hades()
    src_dir = clf.settings['src_dir']
    # Priority: search_list > search_range > search_auto
    if search_list is not None:
        # Mode 1. Hyperparameter search using search_list
        for item in search_list:
            if 'a' in item.keys():
                clf.settings['mmd_filename'] = f'{src_dir}/mmd_data_big'

        search_list = [hpdict_convert(item) for item in search_list]
        clf.fit(X, hp=search_list, probe_prop=witness_prop, verbose=verbose)

    elif search_range is not None:
        # Mode 2. Hyperparameter search using search_range
        if 'a' in search_range.keys():
            clf.settings['mmd_filename'] = f'{src_dir}/mmd_data_big'

        search_1dgrids = {key: np.linspace(search_range[key][0], search_range[key][1], search_res[key]) for key in search_range.keys()}
        neighbor = {'r', 'k'}.intersection(search_range.keys())
        assert len(neighbor) == 1, 'Must specify exactly one of "r" or "k" in search_range.'
        neighbor = list(neighbor)[0]

        if 't' not in search_range.keys():
            search_1dgrids['t'] = np.array([0.9])
        if 'a' not in search_range.keys():
            search_1dgrids['a'] = np.array([0.5])

        if neighbor == 'r':
            search_list = hp_grid(r=search_1dgrids['r'], 
                                  pca_thr=search_1dgrids['t'], 
                                  ker_alpha=search_1dgrids['a'])
        elif neighbor == 'k':
            search_list = hp_grid(k=search_1dgrids['k'], 
                                  pca_thr=search_1dgrids['t'], 
                                  ker_alpha=search_1dgrids['a'])
        search_1dgrids = hpdict_convert(search_1dgrids)

        # Main run
        clf.fit(X, hp=search_list, probe_prop=witness_prop, verbose=verbose)

    elif search_auto is not None:
        # Mode 3. Automatic hyperparameter search using search_auto
        # Process inputs
        search_auto_set = set(search_auto)
        if 'a' in search_auto_set:
            clf.settings['mmd_filename'] = f'{src_dir}/mmd_data_big'
        assert search_auto_set.issubset({'r', 'k', 't', 'a'}), 'search_auto must be a subset of {"r", "k", "t", "a"}.'
        neighbor = {'r', 'k'}.intersection(search_auto_set)
        assert len(neighbor) < 2, 'Cannot specify both "r" and "k" in search_auto.'
        if len(neighbor) == 1:
            neighbor = list(neighbor)[0]
        else:
            neighbor = 'r'
        all_hp_names = {'t', 'a'}.union({neighbor})
        freeze_pre = list(all_hp_names.difference(search_auto_set))
        freeze = hplist_convert(freeze_pre)
        search_res_convert = hpdict_convert(search_res)

        # Main run
        clf.hhp['grid_resolution'] = search_res_convert
        clf.fit(X, freeze=freeze, neighbor=neighbor, probe_prop=witness_prop, verbose=verbose)

    # Compute output
    y = clf.predict(X)
    s = clf.score_samples(X)
    dim = clf.params['dim']
    best_hp = clf.params_hp
    output = {'label': y, 'score': s, 'dim': dim, 'best_hp': best_hp}
    return output


def test_run():
    X = gen.two_circles(5000)
    clf = Hades()
    clf.fit(X, verbose=True, freeze=['r', 'pca_thr', 'ker_alpha'])
    s = clf.score_samples(X)
    misc.plot(X, c=s)
    plt.show()


if __name__ == '__main__':
    test_run()
