import random
import numpy as np
from scipy.io import loadmat
from . import misc
from .misc import bounding_box_size


def _sphere(N, dim, ambient_dim=None, **kwargs):
    if ambient_dim is None:
        ambient_dim = dim + 1
    output = np.random.multivariate_normal(mean=np.zeros(dim + 1), cov=np.identity(dim + 1), size=N)
    output = output / np.linalg.norm(output, axis=1).reshape(-1, 1)
    output = np.hstack((output, np.zeros((N, int(ambient_dim - (dim + 1))))))
    return output


def _disk(N, dim, ambient_dim=None, **kwargs):
    if ambient_dim is None:
        ambient_dim = dim
    Y = _sphere(N, dim - 1, ambient_dim=dim)
    radii = np.power(np.random.random(N), 1 / dim)
    output = radii.reshape(-1, 1) * Y
    output = np.hstack((output, np.zeros((N, int(ambient_dim - dim)))))
    return output


def gen_helper(func):
    """
    Decorator for adding scale, noise, permutation
    """
    def out_func(*args, **kwargs):
        X0 = func(*args, **kwargs)
        N, D = X0.shape

        # scale, noise, permute
        if 'scale' not in kwargs.keys():
            scale = 1
        else:
            scale = kwargs['scale']

        if 'force_scale' not in kwargs.keys():
            force_scale = True
        else:
            force_scale = kwargs['force_scale']

        if 'noise' not in kwargs.keys():
            noise = 0
        else:
            noise = kwargs['noise']

        if 'permute' not in kwargs.keys():
            permute = True
        else:
            permute = kwargs['permute']

        if force_scale:
            current_scale = bounding_box_size(X0)
            total_scale = scale / current_scale
        else:
            total_scale = scale

        X = total_scale * X0 + noise * _disk(N, D)
        if permute:
            inds_perm = np.random.choice(N, N, replace=False)
            X = X[inds_perm]
        return X

    return out_func


@gen_helper
def sphere(*args, **kwargs):
    return _sphere(*args, **kwargs)


@gen_helper
def disk(*args, **kwargs):
    return _disk(*args, **kwargs)


@gen_helper
def ellipsoid(N, dim, stretches, **kwargs):
    X = _sphere(N, dim)
    assert len(stretches) == dim+1
    for i in range(dim+1):
        X.T[i] *= stretches[i]
    return X


# Cartesian product of spheres
@gen_helper
def sphere_prod(N, dims, **kwargs):
    X_proj = [_sphere(N, d) for d in dims]
    X = np.hstack(X_proj)
    return X


@gen_helper
def sphere_1d(N, **kwargs):
    return _sphere(N, dim=1)


@gen_helper
def sphere_2d(N, **kwargs):
    return _sphere(N, dim=2)


@gen_helper
def ellipse_simple(N, stretches=[1.5,1], **kwargs):
    return ellipsoid(N, dim=1, stretches=stretches)


@gen_helper
def ellipsoid_simple(N, stretches=[1.5,1,1], **kwargs):
    return ellipsoid(N, dim=2, stretches=stretches)


@gen_helper
def torus_3d(N, r1=1, r2=0.5, **kwargs):
    T1 = np.random.random(N)
    T2 = np.random.random(N)
    Y = np.hstack([T1.reshape(-1, 1), T2.reshape(-1, 1)])
    Y *= (2 * np.pi)
    X = []
    for y in Y:
        t1, t2 = y
        pt_x = (r1 + r2 * np.cos(t2)) * np.cos(t1)
        pt_y = (r1 + r2 * np.cos(t2)) * np.sin(t1)
        pt_z = r2 * np.sin(t2)
        pt = np.array([pt_x, pt_y, pt_z])
        X.append(pt)
    X = np.array(X)
    return X


@gen_helper
def paraboloid(N, **kwargs):
    base = _disk(N, 2)
    height = np.array([(pt[0] ** 2 - pt[1] ** 2) for pt in base]).reshape(-1, 1)
    X = np.hstack((base, height))
    return X


@gen_helper
def two_squares(N, **kwargs):
    N2 = int(N / 7)

    X = []
    ones = np.ones(N2)
    ones2 = np.ones(2 * N2)

    X.append(np.vstack([0 * ones, 0.5 * _disk(N2, 1).T]).T)
    for i in [-1, 1]:
        X.append(np.vstack([i * ones, 0.5 * _disk(N2, 1).T]).T)
        X.append(np.vstack([_disk(2 * N2, 1).T, 0.5 * i * ones2]).T)
    X = np.vstack(X)
    return X


@gen_helper
def two_circles(N, **kwargs):
    N2 = int(N / 2)

    X1 = _sphere(N2, 1) + np.array([0.5, 0])
    X2 = _sphere(N2, 1) + np.array([-0.5, 0])
    X = np.vstack([X1, X2])
    return X


@gen_helper
def tri_spoke(N, **kwargs):
    N2 = int(N / (2 * np.pi + 3))
    N3 = int(N * (2 * np.pi) / (2 * np.pi + 3))

    X_circ = _sphere(N3, 1)
    cos120 = -0.5
    sin120 = np.sin((2 / 3) * np.pi)
    rotmat1 = np.array([[cos120, sin120], [-sin120, cos120]])
    rotmat2 = np.array([[cos120, -sin120], [sin120, cos120]])
    X_prong0 = np.array([0.5, 0]) + 0.5 * _disk(N2, 1, 2)
    X_prong1 = (np.array([0.5, 0]) + 0.5 * _disk(N2, 1, 2)) @ rotmat1.T
    X_prong2 = (np.array([0.5, 0]) + 0.5 * _disk(N2, 1, 2)) @ rotmat2.T
    X = np.vstack([X_circ, X_prong0, X_prong1, X_prong2])
    X = X[:, [1, 0]]  # Flip x and y
    return X


@gen_helper
def venus(N, **kwargs):
    N2 = int(N * 5 / 7)
    plus_scale = 0.8
    X1 = np.hstack([np.zeros(int(N2 / 5)).reshape(-1, 1), plus_scale * _disk(int(N2 / 5), 1)]) + np.array([0, -(1 + plus_scale)])
    X2 = np.hstack([plus_scale * _disk(int(N2 / 5), 1), np.zeros(int(N2 / 5)).reshape(-1, 1)]) + np.array([0, -(1 + plus_scale)])
    X3 = _disk(N2, 2)
    X = np.vstack([X1, X2, X3])
    return X


@gen_helper
def box(N, **kwargs):
    N2 = int(N / 6)
    X = []
    ones = np.ones(N2)
    for i in range(2):
        X.append(np.vstack([i * ones, np.random.random(N2), np.random.random(N2)]).T)
        X.append(np.vstack([np.random.random(N2), i * ones, np.random.random(N2)]).T)
        X.append(np.vstack([np.random.random(N2), np.random.random(N2), i * ones]).T)
    X = np.vstack(X)
    return X


@gen_helper
def two_spheres(N, d=2, **kwargs):
    N2 = int(N / 2)
    shift1 = np.zeros(d+1)
    shift2 = np.zeros(d+1)
    shift1[0] = 0.5
    shift2[0] = -0.5

    X1 = _sphere(N2, d) + shift1
    X2 = _sphere(N2, d) + shift2
    X = np.vstack([X1, X2])
    return X


@gen_helper
def cone(N, **kwargs):
    N2 = int(N / 2)
    cone1 = _disk(N2, 2, 3)
    cone2 = _disk(N2, 2, 3)
    cone1.T[2] = np.linalg.norm(cone1, axis=1)
    cone2.T[2] = -np.linalg.norm(cone2, axis=1)
    X = np.vstack([cone1, cone2])
    return X


@gen_helper
def saturn(N, **kwargs):
    N2 = int(N / 2)
    X1 = 1.6 * _disk(N2, 2, 3)
    X2 = _sphere(N2, 2)
    X = np.vstack([X1, X2])
    return X


@gen_helper
def three_disks(N, **kwargs):
    N2 = int(N / 3)
    X1_pre = _disk(N2, 2)
    X2_pre = _disk(N2, 2)
    X3_pre = _disk(N2, 2)
    zeros = np.zeros(N2).reshape(-1, 1)
    X1 = np.hstack([zeros, X1_pre])
    X2 = np.hstack([X2_pre[:, 0].reshape(-1, 1),
                    zeros,
                    X2_pre[:, 1].reshape(-1, 1)])
    X3 = np.hstack([X3_pre, zeros])
    X = np.vstack([X1, X2, X3])
    return X


@gen_helper
def saddle_skewer(N, **kwargs):
    N2 = int(N * 0.8)
    N3 = int(N * 0.2)

    X1 = paraboloid(N2)
    zeros = np.zeros(N3).reshape(-1, 1)
    X2 = 1.2 * _disk(N3, 1).reshape(-1, 1)
    X2 = np.hstack([zeros, zeros, X2])

    X = np.vstack([X1, X2])
    return X


@gen_helper
def pinch_torus(N, r1=1, r2=1, better_3d_view=True, **kwargs):
    # Pinched torus
    # theta_res = int(np.sqrt(N))
    # phi_res = int(N / theta_res)

    X = []
    Y = np.hstack([np.random.random(N).reshape(-1,1),np.random.random(N).reshape(-1,1)])
    Y *= (2 * np.pi)
    for y in Y:
        theta, phi = y
        len_handle = r2 * np.sin(theta / 2)
        len_spoke = r1 + r2 * np.sin(theta / 2)
        pt_spoke = len_spoke * np.array([np.cos(theta), np.sin(theta)])
        # for phi in np.linspace(0, 2 * np.pi, phi_res):

        stretch = (len_spoke + len_handle * np.cos(phi)) / len_spoke
        pt_x = stretch * pt_spoke[0]
        pt_z = stretch * pt_spoke[1]
        pt_y = r2 * np.sin(phi) * np.sin(theta / 2)
        X.append(np.array([pt_x, pt_y, pt_z]))
    X = np.array(X)

    if better_3d_view:
        t_x, t_y, t_z = 0, 90, 225
        X = misc.rotate_3d(X, t_x=t_x, t_y=t_y, t_z=t_z)

    return X


@gen_helper
def gm_torus(N, r1=1, r2=0.5, prop_torus=0.7, disk_regular=True, **kwargs):
    # Pinched torus
    N_torus = int(N * prop_torus)
    theta_res = int(np.sqrt(N_torus))
    phi_res = int(N_torus / theta_res)
    X_torus = []
    for theta in np.linspace(0, 2 * np.pi, theta_res + 1, endpoint=False):
        if theta > 0:
            len_handle = r2 * np.sin(theta / 2)
            len_spoke = r1 + r2 * np.sin(theta / 2)
            pt_spoke = len_spoke * np.array([np.cos(theta), np.sin(theta)])
            for phi in np.linspace(0, 2 * np.pi, phi_res):
                stretch = (len_spoke + len_handle * np.cos(phi)) / len_spoke
                pt_x = stretch * pt_spoke[0]
                pt_z = stretch * pt_spoke[1]
                pt_y = r2 * np.sin(phi) * np.sin(theta / 2)
                X_torus.append(np.array([pt_x, pt_y, pt_z]))
    X_torus = np.array(X_torus)

    # Disk
    N_disk = int(N * (1 - prop_torus))
    N_sq = int(N_disk * 4 / np.pi)
    if disk_regular:
        disk_res1 = int(np.sqrt(N_sq))
        disk_res2 = int(N_sq / disk_res1)
        square_pts = np.meshgrid(np.linspace(-1, 1, disk_res1), np.linspace(-1, 1, disk_res2), [0])
        square_pts = np.vstack([square_pts[1].reshape(-1),
                                square_pts[2].reshape(-1),
                                square_pts[0].reshape(-1)]).T
        X_disk = square_pts[np.nonzero(np.linalg.norm(square_pts, axis=1) < 1)[0]]
    else:
        ones = r1 * np.zeros(N_disk).reshape(-1, 1)
        X_disk = r1 * _disk(N_disk, 2)
        X_disk_x = X_disk.T[0].reshape(-1, 1)
        X_disk_y = X_disk.T[1].reshape(-1, 1)
        X_disk = np.hstack([X_disk_x, ones, X_disk_y])

    X = np.vstack([X_torus, X_disk])
    return X


@gen_helper
def henneberg_load(N=5456, rot_angles=(0,0,35), data_dir='data', **kwargs):
    X = loadmat(data_dir + '/henneberg.mat')['X'].T
    X /= 10
    # t_x, t_y, t_z = 20, 255, 28
    t_x, t_y, t_z = rot_angles
    X = misc.rotate_3d(X, t_x=t_x, t_y=t_y, t_z=t_z)
    return X


@gen_helper
def henneberg(N, beta_fun=None, rot_angles=(0,0,35), **kwargs):
    X1, X2 = np.random.random(N), np.random.random(N)

    if beta_fun is None:
        beta_fun = lambda x: np.power(x, 2)
    beta_all = 0.4 + 0.2 * beta_fun(X1)
    phi_all = 2 * np.pi * X2

    x1 = 2 * (np.power(beta_all, 2) - 1) * np.cos(phi_all) / beta_all
    x2 = 2 * (np.power(beta_all, 6) - 1) * np.cos(3 * phi_all) / (3 * np.power(beta_all, 3))
    x = x1 - x2
    y1 = 6 * (np.power(beta_all, 2) * (np.power(beta_all, 2) - 1)) * np.sin(phi_all)
    y2 = 2 * (np.power(beta_all, 6) - 1) * np.sin(3 * phi_all)
    y = -(y1 + y2) / (3 * np.power(beta_all, 3))
    z = 2 * (np.power(beta_all, 4) + 1) * np.cos(2 * phi_all) / np.power(beta_all, 2)

    X = np.vstack([x, y, z]).T
    X *= 0.1

    t_x, t_y, t_z = rot_angles
    X = misc.rotate_3d(X, t_x=t_x, t_y=t_y, t_z=t_z)
    return X
    

# def henneberg(N, noise=0, shuffle=True, mode='load', data_dir='data', better_3d = True):
#     if mode == 'load':
#         X = loadmat(data_dir + '/henneberg.mat')['X'].T
#         X /= 10
#
#         t_x, t_y, t_z = 20, 255, 28
#         X = misc.rotate_3d(X, t_x=t_x, t_y=t_y, t_z=t_z)
#
#     elif mode == 'generate':
#         rand1 = np.random.random(N)
#         rand2 = np.random.random(N)
#
#         beta_all = 0.4 + 0.2 * rand1
#         phi_all = 2 * np.pi * rand2
#
#         x1 = 2 * (np.power(beta_all, 2) - 1) * np.cos(phi_all) / beta_all
#         x2 = 2 * (np.power(beta_all, 6) - 1) * np.cos(3 * phi_all) / (3 * np.power(beta_all, 3))
#         x = x1 - x2
#         y1 = 6 * (np.power(beta_all, 2) * (np.power(beta_all, 2) - 1)) * np.sin(phi_all)
#         y2 = 2 * (np.power(beta_all, 6) - 1) * np.sin(3 * np.sin(phi_all))
#         y = -(y1 + y2) / (3 * np.power(beta_all, 3))
#         z = 2 * (np.power(beta_all, 4) + 1) * np.cos(2 * phi_all) / np.power(beta_all, 2)
#
#         X = np.vstack([x, y, z]).T
#
#         X /= 10
#         # Rotate dataset in xy by 30 degrees
#         angle_now = np.pi * (30 / 180)
#         cos_now = np.cos(angle_now)
#         sin_now = np.sin(angle_now)
#         rotmat = np.array([[cos_now, sin_now, 0], [-sin_now, cos_now, 0], [0, 0, 1]])
#         X = X @ rotmat
#
#     X += noise * disk(*X.shape)
#     if shuffle:
#         inds_shuffle = np.random.choice(X.shape[0], X.shape[0], replace=False)
#         X = X[inds_shuffle]
#
#     if better_3d:
#
#     return X


def orth(method, N, axis_spans, ambient_dim=None, sizes=None, r=1, shuffle=True):
    '''
  Generates a point cloud sampled near orthogonally interlocking samples
  Using spans of axes
  '''

    if ambient_dim == None:
        ambient_dim = np.max([np.max(axes_span) for axes_span in axis_spans]) + 1

    num_disks = len(axis_spans)
    if sizes is None:
        sizes = int(N / num_disks) * np.ones(num_disks)
        residue = int(N % num_disks)
        if residue > 0:
            lastbits = np.concatenate((np.ones(residue), np.zeros(num_disks - residue)))
            random.shuffle(lastbits)
            sizes += lastbits

    samples = np.zeros((N, ambient_dim))
    sizes_cum = 0
    for i, axes_span in enumerate(axis_spans):
        size = int(sizes[i])
        ord_axes = list(axes_span)
        ord_axes.sort()

        if method == "sphere":
            sample_pure_unleaved = globals()[method](size, len(axes_span) - 1, r=r)
        else:
            sample_pure_unleaved = globals()[method](size, len(axes_span), r=r)

        sample_tuple = (np.zeros((size, ord_axes[0])),)
        for j in np.arange(1, len(ord_axes)):
            sample_tuple = (*sample_tuple, sample_pure_unleaved[:, j - 1].reshape(-1, 1),
                            np.zeros((size, ord_axes[j] - ord_axes[j - 1] - 1)))
        sample_tuple = (*sample_tuple, sample_pure_unleaved[:, -1].reshape(-1, 1),
                        np.zeros((size, ambient_dim - ord_axes[-1] - 1)))
        sample_pure = np.hstack(sample_tuple)
        samples[sizes_cum: sizes_cum + size] = sample_pure  # random_rotation(sample_pure, bound=tilt)
        sizes_cum += size

    if shuffle:
        np.random.shuffle(samples)
    return samples
