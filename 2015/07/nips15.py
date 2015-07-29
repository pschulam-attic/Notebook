import numpy as np
import os

from scipy.stats import multivariate_normal
from scipy.misc import logsumexp

from mypy.bsplines import universal_basis
from mypy.models import softmax
from mypy.util import as_row, as_col


class NipsModel:
    def __init__(self, b, B, W, basis_param, kernel_param):
        self.b = b
        self.B = B
        self.W = W
        self.k = B.shape[0]
        
        self.basis_param = basis_param
        self.basis = universal_basis(*self.basis_param.values())
        self.kernel_param = kernel_param

    @classmethod
    def from_directory(cls, directory):
        param_files = ['basis.dat', 'kernel.dat', 'pop.dat', 'subpop.dat', 'marginal.dat']
        param_paths = [os.path.join(directory, f) for f in param_files]

        basis  = np.loadtxt(param_paths[0])
        bparam = BasisParam(tuple(basis[:2]), basis[2], basis[3])

        kernel = np.loadtxt(param_paths[1])
        kparam = KernelParam(*tuple(kernel))

        b = np.loadtxt(param_paths[2])
        B = np.loadtxt(param_paths[3])
        W = np.loadtxt(param_paths[4])
        W = np.r_[ np.zeros((1, W.shape[1])), W ]

        return cls(b, B, W, bparam, kparam)

    @property
    def num_subtypes(self):
        return self.k
        
    def phi1(self, x):
        return np.ones((x.size, 1))

    def phi2(self, x):
        return self.basis.eval(x)

    def covariance(self, x1, x2=None):
        return kernel(x1, x2, *self.kernel_param.values())

    def trajectory_means(self, t, x):
        from numpy import dot
        b, B = self.b, self.B

        P1 = self.phi1(t)
        P2 = self.phi2(t)

        m1 = dot(P1, dot(b, x)).ravel()
        m2 = dot(B, P2.T)

        return m1 + m2

    def trajectory_logl(self, t, x, y, z):
        if t.size < 1:
            return 0.0

        m = self.trajectory_means(t, x)[z]
        S = self.covariance(t)

        return multivariate_normal.logpdf(y, m, S)

    def prior(self, t, x1, x2, y):
        return softmax.regression_log_proba(x2, self.W)

    def likelihood(self, t, x1, x2, y):
        subtypes = range(self.k)
        return np.array([self.trajectory_logl(t, x1, y, z) for z in subtypes])

    def joint(self, t, x1, x2, y):
        prior = self.prior(t, x1, x2, y)
        likel = self.likelihood(t, x1, x2, y)
        return prior + likel

    def posterior(self, t, x1, x2, y):
        if len(t) == 0:
            return np.exp(self.prior(t, x1, x2, y))
        else:
            j = self.joint(t, x1, x2, y)
            return np.exp(j - logsumexp(j))

    def evidence(self, t, x1, x2, y):
        j = self.joint(t, x1, x2, y)
        return logsumexp(j)

    def predict(self, tnew, t, x1, x2, y):
        if len(t) == 0:
            Y = trajectory_means(tnew, x1)
            K = self.covariance(tnew)

        else:
            R = y - trajectory_means(t, x1)
            Y = trajectory_means(tnew, x1)
            K = None

            for i, r in enumerate(R):
                yhat, Khat = _gp_posterior(tnew, t, r, self.covariance)
                Y[i] += yhat
                K = Khat

        return Y, K


class BasisParam:
    def __init__(self, boundaries, degree, num_features):
        self.boundaries = boundaries
        self.degree = degree
        self.num_features = num_features

    def values(self):
        return self.boundaries, self.degree, self.num_features


class KernelParam:
    def __init__(self, a_const=1.0, a_ou=1.0, l_ou=1.0, a_noise=1.0):
        self.a_const = a_const
        self.a_ou = a_ou
        self.l_ou = l_ou
        self.a_noise = a_noise

    def values(self):
        return self.a_const, self.a_ou, self.l_ou, self.a_noise


class PatientData:
    def __init__(self, ptid, t, y, x1, x2):
        self.ptid = ptid
        self.t = np.array([]) if np.all(np.isnan(t)) else t.copy()
        self.y = np.array([]) if np.all(np.isnan(y)) else y.copy()
        self.x1 = x1
        self.x2 = x2

    @classmethod
    def from_tbl(cls, tbl, t, y, x1, x2):
        pd = {}
        pd['ptid'] = int(tbl['ptid'].values[0])
        pd['t'] = tbl[t].values
        pd['y'] = tbl[y].values
        pd['x1'] = np.asarray(tbl.loc[:, x1].drop_duplicates()).ravel()
        pd['x2'] = np.asarray(tbl.loc[:, x2].drop_duplicates()).ravel()
        pd['x2'] = np.r_[1.0, pd['x2']]
        return cls(**pd)

    def unpack(self):
        return self.t, self.x1, self.x2, self.y

    def truncate(self, censor_time, after=False):
        if after:
            obs = self.t > censor_time
        else:
            obs = self.t <= censor_time
        return self.__class__(self.ptid, self.t[obs], self.y[obs], self.x1, self.x2)

    
def kernel(x1, x2=None, a_const=1.0, a_ou=1.0, l_ou=1.0, a_noise=1.0):
    symmetric = x2 is None
    d = _differences(x1, x1) if symmetric else _differences(x1, x2)
    
    K = a_const * np.ones_like(d)
    K += _ou_kernel(d, a_ou, l_ou)
    
    if symmetric:
        K += a_noise * np.eye(x1.size)
        
    return K


def _ou_kernel(d, a, l):
    return a * np.exp( - np.abs(d) / l )


def _differences(x1, x2):
    return as_col(x1) - as_row(x2)


def _gp_posterior(tnew, t, y, kern):
    from numpy import dot
    from scipy.linalg import inv, solve

    K11 = kern(tnew)
    K12 = kern(tnew, t)
    K22 = kern(t)

    m = dot(K12, solve(K22, y))
    K = K11 - dot(K12, solve(K22, K12.T))

    return m, K
