'''Functions for estimating an adjustment to the posterior prediction
over subtypes when making predictions online.

Author: Peter Schulam

'''

import numpy as np

from scipy.optimize import minimize
from sklearn.cross_validation import KFold
from mypy.models import softmax


def train_adjustment(P, Q, QQaux, n_dev=4):
    n    = P.shape[0]
    Pmap = map_encode(P)
    Qmap = map_encode(Q)

    XXaux = [np.c_[np.ones(n), Qi[:, 1:]] for Qi in QQaux]

    QQdev = [np.zeros_like(Qmap) for _ in XXaux]
    dev_folds = KFold(n, n_dev, shuffle=True, random_state=0)
    for i, (train, test) in enumerate(dev_folds):
        for j, Xj in enumerate(XXaux):
            Wj = fit_multinomial(Pmap[train], Xj[train])
            QQdev[j][test] = predict_multinomial(Wj, Xj[test])

    for i, _ in enumerate(QQdev):
        QQdev[i] = map_encode(QQdev[i])

    weights = interpolate(P, [Qmap] + QQdev)

    WWaux = []
    for i, Xi in enumerate(XXaux):
        Wi = fit_multinomial(Pmap, Xi)
        WWaux.append(Wi)

    return WWaux, weights


def apply_adjustment(Q, QQaux, WWaux, weights):
    n    = Q.shape[0]
    Qmap = map_encode(Q)

    XXaux = [np.c_[np.ones(n), Qi[:, 1:]] for Qi in QQaux]
    QQmap = [predict_multinomial(Wi, Xi) for Wi, Xi in zip(WWaux, XXaux)]
    QQmap = [map_encode(Qi) for Qi in QQmap]

    Qhat = interp_mixture(weights, [Qmap] + QQmap)

    return Qhat


def fit_multinomial(P, X):
    k = P.shape[1]
    d = X.shape[1]
    W0 = np.zeros((k, d))

    def f(w):
        W = w.reshape(W0.shape)
        y = [softmax.regression_ll(x, y, W) for x, y in zip(X, P)]
        return -sum(y)

    def g(w):
        W = w.reshape(W0.shape)
        y = [softmax.regression_ll_grad(x, y, W) for x, y in zip(X, P)]
        y = -sum(y)
        return y.ravel()

    s = minimize(f, W0.ravel(), jac=g, method='BFGS')

    if not s.success:
        raise RuntimeError('Multinomial fit optimization failed.')

    W = s.x.reshape(W0.shape)

    return W


def predict_multinomial(W, X):
    P = np.array([softmax.regression_proba(x, W) for x in X])
    return P    


def interpolate(P, QQ, seed=1):
    '''Estimate interpolation weights of distributions in QQ to minimize
    cross-entropy under P.

    Author: Peter Schulam

    '''
    rnd = np.random.RandomState(seed)
    
    M    = len(QQ)
    v    = rnd.normal(size=M)
    v[0] = 0.0
    
    obj = lambda x: interp_perplexity(P, QQ, x)
    jac = lambda x: interp_perplexity_jac(P, QQ, x)
    sol = minimize(obj, v, jac=jac, method='BFGS')
    
    if not sol.success:
        raise RuntimeError('Interpolation optimization failed.')
    
    w = softmax.softmax_func(sol.x)

    return w


def interp_perplexity(P, QQ, v):
    '''Compute cross-entropy of interpolated distributions under P.

    Author: Peter Schulam

    '''
    w = softmax.softmax_func(v)
    Q = interp_mixture(w, QQ)
    return - np.sum(P * np.log(Q)) / P.shape[0]


def interp_perplexity_jac(P, QQ, v):
    '''Compute the jacobian of the cross-entropy with respect to `v`.

    Author: Peter Schulam

    '''
    M = v.size
    w = softmax.softmax_func(v)
    Q = interp_mixture(w, QQ)
    
    dp_dw = np.zeros(M)
    for m in range(M):
        dp_dw[m] = np.sum(P * QQ[m] / Q)
        
    dw_dv = -softmax.softmax_grad(v)
    
    return np.dot(dp_dw, dw_dv) / P.shape[0]


def interp_mixture(w, QQ):
    '''Compute the interpolated distribution.

    Author: Peter Schulam

    '''
    Q = np.zeros_like(QQ[0])
    for wi, Qi in zip(w, QQ):
        Q += wi * Qi
    return Q


def map_encode(P, eps=1e-2):
    '''Compute a new distribution in each row that is dominated by the MAP
    in the original.

    Author: Peter Schulam

    '''
    cx = np.argmax(P, axis=1)
    rx = list(range(P.shape[0]))
    Q  = eps * np.ones_like(P)
    Q[rx, cx] = 1
    Q /= Q.sum(axis=1)[:, np.newaxis]
    return Q
