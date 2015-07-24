import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from mypy.models import softmax


def fit_adjustment(P, Q, X, M):
    k  = P.shape[1]
    d  = X.shape[1]
    W0 = np.zeros((k, d))
    C  = loglik_ratio(Q)

    def f(w):
        W = w.reshape(W0.shape)
        y = [softmax.regression_ll(x, y, W, c) for x, y, c in zip(X, P, C)]
        return -sum(y)

    def g(w):
        W = w.reshape(W0.shape)
        y = [softmax.regression_ll_grad(x, y, W, c) for x, y, c in zip(X, P, C)]
        y = -sum(y)
        y[~M] = 0.0
        return y.ravel()

    s = minimize(f, W0.ravel(), jac=g, method='BFGS')
    if not s.success:
        raise RuntimeError('Optimization did not terminate successfully.')
    
    W = s.x.reshape(W0.shape)

    return W


def fit_adjustment2(P, X):
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
        raise RuntimeError('Optimization did not terminate successfully')

    W = s.x.reshape(W0.shape)

    return W


def choose_features(P, Q, alpha=0.05):
    k1      = P.shape[1]
    k2      = Q.shape[1]
    pvalues = np.ones((k1, k2))
    z       = np.argmax(P, axis=1)

    for i in range(1, k1):
        p1, s1 = estimate_probs(Q[z == 0])
        p2, s2 = estimate_probs(Q[z == i])
        d  = p1 - p2
        s  = np.sqrt(s1 ** 2 + s2 ** 2)
        pvalues[i] = 2 * (1 - st.norm.cdf(np.abs(d / s)))

    return pvalues <= alpha


def make_adjustment(W, Q, X):
    C = loglik_ratio(Q)
    P = np.array([softmax.regression_proba(x, W, c) for x, c in zip(X, C)])
    return P


def make_adjustment2(W, X):
    P = np.array([softmax.regression_proba(x, W) for x in X])
    return P
    

def loglik_ratio(Q):
    llr = np.log(Q) - np.log(Q[:, 0][:, np.newaxis])
    return llr


def xentropy(P, Q=None):
    if Q is None:
        Q = P
    return - np.sum(P * np.log(Q))


def estimate_probs(Q):
    n = Q.sum()
    p = Q.sum(axis=0) / n
    v = p * (1 - p)
    s = np.sqrt(v / n)
    return p, s
