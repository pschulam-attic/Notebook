'''Functions for estimating an adjustment to the posterior prediction
over subtypes when making predictions online.

Author: Peter Schulam

'''

import numpy as np
import logging
from scipy.optimize import minimize
from scipy.misc import logsumexp


class OnlineAdjustment:
    def __init__(self, model, penalty, seed=0):
        self.model = model
        self.penalty = penalty
        self.seed = seed

    def fit(self, training_data, w0=None, **options):
        self.objective = OnlineLoss(training_data, self.model, self.penalty)
        f = self.objective.value
        g = self.objective.gradient

        if w0 is None:
            num_feat = self.objective.encoder.num_features
            w0 = np.ones(num_feat)
            #w0 = np.random.RandomState(self.seed).normal(size=num_feat)
            
        self.solution = minimize(f, w0, jac=g, method='BFGS', options=options)
        self.w = self.solution.x

        return self

    def proba(self, histories):
        p = [self.objective.engine.run(X, self.w) for X in histories]
        return np.array(p)

    def log_proba(self, histories):
        p = self.proba(histories)
        return np.log(p)


class OnlineLoss:
    def __init__(self, training_data, model, penalty):
        self.training_data = training_data
        self.model = model
        self.penalty = penalty
        self.encoder = OnlineFeatureEncoder(model)
        self.engine = InferenceEngine(self.encoder)
        
    def value(self, w):
        v = 0.0
        n = 0

        for X, y in self.training_data:
            if len(y[1][0][0]) < 1:
                continue

            lp = np.log(self.engine.run(X, w))
            ll = self.model.likelihood(*y[1][0])
            v -= logsumexp(lp + ll)
            n += 1

        v /= n
        v += self.penalty / 2.0 * np.dot(w, w)

        logging.info('f(w) = {:.06f}'.format(v))

        return v

    def gradient(self, w):
        g = np.zeros_like(w)
        n = 0

        for X, y in self.training_data:
            if len(y[1][0][0]) < 1:
                continue

            lp = np.log(self.engine.run(X, w))
            ll = self.model.likelihood(*y[1][0])
            lj = lp + ll
            wt = np.exp(lj - logsumexp(lj))

            logging.debug('Predicted {}'.format(np.round(np.exp(lp), 2)))
            logging.debug('Posterior {}'.format(np.round(wt, 2)))

            f_exp = self.encoder.expected_encoding(np.exp(lp), X)
            g_i = f_exp
            for z, _ in enumerate(wt):
                f_obs = self.encoder.encode(z, X)
                g_i -= wt[z] * f_obs
                # g += wt[z] * (f_exp - f_obs)

            logging.debug('Gradient {}'.format(np.round(g_i, 2)))

            g += g_i
            n += 1

        g /= n
        g += self.penalty * w

        logging.info('||g(w)||_inf = {:.06f}'.format(g.max()))

        return g


class OnlineFeatureEncoder:
    def __init__(self, model):
        self.num_subtypes = model.num_subtypes
        self.model = model

    @property
    def num_features(self):
        # return 2
        return 2 * self.num_subtypes

    @property
    def num_outputs(self):
        return self.num_subtypes

    # def encode(self, z, history):
    #     'Single weight encoding.'
    #     d = history[0]
    #     lp = self.model.prior(*d)
    #     ll = self.model.likelihood(*d)

    #     f = np.zeros(2)
    #     f[0] = lp[z]
    #     f[1] = ll[z]

    #     return f

    def encode(self, z, history):
        'Likelihood ratio encoding.'
        d = history[0]
        lp = self.model.prior(*d)
        ll = self.model.likelihood(*d)

        lpr = lp - lp[0]
        llr = ll - ll[0]

        k = self.num_subtypes
        f = np.zeros(2 * k)
        f[z] = lpr[z]
        f[k + z] = llr[z]

        return f

    # def encode(self, z, history):
    #     d = history[0]
    #     lp = self.model.prior(*d)
    #     ll = self.model.likelihood(*d)

    #     k = self.num_subtypes
    #     f = np.zeros(2 * k)
    #     f[z] = lp[z]
    #     f[k + z] = ll[z]

    #     return f

    # def expected_encoding(self, pz, history):
    #     'Single weight encoding.'
    #     d = history[0]
    #     lp = self.model.prior(*d)
    #     ll = self.model.likelihood(*d)

    #     f = np.zeros(2)
    #     f[0] = (pz * lp).sum()
    #     f[1] = (pz * ll).sum()

    #     return f

    def expected_encoding(self, pz, history):
        'Likelihood ratio encoding.'
        d = history[0]
        lp = self.model.prior(*d)
        ll = self.model.likelihood(*d)

        lpr = lp - lp[0]
        llr = ll - ll[0]

        k = self.num_subtypes
        f = np.zeros(2 * k)
        f[:k] = pz * lpr
        f[k:] = pz * llr

        return f

    # def expected_encoding(self, pz, history):
    #     d = history[0]
    #     lp = self.model.prior(*d)
    #     ll = self.model.likelihood(*d)

    #     k = self.num_subtypes
    #     f = np.zeros(2 * k)
    #     f[:k] = pz * lp
    #     f[k:] = pz * ll

    #     return f


class InferenceEngine:
    def __init__(self, encoder):
        self.encoder = encoder

    def run(self, history, w):
        s = np.zeros(self.encoder.num_outputs)
        for z, _ in enumerate(s):
            f = self.encoder.encode(z, history)
            s[z] = np.dot(f, w)
        p = np.exp(s - logsumexp(s))
        
        return p
