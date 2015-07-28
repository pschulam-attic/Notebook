import numpy as np
import itertools
import logging
from scipy.optimize import minimize
from scipy.misc import logsumexp


class SubtypeModel:
    def __init__(self, penalty, models, seed=0, **options):
        self.penalty = penalty
        self.models = models
        self.seed = seed
        self.objective = None
        self.solution = None
        self.options = options

    def fit(self, training_data):
        self.objective = ModelObjective(training_data, self.penalty, self.models, self.seed)
        w0 = self.objective.initial_weights()
        self.solution = minimize(self.objective.value, w0,
                                 jac=self.objective.gradient, method='BFGS',
                                 options=self.options)
        
        self.objective.weights.set_weights(self.solution.x)

    def proba(self, histories):
        junct_trees = [self.objective.engine.run(h) for h in histories]
        return np.array(jt[0][0] for jt in junct_trees)

    def log_proba(self, histories):
        P = self.proba(histories)
        return np.log(P)

    def predict(self, histories):
        P = self.proba(histories)
        return np.argmax(P, axis=1)


class ModelObjective:
    def __init__(self, training_data, penalty, models, seed=0):
        self.training_data = training_data
        self.penalty = penalty
        self.weights = ModelWeights([m.num_subtypes for m in models])
        self.scorer = Scorer(models, self.weights)
        self.engine = InferenceEngine(self.scorer)

    def initial_weights(self):
        return self.weights.collapsed()

    def value(self, w):
        self.weights.set_weights(w)

        v = 0.0
        n = 0

        for X, y in self.training_data:
            jt = self.engine.run(X)
            lp = np.log(jt[0][0][y])
            n += 1            
            v -= lp

        v /= n
        v += self.penalty / 2.0 * np.dot(w, w)

        logging.info('Evaluated objective: f(w) = {:.08f}'.format(v))
            
        return v

    def gradient(self, w):
        self.weights.set_weights(w)

        g = np.zeros_like(w)
        n = 0

        for X, y in self.training_data:
            jt = self.engine.run(X)
            jtc = self.engine.observe_target(jt, y)
            
            fe = feature_expectations(jt)
            fec = feature_expectations(jtc)

            n += 1
            g += fe - fec  # We are computing the *negative* of the LL.

        g /= n
        g += self.penalty * w

        logging.info('Evaluated gradient: ||g(w)||_inf = {:.08f}'.format(g.max()))
        
        return g


def feature_expectations(junction_tree):
    s, p = junction_tree
    p = [p_i.ravel() for p_i in p[1:]]
    return np.concatenate(s + p)


class InferenceEngine:
    def __init__(self, scorer):
        self.scorer = scorer

    def run(self, data):
        'Return a junction tree.'

        # Initialize singleton clusters
        singletons = self.scorer.data_scores(data)
        for i, s in enumerate(singletons):
            s += self.scorer.singleton_scores(i)

        # Initialize pair clusters
        pairs = []
        for i, _ in enumerate(singletons):
            pairs.append(self.scorer.pairwise_scores(i))

        # Pass messages to root (target singleton)
        for i, s in enumerate(singletons[1:], 1):
            pairs[i] += s
            singletons[0] += marginalize(pairs[i], [0], logsumexp)

        # Normalize root (no further messages received)
        singletons[0] -= logsumexp(singletons[0])

        # Pass messages from root to leaves
        for i, p in enumerate(pairs[1:], 1):
            p += colvec(singletons[0] - marginalize(p, [0], logsumexp))
            singletons[i] += marginalize(p, [1], logsumexp) - singletons[i]

        # Normalize non-root clusters
        for s in singletons[1:]:
            s -= logsumexp(s)

        for p in pairs[1:]:
            p -= logsumexp(p)

        # Move from log-space to prob-space
        singletons = [np.exp(s) for s in singletons]
        pairs = [None] + [np.exp(p) for p in pairs[1:]]

        return singletons, pairs

    def observe_target(self, junction_tree, target_subtype):
        singletons, pairs = junction_tree
        singletons = [np.log(s) for s in singletons]
        pairs = [None] + [np.log(p) for p in pairs[1:]]

        singletons[0][:] = -np.inf
        singletons[0][target_subtype] = 0

        # Propagate information down
        for s, p in zip(singletons[1:], pairs[1:]):
            p += colvec(singletons[0])
            s[:] = marginalize(p, [1], logsumexp)

        # Renormalize beliefs
        for s, p in zip(singletons[1:], pairs[1:]):
            s -= logsumexp(s)
            p -= logsumexp(p)

        singletons = [np.exp(s) for s in singletons]
        pairs = [None] + [np.exp(p) for p in pairs[1:]]

        return singletons, pairs


class Scorer:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        self.num_subtypes = weights.num_subtypes

    @property
    def parameters(self):
        return self.weights.collapsed()

    def data_scores(self, data):
        'Return log-likelihood scores for all marker-subtype combos.'
        scores = []
        for m, d in zip(self.models, data):
            ll = m.likelihood(*d)
            scores.append(ll)

        return scores

    def singleton_scores(self, ix):
        'Return log-linear score for singleton features of marker `ix`.'
        return self.weights.singleton(ix).copy()

    def pairwise_scores(self, ix):
        'Return log-linear score for pairwise features of marker `ix`.'
        s = self.weights.pairwise(ix)
        if s is None:
            return s
        else:
            return s.copy()


class ModelWeights:
    def __init__(self, num_subtypes, seed=0):
        self.num_subtypes = num_subtypes
        self._initialize(seed)

    def _initialize(self, seed):
        rnd = np.random.RandomState(seed)
        
        self.singleton_ = [rnd.normal(size=n) for n in self.num_subtypes]
        
        n_targ, *n_aux = self.num_subtypes
        self.pairwise_ = [rnd.normal(size=(n_targ, n)) for n in n_aux]
        self.pairwise_ = [None] + self.pairwise_

    def singleton(self, ix):
        return self.singleton_[ix]

    def pairwise(self, ix):
        return self.pairwise_[ix]

    def collapsed(self):
        w1 = self.singleton_
        w2 = [w.ravel() for w in self.pairwise_[1:]]
        return np.concatenate(w1 + w2)

    def set_weights(self, flat_weights):
        singleton_ = []
        singleton_offset = 0
        
        pairwise_ = []
        pairwise_offset = sum(self.num_subtypes)

        for i, k in enumerate(self.num_subtypes):
            # Read singleton weights
            i1 = singleton_offset
            i2 = i1 + k
            w1 = flat_weights[i1:i2].copy()
            singleton_.append(w1)
            singleton_offset += k

            # Read pairwise weights
            if i == 0:
                w2 = None
            else:
                k0 = self.num_subtypes[0]
                i1 = pairwise_offset
                i2 = i1 + (k0 * k)
                w2 = flat_weights[i1:i2].copy().reshape((k0, k))
                pairwise_offset += (k0 * k)

            pairwise_.append(w2)

        self.singleton_ = singleton_
        self.pairwise_ = pairwise_


def marginalize(x, remove, func=np.sum):
    axes = tuple(range(x.ndim))
    over = set(axes) - set(remove)
    over = tuple(sorted(list(over)))
    return func(x, axis=over)


def normalize(x):
    return x / np.sum(x)


def rowvec(x):
    x = x.ravel()
    return x[np.newaxis, :]


def colvec(x):
    x = x.ravel()
    return x[:, np.newaxis]


# class ConditionalModel:
#     def __init__(self, models):
#         self.models = models
#         self.num_subtypes = [m.num_subtypes for m in models]

#     @property
#     def num_features(self):
#         n = 0
#         n += sum(self.num_subtypes)
#         for k in self.num_subtypes[1:]:
#             n += self.num_subtypes[0] * k
#         return n

#     def log_proba(self, data, weights):
#         scores = np.zeros(self.num_subtypes)
#         for subtypes in subtypes_iterator(self.num_subtypes):
#             scores[subtypes] = self.score(data, subtypes, weights)
#         return scores - logsumexp(scores)

#     def marg_log_proba(self, data, weights):
#         lp = self.log_proba(data, weights)
#         lm = marginalize(lp, [0], logsumexp)
#         return lm

#     def score(self, data, subtypes, weights):
#         s1 = self.history_score(data, subtypes)
#         s2 = self.subtypes_score(subtypes, weights)
#         return s1 + s2

#     def history_score(self, data, subtypes):
#         ll = 0.0
#         for m, d, z in zip(self.models, data, subtypes):
#             ll += m.likelihood(*d)[z]

#         return ll

#     def subtypes_score(self, subtypes, weights):
#         features = self.subtypes_features(subtypes)
#         return np.dot(features, weights)

#     def subtypes_features(self, subtypes):
#         specs = [(z, k) for z, k in zip(subtypes, self.num_subtypes)]
#         target_spec, *auxiliary_specs = specs

#         singleton = [singleton_features(s) for s in specs]
#         pairwise = [pairwise_features(target_spec, a) for a in auxiliary_specs]
#         features = np.concatenate(singleton + pairwise)

#         return features

#     def feature_expectations(self, data, weights):
#         lp = self.log_proba(data, weights)
        
#         singleton_expectations = []
#         for i, _ in enumerate(self.num_subtypes):
#             lp_i = marginalize(lp, [i], logsumexp)
#             singleton_expectations.append(np.exp(lp_i))

#         pairwise_expectations = []
#         for i, _ in enumerate(self.num_subtypes[1:], 1):
#             lp_i = marginalize(lp, [0, i], logsumexp)
#             pairwise_expectations.append(np.exp(lp_i).ravel())

#         expectations = np.concatenate(singleton_expectations + pairwise_expectations)
#         return expectations

#     def conditional_expectations(self, data, target_subtype, weights):
#         lp = self.log_proba(data, weights)
#         lm = marginalize(lp, [0], logsumexp)
#         lc = lp[target_subtype] - lm[target_subtype]
#         lc = lc[np.newaxis, ...]

#         singleton_expectations = []
#         for i, k in enumerate(self.num_subtypes):
#             if i == 0:
#                 singleton_expectations.append(singleton_features((target_subtype, k)))
#             else:
#                 lc_i = marginalize(lc, [i], logsumexp)
#                 singleton_expectations.append(np.exp(lc_i).ravel())

#         pairwise_expectations = []
#         k0 = self.num_subtypes[0]
#         for i, k in enumerate(self.num_subtypes[1:], 1):
#             fp_i = np.zeros((k0, k))
#             fp_i[target_subtype, :] = singleton_expectations[i]
#             pairwise_expectations.append(fp_i.ravel())

#         expectations = np.concatenate(singleton_expectations + pairwise_expectations)
#         return expectations

#     def initial_weights(self, seed=0):
#         rnd = np.random.RandomState(seed)
#         weights = rnd.normal(size=self.num_features)
#         return weights


# def subtypes_iterator(num_subtypes):
#     iterables = [range(k) for k in num_subtypes]
#     return itertools.product(*iterables)




# def evidence_score(likelihoods, subtypes):
#     return sum(ll[z] for ll, z in zip(likelihoods, subtypes))


# def history_likelihood(model, data):
#     return model.likelihood(*data)


# def configuration_score(subtype_specs, weights):
#     target, *auxiliary = subtype_specs
#     features = configuration_features(target, auxiliary)


# def configuration_features(target_spec, auxiliary_specs):
#     singleton = [singleton_features(s) for s in [target_spec] + auxiliary_specs]
#     pairwise = [pairwise_features(target_spec, aux) for aux in auxiliary_specs]
#     features = np.concatenate(singleton + pairwise)
#     return features


# def singleton_features(subtype_spec):
#     z, k = subtype_spec
#     f = np.zeros(k)
#     f[z] = 1
#     return f


# def pairwise_features(target_spec, auxiliary_spec):
#     f1 = singleton_features(target_spec)
#     f2 = singleton_features(auxiliary_spec)
#     f = colvec(f1) * rowvec(f2)
#     return f.ravel()


# def score(evidence_score, configuration_score):
#     return evidence_score + configuration_score




# class ConditionalModel:
#     def __init__(self, target, auxiliary, **models):
#         self.target = target
#         self.auxiliary = auxiliary
#         self.models = models
#         self.num_subtypes = [models[target].num_subtypes]
#         self.num_subtypes += [models[a].num_subtypes for a in auxiliary]

#     def score(self, subtypes, patient_data, w):
#         s = 0.0

#         # Compute likelihood contributions to score.
#         for n in [self.target] + self.auxiliary:
#             m = self.models[n]
#             d = patient_data[n].unpack()
#             s += m.likelihood(*d)[subtypes[n]]

#         # Compute the compatibility contribution.
#         f = self.features(subtypes)
#         s += np.dot(f, w)

#         return s

#     def proba(self, z, patient_data, w):
#         pass

#     def complete_proba(self):
#         pass

#     def features(self, subtypes):
#         targ = subtypes[self.target]
#         aux = [subtypes[a] for a in self.auxiliary]
#         k_targ, *k_aux = self.num_subtypes

#         singletons = [feat_onehot(x, k) for x, k in zip(subtypes, self.num_subtypes)]
#         pairs = [feat_pair(targ, k_targ, x, k) for x, k in zip(aux, k_aux)]

#         return np.concatenate(singletons + pairs)


# def feat_onehot(x, k):
#     f = np.zeros(k)
#     f[x] = 1
#     return f


# def feat_pair(x1, k1, x2, k2):
#     f1 = feat_onehot(x1, k1)
#     f2 = feat_onehot(x2, k2)
#     f = colvec(f1) * rowvec(f2)
#     return f.ravel()


