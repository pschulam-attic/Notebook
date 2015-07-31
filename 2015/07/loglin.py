import numpy as np
import itertools
import logging
from scipy.optimize import minimize
from scipy.misc import logsumexp
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from owlqn import OWLQN


class SubtypeModel:
    def __init__(self, penalty, models, censor, regularizer='l2', seed=0, **options):
        self.penalty = penalty
        self.regularizer = regularizer
        self.models = models
        self.censor = censor
        self.merges = [merge_subtypes(m, censor) for m in models]
        self.seed = seed
        self.objective = None
        self.solution = None
        self.options = options

    def fit(self, training_data):
        self.objective = ModelObjective(training_data, self.penalty, self.regularizer, self.models, self.merges, self.seed)
        w0 = self.objective.initial_weights()
        f = self.objective.value
        g = self.objective.gradient

        if self.regularizer == 'l2':
            s = minimize(f, w0, jac=g, method='BFGS', options=self.options)
            self.solution = s
            self.objective.weights.set_weights(s.x)
            
        elif self.regularizer == 'l1':
            s = OWLQN(f, g, self.penalty, w0, **self.options).minimize()
            self.solution = s
            self.objective.weights.set_weights(s.x)

    def proba(self, histories):
        junct_trees = [self.objective.engine.run(h) for h in histories]
        return np.array([jt[1][0] for jt in junct_trees])

    def log_proba(self, histories):
        P = self.proba(histories)
        return np.log(P)

    def predict(self, histories):
        P = self.proba(histories)
        return np.argmax(P, axis=1)

    def predict_features(self, histories):
        junct_trees = [self.objective.engine.run(h) for h in histories]
        features = np.array([feature_expectations(jt) for jt in junct_trees])
        return features


def merge_subtypes(model, censor):
    b_censor = model.phi2(censor)
    num_bases = np.nonzero(b_censor)[0].max() + 1
    distances = pdist(model.B[:, :num_bases], 'euclidean')
    links = hierarchy.average(distances)
    return links[:, :2]


class ModelObjective:
    def __init__(self, training_data, penalty, regularizer, models, merges, seed=0):
        self.training_data = training_data
        self.penalty = penalty
        self.regularizer = regularizer
        self.encoder = FeatureEncoder([m.num_subtypes for m in models], merges)
        self.weights = ModelWeights(self.encoder)
        self.scorer = Scorer(models, self.weights, self.encoder)
        self.engine = InferenceEngine(self.scorer)
        self.rnd = np.random.RandomState(seed)

    def initial_weights(self):
        return self.weights.collapsed()

    def value(self, w):
        self.weights.set_weights(w)

        v = 0.0
        n = 0

        for X, y in self.training_data:
            if len(y[1][0][0]) < 1:
                continue
            jt = self.engine.run(X)
            lp = np.log(jt[1][0])
            ll = self.scorer.data_scores(y[1])[0]
            lj = lp + ll
            v -= logsumexp(lj)
            n += 1
            
            # lp = np.log(jt[1][0][y[0]])
            # n += 1
            # v -= lp

        v /= n

        if self.regularizer == 'l2':
            v += self.penalty / 2.0 * np.dot(w, w)
        elif self.regularizer == 'l1':
            v += self.penalty * np.abs(w).sum()
        else:
            raise RuntimError('Regularizer {} not supported'.format(self.regularizer))

        nz = (np.abs(w) > 0).sum()
        logging.info('Evaluated objective: f(w) = {:.08f}, ||w||_0 = {}'.format(v, nz))
            
        return v

    def gradient(self, w):
        self.weights.set_weights(w)

        #pdb.set_trace()

        g = np.zeros_like(w)
        n = 0

        for X, y in self.training_data:
            if len(y[1][0][0]) < 1:
                continue
            jt = self.engine.run(X)
            lp = np.log(jt[1][0])
            ll = self.scorer.data_scores(y[1])[0]
            lj = lp + ll
            wt = np.exp(lj - logsumexp(lj))

            fe = self.encoder.expected_encoding(jt)
            for i, _ in enumerate(wt):
                jtc = self.engine.observe_target(jt, i)
                fec = self.encoder.expected_encoding(jtc)
                g += wt[i] * (fe - fec)

            n += 1

        g /= n

        if self.regularizer == 'l2':
            g += self.penalty * w
        elif self.regularizer == 'l1':
            s = np.sign(w)
            
            is_zero = s == 0
            can_reduce = g < -self.penalty
            can_increase = g > self.penalty

            s[is_zero & can_reduce] =  1.0
            s[is_zero & can_increase] = -1.0

            g += self.penalty * s
        else:
            raise RuntimError('Regularizer {} not supported'.format(self.regularizer))
            

        logging.info('Evaluated gradient: ||g(w)||_inf = {:.08f}'.format(g.max()))
        
        return g


class FeatureEncoder:
    def __init__(self, num_subtypes, merges=None):
        self.num_subtypes = num_subtypes
        self.merges = merges
        self.hierarchical = merges is not None

    @property
    def num_single_features(self):
        h = self.hierarchical
        s = [self.encode_single(0, 0, False).size]
        s = s + [self.encode_single(0, i, h).size for i, _ in enumerate(self.num_subtypes[1:], 1)]
        return s

    @property
    def num_pair_features(self):
        p = [self.encode_pair(0, 0, i).size for i, _ in enumerate(self.num_subtypes[1:], 1)]
        return p

    @property
    def num_features(self):
        return self.num_single_features + self.num_pair_features

    def encode(self, subtypes):
        h = self.hierarchical
        s = [self.encode_single(z, i, h) for i, z in enumerate(subtypes)]
        p = [self.encode_pair(subtypes[0], z, i, h) for i, z in enumerate(subtypes[1:], 1)]
        return np.concatenate(s + p)

    def encode_single(self, z, index, hierarchical):
        k = self.num_subtypes[index]
        f = singleton_features((z, k))
        
        if self.hierarchical and hierarchical:
            m = self.merges[index]            
            f = single_to_hierarchical(f, m)

        return f

    def encode_pair(self, z_targ, z_aux, index_aux):
        f1 = self.encode_single(z_targ, 0, False)
        f2 = self.encode_single(z_aux, index_aux, True)
        f = colvec(f1) * rowvec(f2)
        f = f.ravel()
        return f

    def expected_encoding(self, junction_tree):
        _, ex_s, ex_p = junction_tree
        
        s = [e.copy() for e in ex_s]
        p = ex_p[:1] + [e.copy() for e in ex_p[1:]]
        
        if self.hierarchical:
            ex_s_m = zip(s[1:], self.merges[1:])
            s = s[:1] + [single_to_hierarchical(e, m) for e, m in ex_s_m]
        
            ex_p_m = zip(p[1:], self.merges[1:])
            p = p[:1] + [pair_to_hierarchical(e, m) for e, m in ex_p_m]

        p = [e.ravel() for e in p[1:]]

        return np.concatenate(s + p)


def features(subtypes, num_subtypes):
    specs = list(zip(subtypes, num_subtypes))
    targ_spec, *aux_specs = specs

    singleton = [singleton_features(s) for s in specs]
    pairwise = [pairwise_features(targ_spec, s) for s in aux_specs]

    return np.concatenate(singleton + pairwise)


def hierarchical_features(subtypes, num_subtypes, merges):
    specs = list(zip(subtypes, num_subtypes))
    targ_spec, *aux_specs = specs

    singleton = [singleton_features(s) for s in specs]
    singleton = singleton[:1] + [single_to_hierarchical(s, m) for s, m in zip(singleton[1:], merges[1:])]

    pairwise = []
    for s in singleton[1:]:
        f = colvec(singleton[0]) * rowvec(s)
        f = f.ravel()
        pairwise.append(f)

    return np.concatenate(singleton + pairwise)


def single_to_hierarchical(singleton, merges):
    k = singleton.size
    m = merges.shape[0] - 1
    f = np.zeros(k + m)
    f[:k] = singleton
    
    for i, merge in enumerate(merges[:-1]):
        g1, g2 = merge
        f[k + i] = f[g1] + f[g2]

        # Collapse merged groups
        f[g1] = 0.0
        f[g2] = 0.0

    return f[-2:]


def pair_to_hierarchical(pair, merges):
    k1, k2 = pair.shape
    m = merges.shape[0] - 1
    f = np.zeros((k1, k2 + m))
    f[:, :k2] = pair

    for i, merge in enumerate(merges[:-1]):
        g1, g2 = merge
        f[:, k2 + i] = f[:, g1] + f[:, g2]

        # Collapse merged groups
        f[:, g1] = 0.0
        f[:, g2] = 0.0

    return f[:, -2:]


def singleton_features(subtype_spec):
    z, k = subtype_spec
    f = np.zeros(k)
    f[z] = 1
    return f


def pairwise_features(target_spec, auxiliary_spec):
    f1 = singleton_features(target_spec)
    f2 = singleton_features(auxiliary_spec)
    f = colvec(f1) * rowvec(f2)
    return f.ravel()
    

def feature_expectations(junction_tree):
    _, s, p = junction_tree
    p = [p_i.ravel() for p_i in p[1:]]
    return np.concatenate(s + p)


def hierarchical_feature_expectations(junction_tree, merges):
    _, singles, pairs = junction_tree

    singleton = singles[:1] + [single_to_hierarchical(s, m) for s, m in zip(singles[1:], merges[1:])]

    pairwise = []
    for p, m in zip(pairs[1:], merges[1:]):
        k = p.shape[1]
        m = m.shape[0] - 1
        f = np.zeros((p.shape[0], k + m))
        f[:, :k] = p
        
        for i in enumerate(m[:-1]):
            g1 = m[i, 0]
            g2 = m[i, 1]
            f[:, k + i] = f[:, g1] + f[:, g2]

        f = f.ravel()
        pairwise.append(f)

    return np.concatenate(singleton + pairwise)


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
        log_partition = logsumexp(singletons[0])
        singletons[0] -= log_partition

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

        return log_partition, singletons, pairs

    def observe_target(self, junction_tree, target_subtype):
        _, singletons, pairs = junction_tree
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

        return None, singletons, pairs


class Scorer:
    def __init__(self, models, weights, encoder):
        self.models = models
        self.weights = weights
        self.encoder = encoder
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
        w = self.weights.singleton(ix)
        k = self.encoder.num_subtypes[ix]
        s = np.zeros(k)
        for z in range(k):
            f = self.encoder.encode_single(z, ix, ix > 0)
            s[z] = np.dot(w, f)

        return s

    def pairwise_scores(self, ix):
        'Return log-linear score for pairwise features of marker `ix`.'
        w = self.weights.pairwise(ix)
        
        if w is None:
            s = None
        else:
            k1 = self.encoder.num_subtypes[0]
            k2 = self.encoder.num_subtypes[ix]
            s = np.zeros((k1, k2))

            for z1 in range(k1):
                for z2 in range(k2):
                    f = self.encoder.encode_pair(z1, z2, ix)
                    s[z1, z2] = np.dot(w.ravel(), f)

        return s


class ModelWeights:
    def __init__(self, feature_encoder, seed=0):
        self.feature_encoder = feature_encoder
        self.num_subtypes = feature_encoder.num_subtypes
        self._initialize(seed)

    def _initialize(self, seed):
        rnd = np.random.RandomState(seed)

        n_singles = self.feature_encoder.num_single_features
        self.singleton_ = [rnd.normal(size=n) for n in n_singles]

        n_targ = n_singles[0]
        n_pairs = self.feature_encoder.num_pair_features
        self.pairwise_ = [rnd.normal(size=n).reshape((n_targ, -1)) for n in n_pairs]
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
        pairwise_offset = sum(s.size for s in self.singleton_)

        for i, s in enumerate(self.singleton_):
            n = s.size
            i1 = singleton_offset
            i2 = i1 + n
            w1 = flat_weights[i1:i2].copy()
            singleton_.append(w1)
            singleton_offset += n

            if i == 0:
                w2 = None
            else:
                n0 = self.singleton_[0].size
                i1 = pairwise_offset
                i2 = i1 + (n0 * n)
                w2 = flat_weights[i1:i2].copy().reshape((n0, n))
                pairwise_offset += (n0 * n)

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


