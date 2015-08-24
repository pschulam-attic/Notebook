import os
import pandas as pd
import numpy as np
import nips15
import logging
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist


folds_dir = 'models/jmlr/folds'

demographic = ['female', 'afram']
molecular = ['aca', 'scl']

pfvc_spec = {'t':'years_seen_full', 'y':'pfvc', 'x1':demographic, 'x2':demographic + molecular}
pfvc = pd.read_csv('data/benchmark_pfvc.csv')
pfvc_pd = [nips15.PatientData.from_tbl(tbl, **pfvc_spec) for _, tbl in pfvc.groupby('ptid')]

tss_spec = {'t':'years_seen', 'y':'tss', 'x1':demographic, 'x2':demographic + molecular}
tss = pd.read_csv('data/benchmark_tss.csv')
tss_match = ['ptid'] + tss_spec['x1']
tss = pd.merge(pfvc[['ptid'] + tss_spec['x2']], tss, 'left', tss_match).drop_duplicates()
tss_pd = [nips15.PatientData.from_tbl(tbl, **tss_spec) for _, tbl in tss.groupby('ptid')]

pdlco_spec = {'t':'years_seen', 'y':'pdlco', 'x1':demographic, 'x2':demographic + molecular}
pdlco = pd.read_csv('data/benchmark_pdc.csv')
pdlco_match = ['ptid'] + pdlco_spec['x1']
pdlco = pd.merge(pfvc[['ptid'] + pdlco_spec['x2']], pdlco, 'left', pdlco_match).drop_duplicates()
pdlco_pd = [nips15.PatientData.from_tbl(tbl, **pdlco_spec) for _, tbl in pdlco.groupby('ptid')]

pv1_spec = {'t':'years_seen', 'y':'pfev1', 'x1':demographic, 'x2':demographic + molecular}
pv1 = pd.read_csv('data/benchmark_pv1.csv')
pv1_match = ['ptid'] + pv1_spec['x1']
pv1 = pd.merge(pfvc[['ptid'] + pv1_spec['x2']], pv1, 'left', pv1_match).drop_duplicates()
pv1_pd = [nips15.PatientData.from_tbl(tbl, **pv1_spec) for _, tbl in pv1.groupby('ptid')]

sp_spec = {'t':'years_seen', 'y':'rvsp', 'x1':demographic, 'x2':demographic + molecular}
sp = pd.read_csv('data/benchmark_sp.csv')
sp_match = ['ptid'] + sp_spec['x1']
sp = pd.merge(pfvc[['ptid'] + sp_spec['x2']], sp, 'left', sp_match).drop_duplicates()
sp_pd = [nips15.PatientData.from_tbl(tbl, **sp_spec) for _, tbl in sp.groupby('ptid')]

get_ptids = lambda pd: [p.ptid for p in pd]
pfvc_df   = pd.DataFrame({'ptid': get_ptids(pfvc_pd),  'pfvc' : pfvc_pd}).set_index('ptid')
tss_df    = pd.DataFrame({'ptid': get_ptids(tss_pd),   'tss'  : tss_pd}).set_index('ptid')
pdlco_df  = pd.DataFrame({'ptid': get_ptids(pdlco_pd), 'pdlco': pdlco_pd}).set_index('ptid')
pv1_df    = pd.DataFrame({'ptid': get_ptids(pv1_pd),   'pv1'  : pv1_pd}).set_index('ptid')
sp_df     = pd.DataFrame({'ptid': get_ptids(sp_pd),    'rvsp' : sp_pd}).set_index('ptid')

folds_df = pfvc.loc[:, ['ptid', 'fold']].drop_duplicates().set_index('ptid')

patient_data = pd.concat([folds_df, pfvc_df, tss_df, pdlco_df, pv1_df, sp_df], axis=1, join='inner')

model_names = ['pfvc', 'tss', 'pdc', 'pv1', 'sp']
col_names   = ['pfvc', 'tss', 'pdlco', 'pv1', 'rvsp']


def load_model(marker, fold, folds_dir=folds_dir):
    param_dir = os.path.join(folds_dir, marker, '{:02d}'.format(fold), 'param')
    return nips15.NipsModel.from_directory(param_dir)


def get_models(fold, model_names=model_names):
    return [load_model(m, fold) for m in model_names]


def get_train_test_idx(fold):
    test = patient_data['fold'].values == fold
    train = ~test
    return train, test


def get_train_test(fold, models, col_names, censor_time):
    train, test = get_train_test_idx(fold)

    train_data = make_examples(patient_data[train], col_names, models, censor_time)
    test_data = make_examples(patient_data[test], col_names, models, censor_time)

    return train_data, test_data


def make_examples(patient_data, col_names, models, censor_time, aux_censor=None):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for j, h in enumerate(histories):
            if j > 0 and aux_censor is not None:
                d_obs = h.truncate(aux_censor).unpack()
            else:
                d_obs = h.truncate(censor_time).unpack()
            X.append(d_obs)
            
        X_unobs = []
        for j, (m, h) in enumerate(zip(models, histories)):
            if j > 0 and aux_censor is not None:
                d_unobs = h.truncate(aux_censor, after=True).unpack()
            else:
                d_unobs = h.truncate(censor_time, after=True).unpack()
                
            X_unobs.append(d_unobs)
            
        p = models[0].posterior(*histories[0].unpack())
        y_hat = np.argmax(p)
        
        y = (y_hat, X_unobs)
            
        ex = (X, y)
        examples.append(ex)
        
    return examples


def onehot_features(z, k):
    x = np.zeros(k)
    x[z] = 1
    return x


def onehot_to_hierarchical(x, links, ngroups=2):
    k = x.size
    m = links.shape[0] - (ngroups - 1)
    h = np.zeros(k + m)
    h[:k] = x

    for i, merge in enumerate(links[:-(ngroups - 1)]):
        g1, g2 = merge
        h[k + i] = h[g1] + h[g2]

    return h


def interact_features(x1, x2):
    x1 = x1.ravel()[:, np.newaxis]
    x2 = x2.ravel()[np.newaxis, :]
    x = (x1 * x2).ravel()
    return x


def single_features(z, k, x):
    x = x.ravel()
    X = np.zeros((k, x.size))
    X[z, :] = x
    return X.ravel()


def pair_features(z1, k1, z2, k2, x):
    x = x.ravel()
    X = np.zeros((k1, k2, x.size))
    X[z1, z2, :] = x
    return X.ravel()


def merge_subtypes(model):
    distances = pdist(model.B, 'euclidean')
    links = hierarchy.average(distances)
    return links[:, :2]


def make_factors(censor_time, patient_data=patient_data, model_names=model_names, col_names=col_names):
    folds = sorted(list(set(patient_data['fold'])))

    targ_col = col_names[0]
    targ_mod = model_names[0]
    future_loglik = []
    for k, pd in zip(patient_data['fold'], patient_data[targ_col]):
        m = load_model(targ_mod, k)
        d_unobs = pd.truncate(censor_time, after=True).unpack()
        lp = m.likelihood(*d_unobs)
        future_loglik.append(lp)

    future_loglik = np.array(future_loglik)

    loglik_factor = []
    for i, (mname, cname) in enumerate(zip(model_names, col_names)):
        loglik = []
        for k, pd in zip(patient_data['fold'], patient_data[cname]):
            m = load_model(mname, k)
            d_obs = pd.truncate(censor_time).unpack()
            lp = m.likelihood(*d_obs)
            loglik.append(lp)

        loglik = np.array(loglik)
        loglik_factor.append(loglik)

    single_factor = []
    for i, (mname, cname) in enumerate(zip(model_names, col_names)):
        model = load_model(mname, 1)
        links = merge_subtypes(model)
        num_subtypes = model.num_subtypes
        single = []
        for pd in patient_data[cname]:
            x = pd.x2
            X = []
            for z in range(num_subtypes):
                f = onehot_features(z, num_subtypes)
                # if i > 0:
                #     f = onehot_to_hierarchical(f, links)
                f = interact_features(f, x)
                X.append(f)
            X = np.array(X)
            # X = np.array([single_features(z, num_subtypes, x) for z in range(num_subtypes)])
            single.append(X)

        single = np.array(single)
        single_factor.append(single)

    pair_factor = []
    models = [load_model(m, 1) for m in model_names]
    links0 = merge_subtypes(models[0])
    k0 = models[0].num_subtypes
    for m in models[1:]:
        links = merge_subtypes(m)
        k = m.num_subtypes
        n = patient_data.shape[0]
        x = np.array([1])        
        pair = []
        for i in range(n):
            X = []
            for z0 in range(k0):
                f0 = onehot_features(z0, k0)
                # f0 = onehot_to_hierarchical(f0, links0)
                for z in range(k):
                    f = onehot_features(z, k)
                    # f = onehot_to_hierarchical(f, links)
                    f = interact_features(f0, f)
                    X.append(f)
                    
            X = np.array(X)

            #X = np.array([pair_features(z0, k0, z, k, x) for z0 in range(k0) for z in range(k)])            

            num_feat = X.shape[-1]
            X = X.reshape((k0, k, num_feat))
            pair.append(X)

        pair = np.array(pair)
        pair_factor.append(pair)

    folds = patient_data['fold'].values

    return future_loglik, loglik_factor, single_factor, pair_factor, folds


def make_factors_w_sev(censor_time, patient_data=patient_data, model_names=model_names, col_names=col_names):
    logging.info('Computing base factors.')
    future_loglik, loglik_factor, single_factor, pair_factor, folds = make_factors(censor_time, patient_data, model_names, col_names)

    sev_scores = ['rp', 'gi', 'kid', 'msc', 'hrt']
    sev_score_dir = 'data'
    sev_score_files = ['{}_likelihoods_{}.dat'.format(ss, censor_time) for ss in sev_scores]
    sev_score_paths = [os.path.join(sev_score_dir, f) for f in sev_score_files]

    targ_pd = patient_data[col_names[0]]
    targ_model = load_model(model_names[0], 1)
    num_targ_subtypes = targ_model.num_subtypes

    for p in sev_score_paths:
        logging.info('Starting {}'.format(p))
        
        ll = np.loadtxt(p)
        loglik_factor.append(ll)

        k0 = num_targ_subtypes
        k = ll.shape[1]

        single = []
        for pd in targ_pd:
            x = pd.x2
            X = []
            for z in range(k):
                f = onehot_features(z, k)
                f = interact_features(f, x)
                X.append(f)
            X = np.array(X)
            single.append(X)
            
        single = np.array(single)
        single_factor.append(single)

        logging.info('Finished singles')

        pair = []
        for pd in targ_pd:
            X = []
            for z0 in range(k0):
                f0 = onehot_features(z0, k0)
                for z in range(k):
                    f = onehot_features(z, k)
                    f = interact_features(f0, f)
                    X.append(f)

            X = np.array(X)
            num_feat = X.shape[-1]
            X = X.reshape((k0, k, num_feat))
            pair.append(X)

        pair = np.array(pair)
        pair_factor.append(pair)

        logging.info('Finished pairs.')

    return future_loglik, loglik_factor, single_factor, pair_factor, folds
