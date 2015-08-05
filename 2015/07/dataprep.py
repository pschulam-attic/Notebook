import os
import pandas as pd
import numpy as np
import nips15


folds_dir = 'models/jmlr/folds'

demographic = ['female', 'afram']
molecular = ['aca', 'scl']

pfvc_spec = {'t':'years_seen_full', 'y':'pfvc', 'x1':demographic, 'x2':demographic + molecular}
pfvc = pd.read_csv('data/benchmark_pfvc.csv')
pfvc_pd = [nips15.PatientData.from_tbl(tbl, **pfvc_spec) for _, tbl in pfvc.groupby('ptid')]

tss_spec = {'t':'years_seen', 'y':'tss', 'x1':demographic, 'x2':demographic}
tss = pd.read_csv('data/benchmark_tss.csv')
tss_match = ['ptid'] + tss_spec['x1']
tss = pd.merge(pfvc[tss_match], tss, 'left', tss_match).drop_duplicates()
tss_pd = [nips15.PatientData.from_tbl(tbl, **tss_spec) for _, tbl in tss.groupby('ptid')]

pdlco_spec = {'t':'years_seen', 'y':'pdlco', 'x1':demographic, 'x2':demographic}
pdlco = pd.read_csv('data/benchmark_pdc.csv')
pdlco_match = ['ptid'] + pdlco_spec['x1']
pdlco = pd.merge(pfvc[pdlco_match], pdlco, 'left', pdlco_match).drop_duplicates()
pdlco_pd = [nips15.PatientData.from_tbl(tbl, **pdlco_spec) for _, tbl in pdlco.groupby('ptid')]

pv1_spec = {'t':'years_seen', 'y':'pfev1', 'x1':demographic, 'x2':demographic}
pv1 = pd.read_csv('data/benchmark_pv1.csv')
pv1_match = ['ptid'] + pv1_spec['x1']
pv1 = pd.merge(pfvc[pv1_match], pv1, 'left', pv1_match).drop_duplicates()
pv1_pd = [nips15.PatientData.from_tbl(tbl, **pv1_spec) for _, tbl in pv1.groupby('ptid')]

sp_spec = {'t':'years_seen', 'y':'rvsp', 'x1':demographic, 'x2':demographic}
sp = pd.read_csv('data/benchmark_sp.csv')
sp_match = ['ptid'] + sp_spec['x1']
sp = pd.merge(pfvc[sp_match], sp, 'left', sp_match).drop_duplicates()
sp_pd = [nips15.PatientData.from_tbl(tbl, **sp_spec) for _, tbl in sp.groupby('ptid')]

get_ptids = lambda pd: [p.ptid for p in pd]
pfvc_df   = pd.DataFrame({'ptid': get_ptids(pfvc_pd),  'pfvc' : pfvc_pd}).set_index('ptid')
tss_df    = pd.DataFrame({'ptid': get_ptids(tss_pd),   'tss'  : tss_pd}).set_index('ptid')
pdlco_df  = pd.DataFrame({'ptid': get_ptids(pdlco_pd), 'pdlco': pdlco_pd}).set_index('ptid')
pv1_df    = pd.DataFrame({'ptid': get_ptids(pv1_pd),   'pv1'  : pv1_pd}).set_index('ptid')
sp_df     = pd.DataFrame({'ptid': get_ptids(sp_pd),    'rvsp' : sp_pd}).set_index('ptid')

folds_df = pfvc.loc[:, ['ptid', 'fold']].drop_duplicates().set_index('ptid')

patient_data = pd.concat([folds_df, pfvc_df, tss_df, pdlco_df, pv1_df, sp_df], axis=1, join='inner')

model_names = ['pfvc', 'tss', 'pdc', 'pv1']
col_names   = ['pfvc', 'tss', 'pdlco', 'pv1']


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



    



