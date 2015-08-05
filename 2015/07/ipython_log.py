# IPython log file

import sys
sys.path.append('/Users/pschulam/Git/mypy')
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nips15
import online
import loglin
get_ipython().magic('matplotlib inline')
folds_dir = 'models/jmlr/folds'

def load_model(marker, fold, folds_dir=folds_dir):
    param_dir = os.path.join(folds_dir, marker, '{:02d}'.format(fold), 'param')
    return nips15.NipsModel.from_directory(param_dir)
demographic = ['female', 'afram']
molecular = ['aca', 'scl']

pfvc_spec = {'t':'years_seen_full', 'y':'pfvc', 'x1':demographic, 'x2':demographic + molecular}
pfvc = pd.read_csv('data/benchmark_pfvc.csv')
pfvc_pd = [nips15.PatientData.from_tbl(tbl, **pfvc_spec) for _, tbl in pfvc.groupby('ptid')]

tss_spec = {'t':'years_seen', 'y':'tss', 'x1':demographic, 'x2':demographic}
tss = pd.read_csv('data/benchmark_tss.csv')
tss_match = ['ptid'] + tss_spec['x1']
tss = pd.merge(pfvc[tss_match], tss, 'left', tss_match)
tss_pd = [nips15.PatientData.from_tbl(tbl, **tss_spec) for _, tbl in tss.groupby('ptid')]

pdlco_spec = {'t':'years_seen', 'y':'pdlco', 'x1':demographic, 'x2':demographic}
pdlco = pd.read_csv('data/benchmark_pdc.csv')
pdlco_match = ['ptid'] + pdlco_spec['x1']
pdlco = pd.merge(pfvc[pdlco_match], pdlco, 'left', pdlco_match)
pdlco_pd = [nips15.PatientData.from_tbl(tbl, **pdlco_spec) for _, tbl in pdlco.groupby('ptid')]

pv1_spec = {'t':'years_seen', 'y':'pfev1', 'x1':demographic, 'x2':demographic}
pv1 = pd.read_csv('data/benchmark_pv1.csv')
pv1_match = ['ptid'] + pv1_spec['x1']
pv1 = pd.merge(pfvc[pv1_match], pv1, 'left', pv1_match)
pv1_pd = [nips15.PatientData.from_tbl(tbl, **pv1_spec) for _, tbl in pv1.groupby('ptid')]

sp_spec = {'t':'years_seen', 'y':'rvsp', 'x1':demographic, 'x2':demographic}
sp = pd.read_csv('data/benchmark_sp.csv')
sp_match = ['ptid'] + sp_spec['x1']
sp = pd.merge(pfvc[sp_match], sp, 'left', sp_match)
sp_pd = [nips15.PatientData.from_tbl(tbl, **sp_spec) for _, tbl in sp.groupby('ptid')]
get_ptids = lambda pd: [p.ptid for p in pd]
pfvc_df   = pd.DataFrame({'ptid': get_ptids(pfvc_pd),  'pfvc' : pfvc_pd}).set_index('ptid')
tss_df    = pd.DataFrame({'ptid': get_ptids(tss_pd),   'tss'  : tss_pd}).set_index('ptid')
pdlco_df  = pd.DataFrame({'ptid': get_ptids(pdlco_pd), 'pdlco': pdlco_pd}).set_index('ptid')
pv1_df    = pd.DataFrame({'ptid': get_ptids(pv1_pd),   'pv1'  : pdlco_pd}).set_index('ptid')
sp_df     = pd.DataFrame({'ptid': get_ptids(sp_pd),    'rvsp' : sp_pd}).set_index('ptid')
folds_df = pfvc.loc[:, ['ptid', 'fold']].drop_duplicates().set_index('ptid')
patient_data = pd.concat([folds_df, pfvc_df, tss_df, pdlco_df, pv1_df, sp_df], axis=1, join='inner')
model_names = ['pfvc', 'tss', 'pdc', 'pv1']
col_names   = ['pfvc', 'tss', 'pdlco', 'pv1']
fold = 1
censor = 1.0
models = [load_model(m, fold) for m in model_names]
test = patient_data['fold'].values == fold
pfvc_ll = [history_likel(models[0], d.truncate(censor)) for d in patient_data['pfvc'][test]]
test = patient_data['fold'].values == fold
pfvc_ll = [loglin.history_likel(models[0], d.truncate(censor)) for d in patient_data['pfvc'][test]]
test = patient_data['fold'].values == fold
pfvc_ll = [loglin.history_likelihood(models[0], d.truncate(censor)) for d in patient_data['pfvc'][test]]
test = patient_data['fold'].values == fold
pfvc_ll = [loglin.history_likelihood(models[0], d.truncate(censor).unpack())
           for d in patient_data['pfvc'][test]]
pfvc_ll[0]
pfvc_ll[1]
pfvc_ll[2]
pfvc_ll[3]
loglin.configuration_features((1, 4), [(2, 4), (0, 4)])
import imp
imp.reload(loglin)
loglin.configuration_features((1, 4), [(2, 4), (0, 4)])
loglin.configuration_score([(0, 4), (1, 4), (2, 4)])
features = loglin.configuration_features((0, 4), [(1, 4), (2, 4)])
weights = np.random.normal(size=features.shape)
loglin.configuration_score([(0, 4), (1, 4), (2, 4)], weights)
import imp
imp.reload(loglin)
features = loglin.configuration_features((0, 4), [(1, 4), (2, 4)])
weights = np.random.normal(size=features.shape)
loglin.configuration_score([(0, 4), (1, 4), (2, 4)], weights)
model_names = ['pfvc', 'tss', 'pdc', 'pv1']
col_names   = ['pfvc', 'tss', 'pdlco', 'pv1']

fold = 1
censor = 1.0
models = [load_model(m, fold) for m in model_names]
def make_train_examples(patient_data, col_names, models, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, history in enumerate(marker_histories):
        X = []
        for h in history:
            d = history.truncate(censor_time).unpack()
            X.append(d)
            
        p = models[0].posterior(*history[0].unpack())
        y = np.argmax(p)
        
        ex = (X, y)
        examples.append(ex)
        
    return examples


def make_test_examples(patient_data, col_names, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, history in enumerate(marker_histories)
        X = []
        for h in history:
            d = history.truncate(censor_time).unpack()
            X.append(d)
            
    return examples
def make_train_examples(patient_data, col_names, models, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, history in enumerate(marker_histories):
        X = []
        for h in history:
            d = history.truncate(censor_time).unpack()
            X.append(d)
            
        p = models[0].posterior(*history[0].unpack())
        y = np.argmax(p)
        
        ex = (X, y)
        examples.append(ex)
        
    return examples


def make_test_examples(patient_data, col_names, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, history in enumerate(marker_histories):
        X = []
        for h in history:
            d = history.truncate(censor_time).unpack()
            X.append(d)
            
    return examples
test = patient_data['fold'].values == fold
train = ~test

train_data = make_train_examples(patient_data[train], col_names, models, censor)
test_data = make_test_examples(patient_data[test], col_names, censor)
def make_train_examples(patient_data, col_names, models, censor_time):
    marker_histories = zip(*[list(patient_data[n]) for n in col_names])
    examples = []
    for i, history in enumerate(marker_histories):
        X = []
        for h in history:
            d = history.truncate(censor_time).unpack()
            X.append(d)
            
        p = models[0].posterior(*history[0].unpack())
        y = np.argmax(p)
        
        ex = (X, y)
        examples.append(ex)
        
    return examples


def make_test_examples(patient_data, col_names, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, history in enumerate(marker_histories):
        X = []
        for h in history:
            d = history.truncate(censor_time).unpack()
            X.append(d)
            
    return examples
test = patient_data['fold'].values == fold
train = ~test

train_data = make_train_examples(patient_data[train], col_names, models, censor)
test_data = make_test_examples(patient_data[test], col_names, censor)
mh = [patient_data[n] for n in col_names]
mh
zip(*mh)
list(zip(*mh))[0]
list(zip(*mh))[0][0]
list(zip(*mh))[0][0].unpack()
list(enumerate(zip(*mh)))[0]
def make_train_examples(patient_data, col_names, models, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for h in histories:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
        p = models[0].posterior(*history[0].unpack())
        y = np.argmax(p)
        
        ex = (X, y)
        examples.append(ex)
        
    return examples


def make_test_examples(patient_data, col_names, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, history in enumerate(marker_histories):
        X = []
        for h in history:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
    return examples
test = patient_data['fold'].values == fold
train = ~test

train_data = make_train_examples(patient_data[train], col_names, models, censor)
test_data = make_test_examples(patient_data[test], col_names, censor)
def make_train_examples(patient_data, col_names, models, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for h in histories:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
        p = models[0].posterior(*histories[0].unpack())
        y = np.argmax(p)
        
        ex = (X, y)
        examples.append(ex)
        
    return examples


def make_test_examples(patient_data, col_names, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for h in histories:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
    return examples
test = patient_data['fold'].values == fold
train = ~test

train_data = make_train_examples(patient_data[train], col_names, models, censor)
test_data = make_test_examples(patient_data[test], col_names, censor)
train_data[0]
X, y = train_data[0]
y
X
tss
tss.shape
tss.drop_duplicates()
tss.drop_duplicates().shape
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
pv1_df    = pd.DataFrame({'ptid': get_ptids(pv1_pd),   'pv1'  : pdlco_pd}).set_index('ptid')
sp_df     = pd.DataFrame({'ptid': get_ptids(sp_pd),    'rvsp' : sp_pd}).set_index('ptid')
tss.shape
folds_df = pfvc.loc[:, ['ptid', 'fold']].drop_duplicates().set_index('ptid')
patient_data = pd.concat([folds_df, pfvc_df, tss_df, pdlco_df, pv1_df, sp_df], axis=1, join='inner')
model_names = ['pfvc', 'tss', 'pdc', 'pv1']
col_names   = ['pfvc', 'tss', 'pdlco', 'pv1']

fold = 1
censor = 1.0
models = [load_model(m, fold) for m in model_names]
def make_train_examples(patient_data, col_names, models, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for h in histories:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
        p = models[0].posterior(*histories[0].unpack())
        y = np.argmax(p)
        
        ex = (X, y)
        examples.append(ex)
        
    return examples


def make_test_examples(patient_data, col_names, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for h in histories:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
    return examples
test = patient_data['fold'].values == fold
train = ~test

train_data = make_train_examples(patient_data[train], col_names, models, censor)
test_data = make_test_examples(patient_data[test], col_names, censor)
X, y = train_data[0]
X
y
X = test_data[0]
def make_train_examples(patient_data, col_names, models, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for h in histories:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
        p = models[0].posterior(*histories[0].unpack())
        y = np.argmax(p)
        
        ex = (X, y)
        examples.append(ex)
        
    return examples


def make_test_examples(patient_data, col_names, censor_time):
    marker_histories = zip(*[patient_data[n] for n in col_names])
    examples = []
    for i, histories in enumerate(marker_histories):
        X = []
        for h in histories:
            d = h.truncate(censor_time).unpack()
            X.append(d)
            
        examples.append(X)
            
    return examples
test = patient_data['fold'].values == fold
train = ~test

train_data = make_train_examples(patient_data[train], col_names, models, censor)
test_data = make_test_examples(patient_data[test], col_names, censor)
test_data[0]
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
conditional_model.subtypes_features([0, 0, 0, 0])
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
conditional_model.subtypes_features([0, 0, 0, 0])
conditional_model.subtypes_features([0, 0, 0, 0]).shape
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
conditional_model.initial_weights()
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model.initial_weights()
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
weights
conditional_model.history_score(train_data[0][0], [0, 0, 0, 0])
subtypes = [0, 0, 0, 0]
s1 = conditional_model.history_score(train_data[0][0], subtypes)
s2 = conditional_model.subtypes_score(subtypes, weights)
subtypes = [0, 0, 0, 0]
s1 = conditional_model.history_score(train_data[0][0], subtypes); print(s1)
s2 = conditional_model.subtypes_score(subtypes, weights); print(s2)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.partition_function(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.partition_function(train_data[0][0], weights)
[conditional_model.marginal_score(train_data[0][0], z, weights) for z in range(8)]
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.partition_function(train_data[0][0], weights)
[conditional_model.marginal_score(train_data[0][0], z, weights) for z in range(8)]
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
[conditional_model.marginal_score(train_data[0][0], z, weights) for z in range(8)]
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
[conditional_model.marginal_score(train_data[0][0], z, weights) for z in range(8)]
sum(Out[107])
import imp
imp.reload(loglin)
conditional_model.partition_function(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.partition_function(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.partition_function(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.partition_function(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.partition_function(train_data[0][0], weights)
conditional_model.marginal_score(train_data[0][0], 0, weights)
np.exp(Out[123])
np.exp(Out[123] - np.log(Out[122]))
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.proba(train_data[0][0], weights)
np.round(conditional_model.proba(train_data[0][0], weights), 20
np.round(conditional_model.proba(train_data[0][0], weights), 2)
p = conditional_model.proba(train_data[0][0], weights)
p.sum()
import imp
imp.reload(loglin)
p = conditional_model.proba(train_data[0][0], weights)
p.shape
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
p = conditional_model.proba(train_data[0][0], weights)
p.shape
p.shape[0, 0, 0, 0]
p[0, 0, 0, 0]
p.sum()
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
p = conditional_model.proba(train_data[0][0], weights)
p = conditional_model.log_proba(train_data[0][0], weights)
p.shape
from scipy.misc import logsumexp
logsumexp(p, axis=(1, 2, 3))
from scipy.misc import logsumexp
pmarg = logsumexp(p, axis=(1, 2, 3))
from scipy.misc import logsumexp
pmarg = logsumexp(p, axis=(1, 2, 3))
np.exp(pmarg - logsumexp(pmarg))
from scipy.misc import logsumexp
pmarg = logsumexp(p, axis=(1, 2, 3))
np.round(np.exp(pmarg - logsumexp(pmarg)), 2)
from scipy.misc import logsumexp
pmarg = logsumexp(p, axis=(1, 2, 3))
np.round(np.exp(pmarg - logsumexp(pmarg)), 3)
from scipy.misc import logsumexp
pmarg = logsumexp(p, axis=(1, 2, 3))
np.round(np.exp(pmarg - logsumexp(pmarg)), 4)
from scipy.misc import logsumexp
pmarg = logsumexp(p, axis=(1, 2, 3))
np.round(np.exp(pmarg - logsumexp(pmarg)), 3)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
p = conditional_model.log_proba(train_data[0][0], weights)
m = conditional_model.marg_log_proba(train_data[0][0], weights)
get_ipython().magic('debug ')
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
m = conditional_model.marg_log_proba(train_data[0][0], weights)
np.exp(m)
np.round(np.exp(m), 3)
np.apply_along_axis(np.sum, 0, p)
np.apply_over_axes(np.sum, p, 0)
np.apply_over_axes(np.sum, p, 0).shape
np.apply_along_axis(np.sum, 0, p).shape
get_ipython().magic('pinfo np.swapaxes')
np.ndim
p.ndim
p.shape
tuple(range(p.ndim))
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
p = conditional_model.log_proba(train_data[0][0], weights)
loglin.marginalize(p, 0)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
p = conditional_model.log_proba(train_data[0][0], weights)
loglin.marginalize(p, 0)
loglin.marginalize(p, [0])
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
p = conditional_model.log_proba(train_data[0][0], weights)
loglin.marginalize(p, [0], logsumexp)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
p = conditional_model.log_proba(train_data[0][0], weights)
loglin.marginalize(p, [0], logsumexp)
np.round(np.exp(loglin.marginalize(p, [0], logsumexp)), 3)
loglin.marginalize(p, [0, 1], logsumexp)
np.exp(loglin.marginalize(p, [0, 1], logsumexp))
np.round(np.exp(loglin.marginalize(p, [0, 1], logsumexp)), 2)
np.exp(loglin.marginalize(p, [0, 1], logsumexp))
np.exp(loglin.marginalize(p, [0, 1], logsumexp)).sum()
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.feature_expectations(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.feature_expectations(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.feature_expectations(train_data[0][0], weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
conditional_model.feature_expectations(train_data[0][0], weights)
np.round(Out[209], 3)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
X, y = train_data[0]
e1 = conditional_model.conditional_expectations(X, y, weights)
e2 = conditional_model.feature_expectations(X, weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
X, y = train_data[0]
e1 = conditional_model.conditional_expectations(X, y, weights)
e2 = conditional_model.feature_expectations(X, weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
X, y = train_data[0]
e1 = conditional_model.conditional_expectations(X, y, weights)
e2 = conditional_model.feature_expectations(X, weights)
import imp
imp.reload(loglin)
conditional_model = loglin.ConditionalModel(models)
weights = conditional_model.initial_weights(0)
X, y = train_data[0]
e1 = conditional_model.conditional_expectations(X, y, weights)
e2 = conditional_model.feature_expectations(X, weights)
e1
np.round(e1, 2)
e1 - e2
def obj(w, train_data=train_data, penalty=1.0):
    ll = [conditional_model.marg_log_proba(X, w)[y] for X, y in train_data]
    return np.mean(ll) + penalty / 2.0 * np.dot(w, w)
obj(weights)
import imp
imp.reload(loglin)
model_names = ['pfvc', 'tss', 'pdc', 'pv1']
col_names   = ['pfvc', 'tss', 'pdlco', 'pv1']

fold = 1
censor = 1.0
models = [load_model(m, fold) for m in model_names]
num_subtypes = [m.num_subtypes for m in models]
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
weights
weights.singleton_[0]
weights.singleton_[1]
weights.singleton_[2]
weights.singleton_[3]
weights.singleton_[4]
weights.pairwise_[0]
weights.pairwise_[1]
w = weights.collapsed()
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
w = weights.collapsed()
w
w.shape
weights2 = loglin.ModelWeights(num_subtypes, 1)
weights2.singleton(0)
weights = loglin.ModelWeights(num_subtypes)
weights.singleton(0)
weights = loglin.ModelWeights(num_subtypes)
weights.singleton(0)
weights2 = loglin.ModelWeights(num_subtypes, 1)
weights2.singleton(0)
weights2.set_weights(weights.collapsed())
weights2.set_weights(weights.collapsed())
weights2.singleton(0)
weights2 = loglin.ModelWeights(num_subtypes, 1)
weights2.singleton(3)
weights2.set_weights(weights.collapsed())
weights2.singleton(3)
weights = loglin.ModelWeights(num_subtypes)
weights.singleton(3)
weights = loglin.ModelWeights(num_subtypes)
weights.pairwise(1)
weights2 = loglin.ModelWeights(num_subtypes, 1)
weights2.pairwise(1)
weights2.set_weights(weights.collapsed())
weights2.pairwise(1)
weights2.set_weights(weights.collapsed())
weights2.pairwise(1).shape
weights2 = loglin.ModelWeights(num_subtypes, 1)
weights2.pairwise(1).shape
weights = loglin.ModelWeights(num_subtypes)
weights.pairwise(1).shape
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
weights.pairwise(1).shape
weights = loglin.ModelWeights(num_subtypes)
weights.pairwise(1)
weights2 = loglin.ModelWeights(num_subtypes, 1)
weights2.pairwise(1)
weights2.set_weights(weights.collapsed())
weights2.pairwise(1)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
scorer.data_scores(train_data[0][0])
scorer.data_scores(train_data[0][0])[0]
scorer.data_scores(train_data[0][0])[1]
scorer.data_scores(train_data[0][0])[2]
scorer.data_scores(train_data[0][0])[3]
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
engine.run(train_data[0][0])
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
engine.run(train_data[0][0])
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
engine.run(train_data[0][0])
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
engine.run(train_data[0][0])
s, p = engine.run(train_data[0][0])
s[0]
s[0].sum()
s[1].sum()
s[2].sum()
s[3].sum()
p[0].sum()
p[1].sum()
p[2].sum()
p[3].sum()
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
s, p = engine.run(train_data[0][0])
s[0].sum()
s[1].sum()
s[2].sum()
s[3].sum()
s[3]
loglin.normalize(s[3])
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
s, p = engine.run(train_data[0][0])
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
s, p = engine.run(train_data[0][0])
np.round(s[0])
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
s, p = engine.run(train_data[0][0])
np.round(s[0], 4)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
s, p = engine.run(train_data[0][0])
np.round(s[0], 3)
cm = loglin.ConditionalModel(models)
weights = cm.initial_weights(0)
np.round(weights[:10], 3)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
s, p = engine.run(train_data[0][0])
np.round(s[0], 3)
cm = loglin.ConditionalModel(models)
w = cm.initial_weights(0)
np.round(w[:10], 3)
np.round(weights.collapsed()[:10], 3)
cm = loglin.ConditionalModel(models)
w = cm.initial_weights(0)
np.round(np.exp(cm.marg_log_proba(train_data[0][0], w)), 3)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
s, p = engine.run(train_data[0][0])
np.round(s[0], 3)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
np.round(loglin.feature_expectations(junction_tree)[:20], 3)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
np.round(loglin.feature_expectations(junction_tree)[:20], 3)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
np.round(loglin.feature_expectations(junction_tree)[:20], 3)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
np.round(loglin.feature_expectations(junction_tree)[:20], 3)
cm = loglin.ConditionalModel(models)
w = cm.initial_weights(0)
np.round(cm.feature_expectations(train_data[0][0], w)[:20], 3)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
cm = loglin.ConditionalModel(models)
w = cm.initial_weights(0)
lp = cm.log_proba(train_data[0][0], w)
np.round(junction_tree[0][1], 3)
np.round(np.exp(loglin.marginalize(lp, [1], logsumexp)), 3)
np.round(np.exp(loglin.marginalize(lp, [0, 1], logsumexp)), 3)
np.round(junction_tree[1][1], 3)
np.round(junction_tree[0][1], 3)
np.round(junction_tree[0][2], 3)
np.round(np.exp(loglin.marginalize(lp, [0, 2], logsumexp)), 3)
np.round(np.exp(loglin.marginalize(lp, [2], logsumexp)), 3)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
junction_tree[0]
junction_tree[1][0]
junction_tree[1][0].sum()
logsumexp(junction_tree[1][0])
logsumexp(junction_tree[1][1])
logsumexp(junction_tree[1][2])
logsumexp(junction_tree[1][3])
logsumexp(junction_tree[2][1])
logsumexp(junction_tree[3][1])
logsumexp(junction_tree[3][1])
logsumexp(junction_tree[2][1])
logsumexp(junction_tree[1][1])
logsumexp(junction_tree[2][1])
logsumexp(junction_tree[3][1])
logsumexp(junction_tree[1][1])
logsumexp(junction_tree[2][1])
logsumexp(junction_tree[3][1])
logsumexp(junction_tree[2][1])
logsumexp(junction_tree[2][1])
logsumexp(junction_tree[2][2])
logsumexp(junction_tree[2][3])
logsumexp(junction_tree[2][4])
logsumexp(junction_tree[2][3])
import imp
imp.reload(loglin)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
junction_tree[1][0]
np.round(junction_tree[1][0], 3)
np.round(marginalize(junction_tree[2][1], [0]), 3)
np.round(loglin.marginalize(junction_tree[2][1], [0]), 3)
np.round(loglin.marginalize(junction_tree[2][1], [1]), 3)
np.round(junction_tree[1][1], 3)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
s, p = junction_tree
s[0]
s[1]
s[2]
s[3]
p[1]
p_up = p.copy()
p_up[1] += s[1]
p_up[2] += s[2]
p_up[3] += s[3]
p_up[1]
p[1]
s[1]
s, p = junction_tree
p_up = p.copy()
p_up[1] += s[1]
p_up[2] += s[2]
p_up[3] += s[3]
p_up[1]
p[1]
s[1]
p_up[1, :]
s, p = junction_tree
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
s, p = junction_tree
p_up = [p_i.copy() for p_i in p]
p_up[1] += s[1]
p_up[2] += s[2]
p_up[3] += s[3]
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
s, p = junction_tree
p_up = [None] + [p_i.copy() for p_i in p[1:]]
p_up[1] += s[1]
p_up[2] += s[2]
p_up[3] += s[3]
p_up[1]
p[1]
s[1]
p_up[2]
p[2]
s[2]
loglin.marginalize(p_up[1], [0], logsumexp)
loglin.marginalize(p_up[2], [0], logsumexp)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
s, p = junction_tree
s_up = [s_i.copy() for s_i in s]

p_up = [None] + [p_i.copy() for p_i in p[1:]]
p_up[1] += s[1]
p_up[2] += s[2]
p_up[3] += s[3]
s_up[0] += marginalize(p_up[1], [0], logsumexp)
s_up[0] += marginalize(p_up[2], [0], logsumexp)
s_up[0] += marginalize(p_up[3], [0], logsumexp)
s_up[0] += loglin.marginalize(p_up[1], [0], logsumexp)
s_up[0] += loglin.marginalize(p_up[2], [0], logsumexp)
s_up[0] += loglin.marginalize(p_up[3], [0], logsumexp)
s_up[0]
s_up[0] - logsumexp(s_up[0])
np.exp(s_up[0] - logsumexp(s_up[0]))
np.exp(s_up[0] - logsumexp(s_up[0])).sum()
s_up[0] -= logsumexp(s_up[0]))
s_up[0] -= logsumexp(s_up[0])
s_down = [s_i.copy() for s_i in s_up]
p_down = [p_i.copy() for p_i in p_up]
s_down = [s_i.copy() for s_i in s_up]
p_down = [None] + [p_i.copy() for p_i in p_up[1:]]
p_down[1] += loglin.colvec(s_down[0])
p_down[2] += loglin.colvec(s_down[0])
p_down[3] += loglin.colvec(s_down[0])
p_down[0]
p_down[1]
np.exp(p_down[1] - logsumexp(p_down[1]))
np.exp(p_down[1] - logsumexp(p_down[1])).sum()
np.exp(p_down[1] - logsumexp(p_down[1]))
np.exp(p_down[1] - logsumexp(p_down[1])).sum(axis=1)
np.round(np.exp(p_down[1] - logsumexp(p_down[1])).sum(axis=1), 3)
import imp
imp.reload(loglin)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
junction_tree = engine.run(train_data[0][0])
s, p = junction_tree
logl, s, p = junction_tree
s[0]
np.set_printoptions(precision=3)
s[0]
np.set_printoptions(precision=2)
s[0]
p[1].sum(axis=1)
p[2].sum(axis=1)
p[3].sum(axis=1)
s[1]
p[1].sum(axis=0)
s[2]
p[2].sum(axis=0)
s[3]
p[3].sum(axis=0)
cm = loglin.ConditionalModel(models)
wt = cm.initial_weights()
wt[-10:]
weights.collapsed()[-10:]
lp = cm.log_proba(train_data[0][0], wt)
np.exp(loglin.marginalize(lp, [0], logsumexp))
junction_tree[1][0]
np.exp(loglin.marginalize(lp, [1], logsumexp))
junction_tree[1][1]
np.exp(loglin.marginalize(lp, [2], logsumexp))
junction_tree[1][2]
np.exp(loglin.marginalize(lp, [3], logsumexp))
junction_tree[1][3]
np.exp(loglin.marginalize(lp, [0, 1], logsumexp))
junction_tree[2][1]
junction_tree[2][2]
np.exp(loglin.marginalize(lp, [0, 2], logsumexp))
np.exp(loglin.marginalize(lp, [0, 3], logsumexp))
junction_tree[2][3]
fe = loglin.feature_expectations(junction_tree)
len(junction_tree)
import imp
imp.reload(loglin)
fe = loglin.feature_expectations(junction_tree)
fe
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
def obj(train_data, engine):
    ll = [engine.run(d)[0] for d in train_data]
    return sum(ll)
def obj(train_data, engine):
    ll = [engine.run(d)[0] for d in train_data]
    return np.mean(ll)
obj(train_data, )
obj(train_data, engine)
len(train_data)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
def obj(train_data, engine):
    probs = [engine.run(X)[0][0][y] for X, y in train_data]
    return np.log(probs).sum()
obj(train_data, engine)
def obj(train_data, engine):
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l)
obj(train_data, engine)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
def obj(train_data, penalty=1.0, engine):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) + penalty / 2.0 * np.dot(w, w)
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) + penalty / 2.0 * np.dot(w, w)
obj(train_data, engine)
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)
obj(train_data, engine)
np.inf
-np.exp(np.inf)
np.exp(-np.inf)
x = np.array([0, 0, 1])
y = np.log(x)
y
x = np.array([0, 0, 1])
y = np.log(x)
logsumexp(y)
x = np.array([0, 0, -10, -5])
y = np.log(x)
y
x = np.array([0, 0, 0.4, 0.6])
y = np.log(x)
y
x = np.array([0, 0, 0.4, 0.6])
y = np.log(x)
logsumexp(x)
x = np.array([0, 0, 0.4, 0.6])
y = np.log(x)
np.exp(x - logsumexp(x))
x = np.array([0, 0, 0.4, 0.6])
y = np.log(x)
logsumexp(y)
x = np.array([0, 0, 0.2, 0.3])
y = np.log(x)
logsumexp(y)
np.exp(x - logsumexp(y))
np.exp(y - logsumexp(y))
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
jt = engine.run(train_data[0][0])
jt[0][0]
jt = engine.run(train_data[0][0])
jt_conditioned = engine.observe_target(jt, train_data[0][1])
jt_conditioned[0][0]
jt_conditioned[0][1]
jt_conditioned[0][2]
jt_conditioned[0][1]
jt_conditioned[1][1]
jt_conditioned[0][1].sum()
jt_conditioned[1][1].sum()
jt_conditioned[1][1]
jt_conditioned[0][1]
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
jt = engine.run(train_data[0][0])
jt_conditioned = engine.observe_target(jt, train_data[0][1])
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
jt = engine.run(train_data[0][0])
jt_conditioned = engine.observe_target(jt, train_data[0][1])
jt_conditioned[0][1]
jt_conditioned[1][1]
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)

def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    for X, y in train_data:
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
    return g
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)

def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    
    for X, y in train_data:
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
    
    g += penalty * w
    
    return g
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)

def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    
    for X, y in train_data:
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
    
    g += penalty * w
    
    return g
obj(train_data, engine)
jac(train_data, engine)
import imp
imp.reload(loglin)
weights = loglin.ModelWeights(num_subtypes)
scorer = loglin.Scorer(models, weights)
engine = loglin.InferenceEngine(scorer)
obj(train_data, engine)
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)

def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    
    for X, y in train_data:
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
    
    g += penalty * w
    
    return g
obj(train_data, engine)
jac(train_data, engine)
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)


def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    
    for X, y in train_data:
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
    
    g += penalty * w
    
    return g


def f(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return obj(train_data, engine)

def g(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return jac(train_data, engine)
w0 = engine.scorer.parameters
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)


def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    
    for X, y in train_data:
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
    
    g += penalty * w
    
    return g


def f(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return -obj(train_data, engine)

def g(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return -jac(train_data, engine)
from scipy.optimize import minimize
w0 = engine.scorer.parameters
solution = minimize(f, w0, jac=g, method='BFGS')
from mypy.util import check_grad
from mypy.util import check_grad
w0 = engine.scorer.parameters
check_grad(f, w0)
imp.reload(mypy)
mypy.util.check_grad(f, w0, range(5))
import mypy
imp.reload(mypy)
mypy.util.check_grad(f, w0, range(5))
get_ipython().magic('debug ')
import mypy
import mypy.util
imp.reload(mypy)
imp.reload(mypy.util)
from mypy.util impor check_grad
check_grad(f, w0, range(5))
import mypy
import mypy.util
imp.reload(mypy)
imp.reload(mypy.util)
from mypy.util import check_grad
check_grad(f, w0, range(5))
g(w0)[:5]
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)


def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    n = 0
    
    for X, y in train_data:
        n += 1
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
        
    g /= n
    g += penalty * w
    
    return g


def f(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return -obj(train_data, engine)


def g(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return -jac(train_data, engine)
g(w0)[:5]
import mypy
import mypy.util
imp.reload(mypy)
imp.reload(mypy.util)
from mypy.util import check_grad
check_grad(f, w0, range(2), 1e-8)
g(w0)[:2]
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)


def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    n = 0
    
    for X, y in train_data:
        n += 1
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
        
    g /= n
    g -= penalty * w
    
    return g


def f(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return -obj(train_data, engine)


def g(w, train_data=train_data, engine=engine):
    engine.scorer.weights.set_weights(w)
    return -jac(train_data, engine)
import mypy
import mypy.util
imp.reload(mypy)
imp.reload(mypy.util)
from mypy.util import check_grad
check_grad(f, w0, range(2), 1e-8)
g(w0)[:2]
import mypy
import mypy.util
imp.reload(mypy)
imp.reload(mypy.util)
from mypy.util import check_grad
check_grad(f, w0, range(10), 1e-8)
g(w0)[:10]
from scipy.optimize import minimize
w0 = engine.scorer.parameters
solution = minimize(f, w0, jac=g, method='BFGS', options={'disp': True})
def obj(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    p = [engine.run(X)[0][0][y] for X, y in train_data]
    l = np.log(p)
    return np.mean(l) - penalty / 2.0 * np.dot(w, w)


def jac(train_data, engine, penalty=1e-2):
    w = engine.scorer.parameters
    g = np.zeros_like(w)
    n = 0
    
    for X, y in train_data:
        n += 1
        jt = engine.run(X)
        jt_cond = engine.observe_target(jt, y)
        fe = loglin.feature_expectations(jt)
        fe_cond = loglin.feature_expectations(jt_cond)
        g += fe_cond - fe
        
    g /= n
    g -= penalty * w
    
    return g


def f(w, train_data=train_data[:10], engine=engine):
    engine.scorer.weights.set_weights(w)
    return -obj(train_data, engine)


def g(w, train_data=train_data[:10], engine=engine):
    engine.scorer.weights.set_weights(w)
    return -jac(train_data, engine)
from scipy.optimize import minimize
w0 = engine.scorer.parameters
solution = minimize(f, w0, jac=g, method='BFGS', options={'disp': True})
solution.x
engine.scorer.weights.set_weights(solution.x)
engine.run(train_data[0][0])
engine.run(train_data[0][0])[0][0]
np.round(engine.run(train_data[0][0])[0][0], 2)
for X, y in train_data[:10]:
    p = engine.run(X)[0][0]
    print(y)
    print(np.round(p, 2))
import imp
imp.reload(loglin)
objective = loglin.ModelObjective(training_data, 1e-2, models)
objective = loglin.ModelObjective(train_data, 1e-2, models)
import imp
imp.reload(loglin)
objective = loglin.ModelObjective(train_data, 1e-2, models)
import imp
imp.reload(loglin)
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
from scipy.optimize import minimize
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
objective.value(w0)
import imp
imp.reload(loglin)
from scipy.optimize import minimize
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
objective.value(w0)
check_grad(objective.value, w0, range(2))
objective.gradient(w0)
import imp
imp.reload(loglin)
from scipy.optimize import minimize
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
objective.value(w0)
check_grad(objective.value, w0, range(2))
objective.gradient(w0)
check_grad(objective.value, w0, range(5))
objective.gradient(w0)[:5]
import imp
imp.reload(loglin)
from scipy.optimize import minimize
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
solution = minimize(objective.value, w0, jac=objective.gradient, method='BFGS')
import logging
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
objective.value(w0)
#solution = minimize(objective.value, w0, jac=objective.gradient, method='BFGS')
import imp
imp.reload(loglin)
import logging
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
objective.value(w0)
#solution = minimize(objective.value, w0, jac=objective.gradient, method='BFGS')
logging.info('Hello world')
import logging
from scipy.optimize import minimize
get_ipython().magic('logstart')

logging.basicConfig(level=logging.INFO)
objective = loglin.ModelObjective(train_data, 1e-2, models)
w0 = objective.initial_weights()
objective.value(w0)
#solution = minimize(objective.value, w0, jac=objective.gradient, method='BFGS')
get_ipython().magic('logstop ')
