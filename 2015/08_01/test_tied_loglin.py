import numpy as np
import os
import logging
from longloglin import loglin
from longloglin import learn
from mypy.util import check_grad
from scipy.optimize import minimize
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)


def load_dataset(factors_dir):
    ptids = load_single('ptids.npy', factors_dir)
    folds = load_single('folds.npy', factors_dir)

    future = load_single('future.npy', factors_dir)
    loglik = load_multiple('loglik.npz', factors_dir)
    single = load_multiple('single.npz', factors_dir)
    pair = load_multiple('pair.npz', factors_dir)

    dataset = loglin.LogLinearDataset(future, loglik, single, pair)

    return ptids, folds, dataset


def load_single(name, factors_dir):
    fn = os.path.join(factors_dir, name)
    return np.load(fn)


def load_multiple(name, factors_dir):
    fn = os.path.join(factors_dir, name)
    archive = np.load(fn)
    keys = archive.keys()
    return [archive[k] for k in sorted(keys)]


#censor_times = [1.0, 2.0, 3.0, 4.0, 8.0]
censor_times = [1.0, 2.0, 4.0]
all_dirs = ['data/factors_w_sev/{:.01f}'.format(c) for c in censor_times]
loaded_datasets = [load_dataset(d) for d in all_dirs]

ptids, folds, _ = loaded_datasets[0]
datasets = [d for _, _, d in loaded_datasets]

futures = [d[0] for d in datasets]
has_future = [np.any(f, axis=1) for f in futures]
posteriors = [np.zeros_like(f) for f in futures]

for p in posteriors:
    print(p.shape)

time_penalty = 1e-3
l1_penalty = 9e-4

def estimate_weights(fold):
    holdout = folds == fold
    train_datasets = []
    for dset, has_fut in zip(datasets, has_future):
        d = loglin.sub_dataset(dset, ~holdout & has_fut)
        train_datasets.append(d)

    weights = learn.learn_time_regularized_weights(train_datasets, censor_times, time_penalty=time_penalty, l1_penalty=l1_penalty)
    return weights

pool = Pool(3)
all_weights = pool.map(estimate_weights, [k + 1 for k in range(10)])

for i, weights in enumerate(all_weights):
    fold = i + 1
    holdout = folds == fold

    for j, w in enumerate(weights):
        test_data = loglin.sub_dataset(datasets[j], holdout)
        _, marg, _ = loglin.inference(w, test_data)
        posteriors[j][holdout, :] = marg[0]


# for fold in sorted(set(folds)):
#     logging.info('Staring fold {}'.format(fold))
    
#     holdout = folds == fold
#     train_datasets = []
#     for dset, has_fut in zip(datasets, has_future):
#         d = loglin.sub_dataset(dset, ~holdout & has_fut)
#         train_datasets.append(d)

#     all_weights = learn.learn_time_regularized_weights(train_datasets, censor_times, time_penalty=1e-3, l1_penalty=1e-3)

#     for i, w in enumerate(all_weights):
#         test_data = loglin.sub_dataset(datasets[i], holdout)
#         _, marg, _ = loglin.inference(w, test_data)
#         posteriors[i][holdout, :] = marg[0]

import pandas as pd
for c, p in zip(censor_times, posteriors):
    tbl = pd.DataFrame(p)
    tbl.columns = ['p{}'.format(i + 1) for i in range(9)]
    tbl['ptid'] = ptids

    fn = 'posteriors_{:.01f}.csv'.format(c)
    tbl.set_index('ptid').to_csv(fn)
