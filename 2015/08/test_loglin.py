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


factors_dir = 'data/factors/4.0'
ptids, folds, dataset = load_dataset(factors_dir)

future = dataset[0]
has_future = np.any(future, axis=1)
posteriors = np.zeros_like(future)

def estimate_weights(fold):
    holdout = folds == fold
    train_data = loglin.sub_dataset(dataset, ~holdout & has_future)
    weights = learn.learn_weights(train_data, 1e-3)

pool = Pool(2)
weights = pool.map(estimate_weights, [k + 1 for k in range(10)])

# for fold in sorted(set(folds)):
#     logging.info('Staring fold {}'.format(fold))
    
#     holdout = folds == fold
#     train_data = loglin.sub_dataset(dataset, ~holdout & has_future)

#     weights = learn.learn_weights(train_data, 1e-3)
#     test_data = loglin.sub_dataset(dataset, holdout)
#     _, marg, _ = loglin.inference(weights, test_data)
#     posteriors[holdout, :] = marg[0]
    
#     penalties = np.logspace(-4, -2, 5)
#     scores = learn.learn_weights_cv(train_data, penalties, 10)

#     penalty = max(scores_and_penalties)[1]
#     logging.info('Using penalty {:.03e}'.format(penalty))
#     weights = learn.learn_weights(train_data, penalty)

    
# import pandas as pd
# tbl = pd.DataFrame(posteriors)
# tbl.columns = ['p{}'.format(i + 1) for i in range(8)]
# tbl['ptid'] = ptids
# tbl.set_index('ptid').to_csv('posteriors.csv')
