import os
import numpy as np
import dataprep
import logging

logging.basicConfig(level=logging.INFO)

#base_directory = 'data/factors'
base_directory = 'data/factors_w_sev'

censor_times = [1.0, 2.0, 3.0, 4.0, 8.0]
factors = []

ptids = dataprep.patient_data.index.values

for censor in censor_times:
    factor = dataprep.make_factors_w_sev(censor)
    factors.append(factor)

    directory = '{}/{:.01f}'.format(base_directory, censor)
    os.makedirs(directory, exist_ok=True)

    future, loglik, single, pair, folds = factor
    np.save(os.path.join(directory, 'ptids'), ptids)
    np.save(os.path.join(directory, 'future'), future)
    np.savez(os.path.join(directory, 'loglik'), *loglik)
    np.savez(os.path.join(directory, 'single'), *single)
    np.savez(os.path.join(directory, 'pair'), *pair)
    np.save(os.path.join(directory, 'folds'), folds)



