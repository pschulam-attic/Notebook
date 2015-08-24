'''Usage: train_multipsm_posteriors.py CENSOR PENALTY'''

from docopt import docopt
arguments = docopt(__doc__)

import numpy as np
import logging
import dataprep
import multipsm

logging.basicConfig(level=logging.INFO)

censor = float(arguments['CENSOR'])
penalty = float(arguments['PENALTY'])
folds = [k + 1 for k in range(10)]

for fold in folds:
    models = dataprep.get_models(fold)
    col_names = dataprep.col_names
    train_data, _ = dataprep.get_train_test(fold, models, col_names, 30.0)

    prior = multipsm.MultiPSMPrior([8, 4, 6, 4], [5, 3, 3, 3])
    loglik_1 = multipsm.MultiPSMLikelihood(8, 2, -1, 25, 2, 6, 16.0, 36.0, 2.0, 1.0)
    loglik_2 = multipsm.MultiPSMLikelihood(4, 2, -1, 25, 2, 6,  9.0, 16.0, 2.0, 1.0)
    loglik_3 = multipsm.MultiPSMLikelihood(6, 2, -1, 25, 2, 6, 16.0, 36.0, 2.0, 1.0)
    loglik_4 = multipsm.MultiPSMLikelihood(4, 2, -1, 25, 2, 6, 25.0, 16.0, 2.0, 1.0)

    init_parameters = [(m.b, m.B) for m in models]
    no_adjust = {0}

    model = multipsm.MultiPSM(prior, [loglik_1, loglik_2, loglik_3, loglik_4], init_parameters, no_adjust)
    model.fit([X for X, _ in train_data])

    test_idx = dataprep.get_train_test_idx(fold)
    _, test_data = dataprep.get_train_test(fold, models, col_names, censor_time)

    posteriors = np.array([model.posterior(X) for X, _ in test_data])
