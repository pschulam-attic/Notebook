import numpy as np
import pandas as pd
import dataprep


def get_posteriors(model, data, censor=None):
    if censor is None:
        P = [model.posterior(*d.unpack()) for d in data]
    else:
        P = [model.posterior(*d.truncate(censor).unpack()) for d in data]
        
    return np.array(P)


num_subtypes = 9
folds = [k + 1 for k in range(10)]
censor_times = [1.0, 2.0, 4.0]

for censor in censor_times:
    posteriors = np.zeros((dataprep.patient_data.shape[0], num_subtypes))
    
    for fold in folds:
        model = dataprep.load_model('pfvc', fold)
        test = dataprep.patient_data['fold'].values == fold
        posteriors[test, :] = get_posteriors(model, dataprep.patient_data['pfvc'][test], censor)

    tbl = pd.DataFrame(posteriors)
    tbl.index = dataprep.patient_data.index
    tbl.columsn = ['p{}'.format(k + 1) for k in range(num_subtypes)]
    tbl.to_csv('orig_posteriors_{:.01f}.csv'.format(censor))
