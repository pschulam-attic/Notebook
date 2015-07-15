import pandas as pd

from zipfile import ZipFile

from longitudinal import LongitudinalSpec, AlignmentSpec, read_longitudinal, align_longitudinal


ssc_zip = 'data/scleroderma.zip'


def variables(file, zipfile=ssc_zip):
    with zipfile.open(file) as f:
        tbl = pd.read_csv(f, nrows=1)
    return list(tbl.columns)


def is_variable(name, file, zipfile=ssc_zip):
    with zipfile.open(file) as f:
        tbl = pd.read_csv(f, nrows=1)
    return name in tbl


def read_dates(file='tPtData.csv', zipfile=ssc_zip):
    with zipfile.open(file) as f:
        ptdata = pd.read_csv(f)
    dates = ptdata.loc[:, ['PtID', 'DateFirstSeen', 'Date1stSymptom']]
    dates['DateFirstSeen'] = pd.to_datetime(dates['DateFirstSeen'])
    dates['Date1stSymptom'] = pd.to_datetime(dates['Date1stSymptom'])
    dates.columns = ['ptid', 'first_seen', 'first_symptom']
    return dates


def read_from_table(index, date, file, zipfile=ssc_zip, **kwargs):
    columns = [(k, v) for k, v in kwargs.items() if is_variable(v, file, zipfile)]
    if not len(columns) == 1:
        raise RuntimeError('Must specify exactly one outcome column.')
    else:
        rename, outcome = columns[0]
    
    names = LongitudinalSpec(index, date, outcome)
    renames = LongitudinalSpec('ptid', 'date', rename)
    
    with zipfile.open(file) as f:
        tbl = read_longitudinal(f, names, renames)
        
    return tbl


def read_from_visits(zipfile=ssc_zip, **kwargs):
    return read_from_table('PtID', 'Visit.Date', 'tVisit.csv', zipfile, **kwargs)


def read_from_echos(zipfile=ssc_zip, **kwargs):
    return read_from_table('PtID', 'Date.of.ECHO', 'tECHO.csv', zipfile, **kwargs)


def read_from_pfts(zipfile=ssc_zip, **kwargs):
    return read_from_table('PtID', 'Date', 'tPFT.csv', zipfile, **kwargs)
