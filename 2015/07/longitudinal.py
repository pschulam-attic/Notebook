import pandas as pd


def LongitudinalSpec(index, date, outcome):
    'The index, timestamp, and outcome column names of a longitudinal dataset.'
    return {'index': index, 'date': date, 'outcome': outcome}


def AlignmentSpec(index, time, date, baseline):
    'The date and baseline column names used to align longitudinal data.'
    return {'index': index, 'time': time, 'date': date, 'baseline': baseline}


def read_longitudinal(stream, names, renames):
    tbl = pd.read_csv(stream).loc[:, [names['index'], names['date'], names['outcome']]]
    tbl.columns = [renames['index'], renames['date'], renames['outcome']]
    tbl[renames['date']] = pd.to_datetime(tbl[renames['date']])
    return tbl[~tbl[renames['outcome']].isnull()]


def align_longitudinal(dataset, dates, alignment):
    aligned_dataset = pd.merge(dataset, dates, 'left', alignment['index'])
    date = aligned_dataset[alignment['date']]
    base = aligned_dataset[alignment['baseline']]
    aligned_dataset[alignment['time']] = (date - base).dt.days / 365.0
    return aligned_dataset[[alignment['index'], alignment['time'], dataset.columns[2]]]
