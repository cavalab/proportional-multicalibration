from collections import defaultdict

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def jsonify(d):
    """recursively formats dicts for json serialization"""
    if isinstance(d, list):
        d_new = []
        for v in d:
            d_new.append(jsonify(v))
        return d_new
    elif isinstance(d, dict):
        for k in d.keys():
            d[k] = jsonify(d[k])
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif d.__class__.__name__.startswith('int'):
        return int(d)
    elif d.__class__.__name__.startswith('float'):
        return float(d)
    elif isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.values.tolist()
    elif isinstance(d, bool):
        return d
    elif d == None:
        return None
    elif hasattr(d, '__dict__'):
        if hasattr(d, '__name__'):
            tmp = d.__name__
        else:
            tmp = type(d).__name__
        obj = {'object': tmp, 'vars': {}}
        for k, v in vars(d).items():
            obj['vars'][k] = jsonify(v)
        return obj
    elif not isinstance(d, str):
        logger.debug("attempting to store ", d, "as a str for json")
        return str(d)
    return d


def hasattranywhere(C, attr: str):
    """Recursively look thru the class for the attribute in any subclasses. 
    Return None if it's nowhere, or list of nested attribute name otherwise.
    """
    attrs = []
    if hasattr(C, attr):
        attrs.append(attr)

    for k, v in vars(C):
        search = hasattranywhere(v)
        for s in search:
            attrs.append(k + '.' + s)

    if len(attrs) == 0:
        return None

    return attrs


# demographics
A = ['Race', 'Ethnicity', 'Gender', 'Age']

A_rename = {'Hispanic Yes No': 'Ethnicity', 'age_group': 'Age'}

A_options = {
    'Race': defaultdict(lambda: 'Other'),
    'Ethnicity': {
        'Yes': 'HL',
        'No': 'NHL'
    },
    'Gender': {
        'M': 'M',
        'F': 'F'
    },
    # 'Age': {
    #     '>5Y':'older than 5Y',
    #     '18M-3Y':'5Y or younger',
    #     '3-5Y':'5Y or younger',
    #     '12-18M':'5Y or younger',
    #     '0-3M':'5Y or younger',
    #     '6-12M':'5Y or younger',
    #     '3-6M':'5Y or younger'
    #
    # }
}
A_options['Race'].update({
    'Black or African American': 'Black',
    #             'Other',
    'White': 'white',
    #             'Unable to Answer',
    #             'Declined to Answer',
    #             'Unknown',
    'Asian': 'Asian',
    'American Indian or Alaska Native': 'AI',
    'Native Hawaiian or Other Pacific Islander': 'NHPI',
    #         'Hispanic or Latino': 'HLTN'
})


def clean_dems(df):
    """Re-codes demographics according to dictionaries above"""
    df = df.rename(columns=A_rename)
    for a in A:
        if a in A_options.keys():
            df = df.loc[df[a].isin(list(A_options[a].keys())), :]
            df[a] = df[a].apply(lambda x: A_options[a][x])
    return df
