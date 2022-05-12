from .xgb import est as base_est
from .cv.halving import halving_params
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np

hyper_params = [
    {
        'n_estimators' : (100,),
        'learning_rate' : (0.003, 0.03, 0.3),
        'gamma' : (0.1,0.4),
        'subsample' : (0.5, 1),
    },
]

est = HalvingGridSearchCV(base_est,
                          param_grid=hyper_params,
                          **halving_params
                         )
