from .xgb import est as base_est
from .xgb import hyper_params
from .cv.halving import CV,params

est = CV(base_est,
         param_grid=hyper_params,
         **params
        )
