from .xgb_mc import est as base_est
from .xgb_mc import hyper_params
from .cv.halving import CV,params
from .pmc.metrics import multicalibration_loss

est = CV(base_est,
         param_grid=hyper_params,
         **params,
         scoring=make_scorer(multicalibration_loss,
                             greater_is_better=False)
        )
