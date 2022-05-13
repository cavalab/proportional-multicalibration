from .xgb_pmc import est as base_est
from .xgb_pmc import hyper_params
from .cv.halving import CV,params
from .pmc.metrics import proportional_multicalibration_loss

est = CV(base_est,
         param_grid=hyper_params,
         **params,
         scoring=make_scorer(proportional_multicalibration_loss,
                             greater_is_better=False)
        )
