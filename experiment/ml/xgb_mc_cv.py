from .xgb_mc import est as base_est
from .xgb_mc import hyper_params
from .cv.randhalving import CV,params
from .pmc.metrics import multicalibration_score
from sklearn.metrics import make_scorer

base_est.verbosity=0

est = CV(base_est,
         param_distributions=hyper_params,
         **params,
         resource='max_iters',
         min_resources=10,
         max_resources=100,
         scoring=make_scorer(multicalibration_loss,
                             greater_is_better=False)
        )
