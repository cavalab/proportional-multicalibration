from .lr_mc import est as base_est
from .lr_mc import hyper_params
# from .cv.randhalving import CV,params
from .cv.halving import CV,params
from .pmc.metrics import multicalibration_score
from sklearn.metrics import make_scorer

base_est.verbosity=0

# est = CV(base_est,
#          param_distributions=hyper_params,
#          **params,
#          resource='n_samples',
#          min_resources=10000,
#          # max_resources=10**5,
#         )
est = CV(
        base_est,
        param_grid = hyper_params, 
        **params
        )
