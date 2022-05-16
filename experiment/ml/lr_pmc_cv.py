from .lr_pmc import est as base_est
from .lr_pmc import hyper_params
# from .cv.randhalving import CV,params
from .cv.halving import CV,params

base_est.verbosity=0

# est = CV(base_est,
#          param_distributions=hyper_params,
#          **params,
#          resource='max_iters',
#          min_resources=100,
#          max_resources=10000,
#         )
est = CV(
        base_est,
        param_grid = hyper_params, 
        **params
        )
