from .xgb import est as base_est
from .pmc.multicalibrator import MultiCalibrator
from .pmc.params import params
from .pmc.params import mc_hyper_params as hyper_params

est = MultiCalibrator(
    estimator = base_est,
    metric='MC',
    **params
)

