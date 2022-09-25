from .rf import est as base_est
from pmc.multicalibrator import MultiCalibrator
from .pmc_params import params
from .pmc_params import pmc_hyper_params as hyper_params

est = MultiCalibrator(estimator=base_est, metric='PMC', **params)
