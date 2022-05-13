from ml.pmc.auditor import Auditor
import scipy.stats #import randint

groups = ['ethnicity','gender'] #,'anchor_year_group']

params = dict(
    auditor_type = Auditor(groups=groups),
    eta = 0.25,
    gamma=0.05,
    alpha=0.05,
    rho=0.1,
    # max_iters=10000,
    max_iters=10**5,
    verbosity=1,
    n_bins=7,
    # iter_sample='bootstrap'
)
# Etas = [0.1, 0.5, 1.0]
Etas = scipy.stats.uniform(0.01,1)
# Gammas = [0.1, 0.5, 1.0]
Gammas = scipy.stats.uniform(0.05,.2)
# Alphas = [0.01, 0.05, 0.1]
Alphas = scipy.stats.uniform(0.01,.15)
# N_binses = [2, 5, 10]
N_binses = scipy.stats.randint(2,10)
# Rhos = [0.05, 0.1, 0.2]
Rhos = scipy.stats.uniform(0.05,.15)
Iter_samples = [None, 'bootstrap']
# Splits = [.25, 0.5, 0.75]
Splits = scipy.stats.uniform(0.25,.75)

mc_hyper_params = dict(
    eta = Etas,
    gamma= Gammas,
    alpha= Alphas,
    n_bins=N_binses,
    iter_sample=Iter_samples,
    split=Splits
)
pmc_hyper_params = dict(rho=Rhos)
pmc_hyper_params.update(mc_hyper_params)
