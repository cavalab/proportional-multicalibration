from sklearn import ensemble
import numpy as np

hyper_params = [{
    'n_estimators': (100, 5000),
    'min_weight_fraction_leaf': (0.0,  0.5),
    'max_features': ('sqrt','log2'),
}]


est=ensemble.RandomForestClassfier()

def complexity(est):
    return np.sum([e.tree_.node_count for e in est.estimators_])
model = None
