from sklearn.linear_model import LogisticRegression
import numpy as np

params = {
            'penalty': ['l2', 'none'],
            'C': [0.01, 0.1, 5],
            'n_jobs': [-1]
        }


est=LogisticRegression()

def complexity(est):
    return np.sum([e.tree_.node_count for e in est.estimators_])
model = None

