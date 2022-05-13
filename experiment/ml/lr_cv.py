from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from tempfile import mkdtemp
cachedir = mkdtemp()

params = {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10],
         }


ml=LogisticRegression(n_jobs=1, solver='liblinear')

grid_est = HalvingGridSearchCV(ml,
                          params,
                          cv = 5
                         )


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

est = Pipeline(steps = [('preprocessor', numeric_transformer), 
                        ('estimator',grid_est)
                       ],
               memory=cachedir
              )
