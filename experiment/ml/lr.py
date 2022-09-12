from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
import numpy as np
from tempfile import mkdtemp
cachedir = mkdtemp()

ml = LogisticRegression(n_jobs=-1, solver='saga',penalty='l1')


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

est = Pipeline(steps = [('preprocessor', numeric_transformer), 
                        ('estimator',ml)
                       ],
               memory=cachedir
              )
