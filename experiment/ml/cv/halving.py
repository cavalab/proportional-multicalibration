from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import KFold

n_splits = 5

cv = KFold(
           n_splits=n_splits, 
           shuffle=False
          )

params = dict(
    cv=cv, 
    verbose=2, 
    n_jobs=1, 
)

CV = HalvingGridSearchCV
