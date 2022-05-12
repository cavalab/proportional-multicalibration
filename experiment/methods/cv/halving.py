from sklearn.model_selection import KFold

n_splits = 5

cv = KFold(
           n_splits=n_splits, 
           shuffle=True
          )

halving_params = dict(
    cv=cv, 
    verbose=2, 
    n_jobs=1, 
    scoring='roc_auc'
)
