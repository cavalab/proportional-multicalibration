import xgboost

hyper_params = [
    {
        'n_estimators' : (100,500),
        'learning_rate' : (0.0001,0.01, 0.1 ),
        'gamma' : (0.1,0.4),
        'subsample' : (0.5, 1),
    },
]

est=xgboost.XGBClassifier(
    max_depth=6, 
    eval_metric='logloss',
    use_label_encoder=False
)
