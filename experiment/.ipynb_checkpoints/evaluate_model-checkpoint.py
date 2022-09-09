from datetime import datetime
import sys
import ipdb
import itertools
import pandas as pd
from sklearn.base import clone
# from sklearn.experimental import enable_halving_search_cv # noqa
# from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             average_precision_score)
# from ml.pmc.utils import MC_loss
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from read_file import read_file
import pdb
import numpy as np
import json
import os
import inspect
from util import jsonify, hasattranywhere
from ml.pmc.auditor import Auditor
from ml.pmc.params import (groups, Alphas, Gammas, N_binses, Rhos)
from ml.pmc.metrics import (differential_calibration, 
                            multicalibration_score,
                            proportional_multicalibration_score,
                            multicalibration_loss,
                            proportional_multicalibration_loss,
                           )

def evaluate_model(
    dataset, 
    results_path, 
    random_state, 
    ml, 
    est, 
    alpha,
    n_bins,
    gamma,
    rho,
    n_samples=0, 
    scale_x = False, 
    pre_train=None
):
    """Main evaluation routine."""

    ########################################
    # configure estimators
    ########################################
    setatts = {
        'random_state':random_state, 
        'alpha':alpha,
        'n_bins':n_bins,
        'gamma':gamma,
        'rho':rho,
    }
    np.random.seed(random_state)
    for k,v in setatts.items():
        if hasattr(est, k):
            setattr(est, k, v)
    # if hasattr(est, 'n_jobs'):
    #     est.n_jobs = 1
    if groups is not None:
        if hasattr(est, 'auditor_type'):
            est.auditor_type = Auditor(groups=groups)
    # if 'pmc' in ml and hasattr(est, 'scoring'):
    #     est.scoring = (lambda est,x,y:
    #               proportional_multicalibration_score(
    #                   est,x,y,
    #                   alpha=alpha,
    #                   n_bins=n_bins,
    #                   gamma=gamma,
    #                   rho=rho
    #               )
    #              )
    # elif 'mc' in ml and hasattr(est, 'scoring'):
    #     est.scoring = (lambda est,x,y:
    #               multicalibration_score(
    #                   est,x,y,
    #                   alpha=alpha,
    #                   n_bins=n_bins,
    #                   gamma=gamma
    #               )
    #              )
    # attrs = hasattranywhere(est, 'auditor_type')
    # for a in attrs: 
    #     setattr(est,a,Auditor(groups=groups))
    print(40*'=','Evaluating '+ml+' on ',dataset,40*'=',sep='\n')

    ##################################################
    # setup data
    ##################################################
    features, labels = read_file(dataset)
    print('features:')
    print(features.head())
    print(features.shape)
    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=random_state
                                                    )                                                      

    # if dataset is large, subsample the training set 
    if n_samples > 0 and len(labels) > n_samples:
        print('subsampling training data from',len(X_train),'to',n_samples)
        sample_idx = np.random.choice(np.arange(len(X_train)), size=n_samples)
        X_train = X_train[sample_idx]
        y_train = y_train[sample_idx]

    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler() 
        X_train_scaled = pd.DataFrame(sc_X.fit_transform(X_train),
                                      columns=X_train.columns)
        X_test_scaled = pd.DataFrame(sc_X.transform(X_test),
                                      columns=X_test.columns)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test


    # run any method-specific pre_train routines
    if pre_train:
        pre_train(est, X_train_scaled, y_train)

    print('X_train:',X_train_scaled.shape)
    print('y_train:',y_train.shape)
    

    ################################################## 
    # Fit models
    ################################################## 
    print('training',est)
    t0p = time.process_time()
    t0t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X_train_scaled, y_train)
    process_time = time.process_time() - t0p
    time_time = time.time() - t0t
    print('Training time measures:',process_time, time_time)
    
    ##################################################
    # store results
    ##################################################
    dataset_name = dataset.split('/')[-1].split('.')[0]
    results = {
        'dataset':dataset,
        'algorithm':ml,
        'params':jsonify(est.get_params()),
        'process_time': process_time, 
        'time_time': time_time, 
    }
    results.update(setatts)
        # 'random_state':random_state,
        # 'alpha': alpha,
        # 'n_bins': n_bins,
        # 'gamma': gamma,
        # 'rho': rho
    # }

    ##############################
    # scores
    ##############################
    for fold, target, X in zip(['train','test'],
                               [y_train, y_test], 
                               [X_train_scaled, X_test_scaled]
                              ):
            y_pred = est.predict(X).reshape(-1,1)
            y_pred_proba = est.predict_proba(X)[:,1]
            for score, scorer in [('roc_auc',roc_auc_score),
                                  ('auprc',average_precision_score)]:
                # ipdb.set_trace()
                results[score + '_' + fold] = scorer(target, y_pred_proba) 
                print(score + '_' + fold,
                      f'{scorer(target, y_pred_proba):.3f}')
            for score, scorer in [('accuracy',accuracy_score)]:
                results[score + '_' + fold] = scorer(target, y_pred) 
                print(score + '_' + fold,
                      f'{scorer(target, y_pred):.3f}')
            
            y_pred_proba = pd.Series(y_pred_proba, index=target.index)
            X = X.set_index(target.index)
            # MC
            results['MC_loss_' + fold] = multicalibration_loss(
                estimator=est,
                X=X, 
                y_true=target, 
                groups=groups,
                n_bins=n_bins,
                alpha=alpha,
                gamma=gamma,
                rho=rho,
                proportional=False,
            )
            print('MC_loss_' + fold,
                  f"{results['MC_loss_' + fold]:.3f}")
            # PMC
            results['PMC_loss_' + fold] = \
                    proportional_multicalibration_loss(
                estimator=est,
                X=X, 
                y_true=target, 
                groups=groups,
                n_bins=n_bins,
                proportional=True,
                alpha=alpha,
                gamma=gamma,
                rho=rho
            )
            print('PMC_loss_' + fold,
                  f"{results['PMC_loss_' + fold]:.3f}")
            # differential calibration loss
            results['DC_loss_' + fold] = differential_calibration(
                estimator=est,
                y_true=target, 
                X=X, 
                groups=groups,
                n_bins=n_bins,
                alpha=alpha,
                gamma=gamma,
                rho=rho
            )
            print('DC_loss_' + fold,
                  f"{results['DC_loss_' + fold]:.3f}")

    
    if hasattr(est, 'feature_importances_'):
        results['feature_importances_'] = \
                {fn:imp 
                 for fn,imp in zip(X_train.columns, est.feature_importances_)}
        print('feature importances:',results['feature_importances_'])

    ##############################
    # write to file
    ##############################
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    save_file = os.path.join(results_path, '_'.join([f'{n}' for n in [
        dataset_name,
        ml,
        random_state,
        os.environ['LSB_JOBID'] if 'LSB_JOBID' in os.environ.keys() else '',
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        ]
        ]
        )
    )

    # save_file = (results_path + '/' + dataset_name + '_' + ml + '_' 
    #              + str(random_state))

    print('save_file:',save_file)

    with open(save_file + '.json', 'w') as out:
        json.dump(jsonify(results), out, indent=4)

    # store CV detailed results
    # turning off for now as I dont think we'll need this for our analysis
    # cv_results = grid_est.cv_results_
    # cv_results['random_state'] = random_state

    # with open(save_file + '_cv_results.json', 'w') as out:
    #     json.dump(jsonify(cv_results), out, indent=4)

################################################################################
# main entry point
################################################################################
import argparse
import importlib
import logging
import sys

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stdout, level=logging.INFO) 

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('-file', action='store', type=str,
                        default='data/mimic4_admissions.csv',
                        help='Data file to analyze; ensure that the '
                             'target/label column is labeled as "y". '
                             'If you use the preprocessing file, '
                             'you do not need to do anything')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', default='xgb',type=str, 
            help='Name of estimator (with matching file in ml/)')
    parser.add_argument('-results_path', action='store', dest='RDIR',
                        default='../results', type=str, 
                        help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=42, type=int, help='Seed / trial')
    parser.add_argument('-alpha', action='store', default=0.01, type=float, 
                        help='Calibration tolerance (for metrics)')
    parser.add_argument('-n_bins', action='store', default=10, type=int, 
                        help='Number of bins to consider for calibration')
    parser.add_argument('-gamma', action='store', default=0.05, type=float, 
                        help='Min subpop prevalence (for metrics)')
    parser.add_argument('-rho', action='store', default=0.1, type=float, 
                        help='Min subpop prevalence (for metrics)')
    args = parser.parse_args()
    # import algorithm 
    print('import from','ml.'+args.ml)
    algorithm = importlib.__import__('ml.'+args.ml,
                                     globals(),
                                     locals(),
                                     ['*']
                                    )

    print('algorithm:',algorithm.est)

    # optional keyword arguments passed to evaluate
    eval_kwargs = {}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    # check for conflicts btw cmd line args and eval_kwargs
    # if args.SKIP_TUNE:
    #     eval_kwargs['skip_tuning'] = True

    evaluate_model(
        dataset=args.file, 
        results_path=args.RDIR,
        random_state=args.RANDOM_STATE,
        ml=args.ml,
        est=algorithm.est, 
        alpha=args.alpha,
        n_bins=args.n_bins,
        gamma=args.gamma,
        rho=args.rho,
        **eval_kwargs
    )
