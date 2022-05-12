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
from methods.pmc.utils import MC_loss
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
from util import jsonify
def evaluate_model(dataset, results_path, random_state, est_name, est, 
                   hyper_params, complexity, model,  n_splits = 5,
                   n_samples=0, scale_x = True, groups = ['ethnicity','gender'],
                   pre_train=None):

    print(40*'=','Evaluating '+est_name+' on ',dataset,40*'=',sep='\n')

    np.random.seed(random_state)
    if hasattr(est, 'random_state'):
        est.random_state = random_state

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
                                                    random_state=random_state)

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
        'algorithm':est_name,
        'params':jsonify(best_est.get_params()),
        'random_state':random_state,
        'process_time': process_time, 
        'time_time': time_time, 
    }

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
                print(score + '_' + fold,scorer(target, y_pred_proba) )
            for score, scorer in [('accuracy',accuracy_score)]:
                results[score + '_' + fold] = scorer(target, y_pred) 
                print(score + '_' + fold,scorer(target, y_pred) )
            
            y_pred_proba = pd.Series(y_pred_proba, index=target.index)
            X = X.set_index(target.index)
            results['MC_loss_' + fold] = MC_loss(target, y_pred_proba, 
                                                 X=X, 
                                                 groups=GROUPS,
                                                 n_bins=10,
                                                 # bins=None,
                                                 # return_cat=False, 
                                                 proportional=False,
                                                 alpha=0.01,
                                                 gamma=0.1,
                                                 rho=0.01
                                                )
            print('MC_loss_' + fold,results['MC_loss_' + fold])
            results['PMC_loss_' + fold] = MC_loss(target, y_pred_proba, 
                                                 X=X, 
                                                 groups=GROUPS,
                                                 n_bins=10,
                                                 # bins=None,
                                                 # return_cat=False, 
                                                 proportional=True,
                                                 alpha=0.01,
                                                 gamma=0.1,
                                                 rho=0.01
                                                )
            print('PMC_loss_' + fold,results['MC_loss_' + fold])
            # TODO: add differential fairness loss

    
    if hasattr(est, 'feature_importances_'):
        results['feature_importances_'] = \
                {fn:imp 
                 for fn,imp in zip(X_train.columns, est.feature_importances_)}
        print('feature importances:',results['feature_importances_'])

    ##############################
    # write to file
    ##############################
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file = (results_path + '/' + dataset_name + '_' + est_name + '_' 
                 + str(random_state))

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

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "y". '
                        'If you use the preprocessing file, you do not need to do anything')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-results_path', action='store', dest='RDIR',
                        default='results', type=str, 
                        help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=42, type=int, help='Seed / trial')
    parser.add_argument('-split', action='store', dest='NSPLIT',
                        default=5, type=int, help='Number of Split for Cross Validation')
    # parser.add_argument('-test',action='store_true', dest='TEST', 
    #                    help='Used for testing a minimal version')
    # parser.add_argument('-skip_tuning',action='store_true', dest='SKIP_TUNE', 
    #                     default=False, help='Dont tune the estimator')

    args = parser.parse_args()
    # import algorithm 
    print('import from','methods.'+args.ALG)
    algorithm = importlib.__import__('methods.'+args.ALG,
                                     globals(),
                                     locals(),
                                     ['*']
                                    )

    print('algorithm:',algorithm.est)
    if 'hyper_params' not in dir(algorithm):
        algorithm.hyper_params = {}
    print('hyperparams:',algorithm.hyper_params)

    # optional keyword arguments passed to evaluate
    eval_kwargs = {}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    # check for conflicts btw cmd line args and eval_kwargs
    # if args.SKIP_TUNE:
    #     eval_kwargs['skip_tuning'] = True

    evaluate_model(args.INPUT_FILE, args.RDIR, args.RANDOM_STATE, args.ALG,
                   algorithm.est, algorithm.hyper_params, algorithm.complexity,
                   algorithm.model, n_splits= args.NSPLIT,
                   **eval_kwargs)
