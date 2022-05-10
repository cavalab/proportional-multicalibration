"""
Proportional Multicalibration Post-processor
copyright William La Cava, Elle Lett
License: GNU GPL3
"""
import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error as mse
from copy import copy
# from utils import squash_array, squash_series
import utils

class MultiCalibrator(ClassifierMixin, BaseEstimator):
    """ A classifier post-processor that updates a model to satisfy different
    notions of fairness.

    Parameters
    ----------
    estimator : Probabilistic Classifier 
        A pre-trained classifier that outputs probabilities. 
    auditor: Classifier or callable
        Method that returns a subset of sample from the data, belonging to a 
        specific group.
    metric: 'MC' or 'PMC', default: PMC
    alpha: float, default: 0.01
        tolerance for calibration error per group. 
    n_bins: int, default: 10
        used to discretize probabilities. 
    gamma: float, default: 0.1
        the minimum probability of a group occuring in the data. 
    rho: float, default: 0.1
        the minimum risk prediction to attempt to adjust. 
        relevant for proportional multicalibration.
    max_iters: int, default: None
        maximum iterations. Will terminate whether or not alpha is achieved.
    random_state: int, default: 0
        random seed.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, 
                 estimator=None,
                 auditor=None,
                 metric='PMC',
                 alpha=0.01,
                 n_bins=10,
                 gamma=0.01,
                 rho=0.1,
                 eta=1.0,
                 max_iters=100,
                 random_state=0
                ):
        self.estimator=estimator
        self.auditor=auditor
        self.metric=metric
        self.alpha=alpha
        self.n_bins=n_bins
        self.gamma=gamma
        self.rho=rho
        self.eta=eta
        self.max_iters=max_iters
        self.random_state=random_state

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        assert len(self.classes_) == 2, "Only binary classification supported"

        y_true = y.astype(float)
        self.X_ = X
        self.y_ = y_true

        assert hasattr(self.estimator, 'predict_proba'), ("Classifier has no"
                                                    "'predict_proba' method")

        self.auditor_ = self.auditor
        for att in ['alpha','n_bins','gamma','rho','metric','random_state']:
            setattr(self.auditor_, att, getattr(self,att))
        
        # map groups to adjustments
        self.adjustments_ = {} 
        iters, n_updates = 0, 0 
        updated = True
        # predictions
        y_init = self.estimator.predict_proba(X)[:,1]
        y_init = pd.Series(y_init, index=X.index)
        y_adjusted = copy(y_init)
        print('initial MSE:', mse(y_true, y_init))
        log = dict(
            iteration=[],
            r=[],
            ybar=[],
            delta=[],
            alpha=[],
            category=[] 
        )

        self.auditor.make_categories(X, y_init)
        # bootstrap sample X,y
        bootstraps = 0
        worst_cat = None
        while iters < self.max_iters and updated == True:
            Xs, ys, ys_pred = X, y_true, y_adjusted
            # Xs, ys, ys_pred = resample(X, y_true, y_adjusted,
            #                   stratify=y_true, 
            #                   random_state=self.random_state
            #                  )
            bootstraps +=1 
            # print(f'ys balance: {ys.sum()/len(ys)}')
            updated=False
            # make an iterable over groups, intervals
            # if bootstraps==1:
            #     ys_pred = pd.Series(self.estimator.predict_proba(Xs)[:,1],
            #                         index=Xs.index)
            # else:
            #     ys_pred = pd.Series(self.predict_proba(Xs)[:,1],
            #                         index=Xs.index)
            categories = self.auditor.categorize(Xs, ys_pred)
            # if worst_cat != None:
            #     if worst_cat not in categories:
            #         ipdb.set_trace()
            progress_bar = tqdm(categories.items())
            # for category, idx in categories:
            for category, idx in progress_bar:
                # if category == worst_cat:
                #     ipdb.set_trace()
                # calc average risk prediction 
                r = ys_pred.loc[idx]
                rbar = np.mean(r)
                # calc actual average risk for the group
                # ipdb.set_trace()
                ybar = ys.loc[idx].sum()/len(ys.loc[idx])
                # if ybar is too small, skip
                if ybar < self.rho:
                    continue
                # delta 
                delta = ybar - rbar
                # check 
                alpha = self.alpha if self.metric=='MC' else self.alpha*ybar
                # if category == worst_cat:
                #     print(
                #           f'category:{category},'
                #           # f'prediction: {r:3f}',
                #           f'rbar:{rbar:3f}',
                #           f'ybar:{ybar:3f}',
                #           f'delta:{delta:3f}',
                #           f'alpha:{alpha:3f}'
                #          )
                if np.abs(delta) > alpha:
                    update = self.eta*delta
                    if category == worst_cat:
                        print('updating', category)
                    # print(f'category size: {len(idx)}')
                    # print(f'delta ({delta:.3f}) > alpha ({alpha:.2f})')
                    # print(f'update:{update:3f}')
                    # update estimates 
                    # ipdb.set_trace()
                    y_unadjusted = y_adjusted.loc[idx].copy()
                    y_adjusted.loc[idx] += update
                    y_adjusted.loc[idx] = utils.squash_series(y_adjusted.loc[idx])
                    squashed_update = y_adjusted.loc[idx].mean() - y_unadjusted.mean()  

                    # if np.abs(squashed_update - update) > 0.001:
                    #     print(f'squashed_update:{squashed_update:.4f}',
                    #           f'update: {update:.4f}')

                    if category in self.adjustments_.keys():
                        self.adjustments_[category] += squashed_update
                    else:
                        self.adjustments_[category] = squashed_update

                    updated=True
                    n_updates += 1

                    assert y_adjusted.max() <= 1.0 and y_adjusted.min() >= 0.0
                    # make sure update was good
                    r1 = y_adjusted.loc[idx]
                    rbar1 = r1.mean()
                    ybar1 = ys.loc[idx].sum()/len(ys.loc[idx])
                    # if ( np.abs(ybar1 - rbar1) >= np.abs(ybar-rbar) ):
                    #     ipdb.set_trace()
                    # new_pred = pd.Series(self.predict_proba(X)[:,1],
                    #                      index=X.index)
                    MSE = mse(y_true, y_adjusted)
                     
                    cal_loss, worst_cat = self.auditor_.loss(y_true, y_adjusted)
                    # cal_loss, worst_cat = self.auditor_.loss(ys, ys_pred)
                    progress_bar.set_description(
                                                 f'categories:{len(categories)}, '
                                                 f'updates:{n_updates}, '
                                                 f'{self.metric}:{cal_loss:.3f}, '
                                                 f'MSE:{MSE:.3f} '
                                                )
                iters += 1
                if iters >= self.max_iters: 
                    print('max iters reached')
                    break
            if not updated:
                print('no updates this round')
                break
        print('finished. updates:', n_updates)
        print('initial multicalibration:', self.auditor_.loss(y_true, y_init,X)[0])
        print('final multicalibration:', self.auditor_.loss(y_true,
                                                            y_adjusted,X)[0])
        print('adjustments:',self.adjustments_)
        # Return the classifier
        return self
    

    def predict_proba(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check if fit had been called
        # check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        # X = check_array(X)

        # y_pred = self.estimator.predict_proba(X)[:,1]

        y_pred = pd.Series(self.estimator.predict_proba(X)[:,1],
                            index=X.index)
        
        categories = self.auditor_.categorize(X, y_pred)

        for category, adjustment in self.adjustments_.items(): 
            if category in categories.keys():
                y_pred.loc[categories[category]] += adjustment 
            # else:
            #     print('warn: y_pred missing category',category)

        y_pred = utils.squash_series(y_pred)
        # ipdb.set_trace()
        rety = np.vstack((1-y_pred, y_pred)).T
        return rety

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.predict_proba(X)[:,1] > 0.5
