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
from sklearn.metrics import r2_score
from copy import copy
# from utils import squash_array, squash_series
import utils
import logging
# logging.basicConfig(format='%(asctime)s %(message)s',
#                     datefmt='%m/%d/%Y %I:%M:%S %p',
#                    )
logger = logging.getLogger(__name__)

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
                 random_state=0,
                 verbosity=0
                ):
        self.estimator=estimator
        self.auditor_type=auditor
        self.metric=metric
        self.alpha=alpha
        self.n_bins=n_bins
        self.gamma=gamma
        self.rho=rho
        self.eta=eta
        self.max_iters=max_iters
        self.random_state=random_state
        self.verbosity=verbosity

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
        logger.setLevel({0:logging.WARN, 
                         1:logging.INFO, 
                         2:logging.DEBUG}
                        [self.verbosity]
                       )

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

        self.auditor_ = copy(self.auditor_type)
        for att in vars(self):
            if hasattr(self.auditor_, att):
                setattr(self.auditor_, att, getattr(self,att))
        
        # map groups to adjustments
        self.adjustments_ = [] 
        iters, n_updates = 0, 0 
        updated = True
        # predictions
        y_init = self.estimator.predict_proba(X)[:,1]
        y_init = pd.Series(y_init, index=X.index)
        y_adjusted = copy(y_init)
        MSE = mse(y_true, y_init)
        print('initial MSE:', MSE)
        log = dict(
            iteration=[],
            r=[],
            ybar=[],
            delta=[],
            alpha=[],
            category=[] 
        )

        self.auditor_.make_categories(X, y_init)
        print(f'categories:{len(self.auditor_.categories_)}')
        # bootstrap sample X,y
        bootstraps = 0
        worst_cat = None
        Xs, ys = X, y_true
        # while iters < self.max_iters and updated == True:
        init_cal_loss, _, _ = self.auditor_.loss(
                                                 y_true, 
                                                 y_adjusted,
                                                 X
                                                )
        # prev_cal_loss = init_cal_loss
        progress_bar = tqdm(total=100, unit='%')
        for i in range(self.max_iters):
            # Xs, ys, ys_pred = X, y_true, y_adjusted
            ys_pred = y_adjusted.copy()
            Xs = X.copy()

            MSE = mse(y_true, y_adjusted)
            cal_loss, prev_worst_c, prev_worst_idx, cats =  \
                    self.auditor_.loss(ys, ys_pred, Xs, return_cat=True)

            progress_bar.set_description(
                                         f'updates:{n_updates}, '
                                         f'{self.metric}:{cal_loss:.3f}, '
                                         f'MSE:{MSE:.3f} '
                                        )
            # Xs, ys, ys_pred = resample(X, y_true, y_adjusted,
            #                   stratify=y_true, 
            #                   random_state=self.random_state
            #                  )
            # print('----------------',iters,'-------------------')
            # print('initial worst category from auditor:',
            #       prev_worst_c, 'alpha = ', prev_cal_loss)
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
            categories = self.auditor_.categorize(Xs, ys_pred)
            assert utils.category_diff(categories, cats), ("categories don't"
                                                           "match")

            assert prev_worst_c in categories.keys()

            Mworst_delta = 0
            pmc_adjust = 1

            for category, idx in tqdm(categories.items(), leave=False):
                # calc average predicted risk for the group
                rbar = ys_pred.loc[idx].mean()
                # calc actual average risk for the group
                ybar = ys.loc[idx].mean()

                if self.metric=='PMC':
                    pmc_adjust = max(ybar,self.rho)

                # delta 
                delta = ybar - rbar
                
                # set alpha 
                alpha = self.alpha  
                if self.metric=='PMC':
                    alpha *= pmc_adjust

                logger.debug(
                      f'category:{category}, '
                      f'rbar:{rbar:3f}, '
                      f'ybar:{ybar:3f}, '
                      f'delta:{delta:3f}, '
                      f'alpha:{alpha:3f}, '
                      f'delta/pmc_adjust:{np.abs(delta)/pmc_adjust:.3f}'
                     )

                if ((self.metric=='MC' and np.abs(delta) > Mworst_delta)
                    or (self.metric=='PMC' 
                        and np.abs(delta)/pmc_adjust > Mworst_delta)):
                    Mworst_delta=np.abs(delta) 
                    Mworst_c = category
                    Mworst_idx = idx
                    if self.metric=='PMC':
                        Mworst_delta /= pmc_adjust

                if np.abs(delta) > alpha:
                    update = self.eta*delta

                    logger.debug(f'Updating category:{category}')
                    # update estimates 
                    y_unadjusted = y_adjusted.copy()
                    y_adjusted.loc[idx] += update

                    if updated == False:
                        self.adjustments_.append({})

                    self.adjustments_[-1][category] = update

                    updated=True
                    n_updates += 1

                    # make sure update was good
                    rnew = y_adjusted.loc[idx]
                    rbarnew = rnew.mean()

                    assert not any(y_adjusted.isna())

                    # cal_loss, worst_cat = self.auditor_.loss(ys, ys_pred)
                iters += 1
                if iters >= self.max_iters: 
                    print('max iters reached')
                    logger.warn('max_iters was reached before alpha termination'
                                ' criterion was satisfied.')
                    break

            y_adjusted = utils.squash_series(y_adjusted)
            assert y_adjusted.max() <= 1.0 and y_adjusted.min() >= 0.0
            # MSE = mse(y_true, y_adjusted)
            # prev_cal_loss, prev_worst_c, prev_worst_idx =  self.auditor_.loss(y_true, y_unadjusted, X)
            new_cal_loss, worst_c, worst_idx = self.auditor_.loss(
                                                     y_true, 
                                                     y_adjusted,
                                                     X
                                                    )
            logger.debug(f'worst category from multicalibrator: '
                         f'{Mworst_c}, alpha = {Mworst_delta}')
            logger.debug(f'worst category from auditor: '
                         f'{worst_c}, alpha = {new_cal_loss}')
            # ipdb.set_trace()
            if not updated:
                print('no updates this round')
                break
            else:
                cal_diff = cal_loss - new_cal_loss
                # if cal_diff < 0:
                #     ipdb.set_trace()
                progress_bar.update(round(100*cal_diff/init_cal_loss))
        ## end for loop
        ######################################## 
        print('finished. updates:', n_updates)
        y_end = pd.Series(self.predict_proba(X)[:,1], index=X.index)
        print('mse:',mse(y_adjusted, y_end))
        print('r2:',r2_score(y_adjusted, y_end))
        # assert np.equal(y_adjusted.values.round(5), y_end.round(5) ).all()
        np.testing.assert_allclose(y_adjusted, y_end, rtol=1e-04)
        init_MC = self.auditor_.loss(y_true, y_init,X, metric='MC')[0]
        final_MC = self.auditor_.loss(y_true, y_end,X, metric='MC')[0]
        init_PMC = self.auditor_.loss(y_true, y_init,X, metric='PMC')[0]
        final_PMC = self.auditor_.loss(y_true, y_end,X, metric='PMC')[0]
        print(f'initial multicalibration: {init_MC}')
        print(f'final multicalibration: {final_MC}')
        print(f'initial proportional multicalibration: {init_PMC}')
        print(f'final proportional multicalibration: {final_PMC}')
        # print('adjustments:',self.adjustments_)

        #     print(y_adjusted - y_end) 
        #     print('mse:',mse(y_adjusted, y_end))
        #     print('r2:',r2_score(y_adjusted, y_end))
        #     ipdb.set_trace()
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
        

        for adjust_iter in self.adjustments_:
            categories = self.auditor_.categorize(X, y_pred)
            for category, update in adjust_iter.items(): 
                if category in categories.keys():
                    idx = categories[category]
                    y_pred.loc[idx] += update
                    # y_pred.loc[idx] = utils.squash_series(y_pred.loc[idx])
                # else:
                #     logger.warn(f'y_pred missing category {category}')
            y_pred = utils.squash_series(y_pred)

        # y_pred = utils.squash_series(y_pred)
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
