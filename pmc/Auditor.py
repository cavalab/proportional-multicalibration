import pandas as pd
import numpy as np
import ipdb

class Auditor():
    """A class that determines and manages group membership over which to assess
    multicalibration.

    Parameters
    ----------
    estimator : Regessor or None, default: None 
        Optionally specify an ML method to determine which group to return.
    groups: list of str, default: None
        Specify a list of sensitive attributes to use as groups, instead of
        using an estimator. 
    metric: 'MC' or 'PMC', default: PMC
    alpha: float, default: 0.01
        tolerance for calibration error per group. 
    n_bins: int, default: 10
        used to discretize probabilities. 
    gamma: float, default: 0.1
        the minimum probability of a group occuring in the data. 
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
                 groups=None,
                 alpha=0.01,
                 n_bins=10,
                 bins=None,
                 gamma=0.1,
                 rho=0.1,
                 metric=None,
                 random_state=0
                ):
        self.estimator=estimator
        self.groups = groups
        self.alpha=alpha
        self.n_bins=n_bins
        self.bins=bins
        self.gamma=gamma
        self.rho=rho
        self.metric=metric
        self.random_state=random_state

    def categorize(self, X, y, y_true=None):
        """Map data to an existing set of categories."""

        df = X[self.groups].copy()
        df['interval'],bins  = pd.cut(y, self.bins_, include_lowest=True,
                                 retbins=True)
        assert all([i==j for i,j in zip(bins,self.bins_)])

        categories = df.groupby(self.grouping_).groups
        # categories = df.groupby(self.grouping_, sort=False).groups
        min_size = self.gamma*self.alpha*len(X)/self.n_bins_

        categories = {k:v for k,v in categories.items() 
                      if len(v) > min_size
                     } 

        return categories

    def make_categories(self, X, y, threshold=True):
        """Define categories on data. 

        group:
            a subset of individuals in the population.
        interval:
            a risk prediction interval in (0, 1]. 
        category: a category is a pair of (subgroup, risk interval). 
            we represent these as a pair lambda functions that return True
            if a given sample is in the cateogry. 
        """
        assert isinstance(X, pd.DataFrame), "X should be a dataframe"
        self.N_ = len(X)

        self.categories_ = None 
        # df = X.copy()

        if self.bins is None:
            self.n_bins_ = self.n_bins
            self.bins_ = np.linspace(1/self.n_bins, 1.0, self.n_bins)
            self.bins_ = np.insert(self.bins_, 0, 0.0)
        else:
            self.bins_ = self.bins
            if self.bins_[0] > 0.0:
                self.bins_[0] = 0.0
                self.bins_ = np.insert(self.bins_, 0, 0.0)
            if self.bins_[-1] < 1.0:
                self.bins_ = np.concatenate((self.bins_, 1.0))
            self.n_bins_ = len(self.bins_)-1


        print('self.bins_:',self.bins_)
        # interval, retbins = pd.cut(y, self.bins_, include_lowest=True, 
        #                         retbins=True)
        # interval, bins = pd.cut(y, bins, include_lowest=True, retbins=True)
        # self.bins_ = bins
        # self.bins_[0] = 0.0
        # self.bins_[-1] = 1.0
        self.grouping_ = self.groups+['interval']
        # self.min_size_ = self.gamma*self.alpha*len(X)/self.n_bins_

        self.categories_ = self.categorize(X,y)

        # print('min_size:',self.min_size_)

        return self.categories_ 

    def loss(self, y_true, y_pred, X=None, return_cat=False,
             metric=None):
        """calculate current loss in terms of multicalibration or PMC"""
        metric = self.metric if metric == None else metric
        alpha = 0.0
        worst = None 
        categories = self.categorize(X, y_pred)

        for c, idx in categories.items():

            category_loss = np.abs(y_true.loc[idx].mean() 
                                   - y_pred.loc[idx].mean()
                                  )
            # print(c,len(idx),category_loss)
            if metric=='PMC': 
                category_loss /= max(y_true.loc[idx].mean(), self.rho)

            if  category_loss > alpha:
                alpha = category_loss
                worst = (c, idx)
                worstc = c
                worstidx = idx
        # print('worst category:',worst[0],'size:',len(worst[1]),'loss:',alpha)
        if return_cat:
            return alpha, worstc, worstidx, categories
        else:
            return alpha, worstc, worstidx

