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
                 gamma=0.1,
                 rho=0.1,
                 metric=None,
                 random_state=0
                ):
        self.estimator=estimator
        self.groups = groups
        self.alpha=alpha
        self.n_bins=n_bins
        self.gamma=gamma
        self.rho=rho
        self.metric=metric
        self.random_state=random_state

    def categorize(self, X, y, y_true=None):
        """Map data to an existing set of categories."""

        df = X #.copy()
        df['interval']  = pd.cut(y, self.bins_)

        categories = df.groupby(self.grouping_, sort=False).groups

        # min_size = self.gamma*self.alpha*len(X)/self.n_bins
        categories = {k:v for k,v in categories.items() 
                      if len(v) > self.min_size_
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
        # bins = [(i,i+1) for i in np.linspace(0.0,1.0, self.n_bins)[:-1]]
        # bins = np.linspace(0.0+1/self.n_bins,1.0, self.n_bins)
        # print('bins:',bins)

        self.categories_ = None 
        df = X #.copy()
        df['interval'], bins = pd.cut(y, self.n_bins, retbins=True)
        self.bins_ = bins
        print('bins:',self.bins_)
        self.grouping_ = self.groups+['interval']
        self.min_size_ = self.gamma*self.alpha*len(X)/self.n_bins

        self.categories_ = self.categorize(X,y)

        # categories = df.groupby(self.groups+['interval'], sort=False).groups
        # self.all_categories_ = categories.copy()
        # filter groupings less than gamma * N * alpha * lambda
        # # if threshold:
        # categories = {k:v for k,v in categories.items() 
        #               if len(v) > self.min_size_ 
        #               # if (len(v) > min_size 
        #                   # and y.loc[k].mean() > self.rho)
        #              } 
        # self.categories_ = categories 
        # for (bmin,bmax) in bins:
        #     interval = lambda y: y > bmin and y <= bmax
        #     for g,idx in subgroups:
        #         group = lambda x: all([x[i]==gi for i,gi in
        #                                zip(self.groups,g)])

        #         category = lambda x,y: group(x) and interval(y)
                
        #         subset = [category(xi,yi) for xi,yi in zip(X, y)]
                
        #         idx = np.asarray(subset).nonzero()
        #         if len(idx) > self.gamma * self.alpha * 1/self.n_bins * N:
        #             self.categories_.append( (category, idx) )
        return self.categories_ #.items()

    def loss(self, y_true, y_pred):
        """calculate current loss in terms of multicalibration or PMC"""
        alpha = 0.0
        worst = None 
        # min_size = self.gamma*self.alpha*self.N_/self.n_bins

        # categories = self.categorize(X, y_pred)

        for c, idx in self.categories_.items():
            if len(idx) < self.min_size_ or y_pred.iloc[idx].mean() < self.rho:
                continue
            category_loss = np.abs(y_true.iloc[idx].mean() 
                                   - y_pred.iloc[idx].mean()
                                  )
            if self.metric=='PMC': 

                category_loss /= max(y_true.iloc[idx].sum()/len(y_true),
                                     self.rho)
            if  category_loss > alpha:
                alpha = category_loss
                worst = (c, idx)
        # print('worst category:',c,'size:',len(idx),'loss:',alpha)
        return alpha, c

