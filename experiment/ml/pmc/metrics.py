import numpy as np
import pandas as pd
from .params import groups as GROUPS

def categorize(X, y, groups,
               n_bins=10,
               bins=None,
               alpha=0.01,
               gamma=0.01
              ):
    """Map data to an existing set of categories."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"

    categories = None 

    if bins is None:
        bins = np.linspace(1/n_bins, 1.0, n_bins)
        bins[0] = 0.0
    else:
        n_bins=len(bins)

    min_size = gamma*alpha*len(X)/n_bins

    df = X[groups].copy()
    df.loc[:,'interval'], retbins = pd.cut(y, bins, 
                                           include_lowest=True,
                                           retbins=True
                                          )

    categories = df.groupby(groups+['interval']).groups

    categories = {k:v for k,v in categories.items() 
                  if len(v) > min_size
                 } 
    
    return categories


def multicalibration_loss(
    estimator,
    X,
    y_true,
    groups=GROUPS,
    n_bins=None,
    bins=None,
    categories=None,
    proportional=False,
    alpha=0.01,
    gamma=0.1,
    rho=0.1
):
    """custom scoring function for multicalibration.
       calculate current loss in terms of (proportional) multicalibration"""
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    y_pred = estimator.predict_proba(X)[:,1]
    y_pred = pd.Series(y_pred, index=y_true.index)


    assert isinstance(y_true, pd.Series)
    assert isinstance(y_pred, pd.Series)
    loss = 0.0

    assert groups is not None, "groups must be defined."

    if categories is None:
        categories = categorize(X, y_pred, groups,
                                n_bins=n_bins,
                                bins=bins,
                                alpha=alpha, 
                                gamma=gamma
                               )

    for c, idx in categories.items():
        category_loss = np.abs(y_true.loc[idx].mean() 
                               - y_pred.loc[idx].mean()
                              )
        # print(c,len(idx),category_loss)
        if proportional: 
            category_loss /= max(y_true.loc[idx].mean(), rho)

        if  category_loss > loss:
            loss = category_loss

    return loss

def proportional_multicalibration_loss(estimator, X, y_true, **kwargs):
    kwargs['proportional'] = True
    return multicalibration_loss(estimator, X, y_true, **kwargs)

def differential_calibration(
    estimator, 
    X, 
    y_true,
    groups=GROUPS,
    n_bins=None,
    bins=None,
    categories=None,
    alpha=0.01,
    gamma=0.1,
    eps=0.01
):
    """Return the differential calibration of estimator on groups."""

    assert isinstance(X, pd.DataFrame), "X needs to be a dataframe"
    assert all([g in X.columns for g in groups]), ("groups not found in"
                                                   " X.columns")
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    y_pred = estimator.predict_proba(X)[:,1]

    if categories is None:
        categories = categorize(X, y_pred, groups,
                                n_bins=n_bins,
                                bins=bins,
                                alpha=alpha, 
                                gamma=gamma
                               )
    dc_max = 0
    for ci, i in categories.items():
        for cj, j in categories.items():
            if ci==cj: 
                continue

            yi = y_true.loc[idx].mean()
            yj = y_true.loc[idx].mean()+eps

            dc = np.abs( np.log(yi) - np.log(yj) )

            if dc > dc_max:
                dc_max = dc

    return dc_max

# def smoothed_empirical_differential_fairness(self, concentration=1.0):
#     """Smoothed EDF 
#     References:
#         .. [#foulds18] J. R. Foulds, R. Islam, K. N. Keya, and S. Pan,
#            "An Intersectional Definition of Fairness," arXiv preprint
#            arXiv:1807.08362, 2018.
#     """
#     sbr = self._smoothed_base_rates(self.dataset.labels, concentration)

#     def pos_ratio(i, j):
#         return abs(np.log(sbr[i]) - np.log(sbr[j]))

#     def neg_ratio(i, j):
#         return abs(np.log(1 - sbr[i]) - np.log(1 - sbr[j]))

#     # overall DF of the mechanism
#     return max(max(pos_ratio(i, j), neg_ratio(i, j))
#                for i in range(len(sbr)) for j in range(len(sbr)) if i != j)
