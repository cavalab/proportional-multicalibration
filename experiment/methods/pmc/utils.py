import numpy as np
import pandas as pd
import ipdb

def squash_array(x):
    x[x<0.0] == 0.0
    x[x>1.0] == 1.0
    return x

def squash_series(x):
    return x.apply(lambda x: max(x,0.0)).apply(lambda x: min(x,1.0))

def category_diff(cat1, cat2):
    different=False
    for k1,v1 in cat1.items():
        if k1 not in cat2.keys():
            print(f'{k1} not in cat2')
            different=True
        else:
            if not v1.equals(cat2[k1]):
                print(f'indices for {k1} different in cat2')
                different=True
    for k2,v2 in cat2.items():
        if k2 not in cat1.keys():
            print(f'{k1} not in cat1')
            different=True
        else:
            if not v2.equals(cat1[k2]):
                print(f'indices for {k2} different in cat1')
                different=True
    if not different:
        # print('categories match.')
        return True
    else:
        return False

def differential_calibration(X, y, groups, gamma=0.01):
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"

    df = X[groups]

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
                                         retbins=True)

    categories = df.groupby(groups+['interval']).groups

    categories = {k:v for k,v in categories.items() 
                  if len(v) > min_size
                 } 
    
    return categories


def MC_loss(y_true, y_pred, 
            X=None, 
            categories=None,
            groups=None,
            n_bins=None,
            bins=None,
            return_cat=False,
            proportional=False,
            alpha=0.01,
            gamma=0.1,
            rho=0.01
           ):
        """calculate current loss in terms of multicalibration or PMC"""
        assert isinstance(y_true, pd.Series)
        assert isinstance(y_pred, pd.Series)
        loss = 0.0

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
                # worst = (c, idx)
                # worstc = c
                # worstidx = idx
        # print('worst category:',worst[0],'size:',len(worst[1]),'loss:',alpha)
        # if return_cat:
        #     return alpha, worstc, worstidx, categories
        # else:
        #     return alpha, worstc, worstidx
        return loss
