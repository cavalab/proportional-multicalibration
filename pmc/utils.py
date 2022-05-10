import numpy as np

def squash_array(x):
    x[x<0.0] == 0.0
    x[x>1.0] == 1.0
    return x

def squash_series(x):
    return x.apply(lambda x: max(x,0.0)).apply(lambda x: min(x,1.0))
