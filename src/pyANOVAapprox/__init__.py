import numpy as np
from scipy.sparse.linalg import lsqr
from math import acos, isnan
from scipy.special import erf
from sklearn.metrics import roc_auc_score
import threading



from GroupedTransforms import *


def bisection(l, r, fun, maxiter=1000):   
    lval = fun(l)
    rval = fun(r)

    if np.sign(lval) * np.sign(rval) == 1:
        raise ValueError("bisection: root is not between l and r")

    if lval > 0:
        gun = fun
        fun = lambda t: -gun(t)

    m = 0.0
    for _ in range(maxiter):
        m = (l + r) / 2
        mval = fun(m)
        if abs(mval) < 1e-16:
            break
        if mval < 0:
            l = m
            lval = mval
        else:
            r = m
            rval = mval

    return m


from .fista import *
from .approx import *
from .analysis import *
from .errors import *
from .trafos import *    


# Export functions and classes:
__all__ = [
    #from Analysis.py:
    "get_variances",     
    "get_GSI",
    "get_AttributeRanking",
    "get_ShapleyValues",
    
    #from approx.py:
    "approx",                   

    #from Errors.py:
    "get_l2_error",               
    "get_mse",
    "get_mad",
    "get_L2_error",
    "get_acc",
    "get_auc",
    
    #from trafo.py:
    "transform_cube",
    "transform_R",

    #from fista.py:
    "bisection",
    "newton",
    "λ2ξ",
    "fista"
]

