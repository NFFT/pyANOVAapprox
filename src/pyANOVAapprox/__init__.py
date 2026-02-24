import math
import threading
from math import acos, isnan

import matplotlib.pyplot as plt
import numpy as np
from pyGroupedTransforms import *
from scipy.optimize import bisect
from scipy.sparse.linalg import lsqr
from scipy.special import erf

# from sklearn.metrics import roc_auc_score


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


# from .analysis import *
from .approx import *
# from .errors import *
# from .fista import *
from .trafos import *

# from .bandwidth import *


def approximate(
    a, lam=None, max_iter=50, weights=None, verbose=False, solver=None, tol=1e-8
):
    a.approximate(lam, max_iter, weights, verbose, solver, tol)


def autoapproximate(
    a,
    settingnr=None,
    B=None,
    maxiter=2,
    lmbda=0.0,
    solver="lsqr",
    verbose=False,
    solver_max_iter=50,
    solver_weights=None,
    solver_verbose=False,
    solver_tol=1e-8,
):
    a.autoapproximate(settingnr, B, maxiter, lmbda, solver, verbose)


def evaluate(a, settingnr=None, lam=None, X=None):
    return a.evaluate(settingnr, lam, X)


def evaluateANOVAterms(a, settingnr=None, X=None, lam=None):
    return a.evaluateANOVAterms(settingnr, X, lam)


def evaluateSHAPterms(a, settingnr=None, X=None, lam=None):
    return a.evaluateSHAPterms(settingnr, X, lam)


def get_l2_error(a, settingnr=None, X=None, y=None, lam=None):
    return a.get_l2_error(settingnr, X, y, lam)


def get_mse(a, settingnr=None, X=None, y=None, lam=None):
    return a.get_mse(settingnr, X, y, lam)


def get_mad(a, settingnr=None, X=None, y=None, lam=None):
    return a.get_mad(settingnr, X, y, lam)


def get_L2_error(a, norm, bc_fun, settingnr=None, lam=None):
    return a.get_L2_error(settingnr, norm, bc_fun, lam)


def get_acc(a, settingnr=None, X=None, y=None, lam=None):
    return a.get_acc(settingnr, X, y, lam)


def get_auc(a, settingnr=None, X=None, y=None, lam=None):
    return a.get_auc(settingnr, X, y, lam)


def estimate_rates(a, lam, settingnr=None, verbose=False):
    return a.estimate_rates(lam, settingnr, verbose)


def get_variances(a, settingnr=None, lam=None, Dict=False):
    return a.get_variances(settingnr, lam, Dict)


def get_GSI(a, settingnr=None, lam=None, Dict=False):
    return a.get_GSI(settingnr, lam, Dict)


def get_AttributeRanking(a, settingnr=None, lam=None):
    return a.get_AttributeRanking(settingnr, lam)


def get_ActiveSet(a, eps, settingnr=None, lam=None):
    return a.get_ActiveSet(eps, settingnr, lam)


def get_ShapleyValues(a, settingnr=None, lam=None):
    return a.get_ShapleyValues(settingnr, lam)


# Export functions and classes:
__all__ = [
    # from Analysis.py:
    "get_variances",
    "get_GSI",
    "get_AttributeRanking",
    "get_ActiveSet",
    "get_ShapleyValues",
    # from approx.py:
    "approx",
    "approximate",
    "autoapproximate",
    "evaluate",
    "evaluateANOVAterms",
    # from Errors.py:
    "get_l2_error",
    "get_mse",
    "get_mad",
    "get_L2_error",
    "get_acc",
    "get_auc",
    # from trafo.py:
    "transform_cube",
    "transform_R",
    # from fista.py:
    # "bisection",
    # "newton",
    # "λ2ξ",
    # "fista",
    # from bandwidth.py:
    "compute_bandwidth",
    "estimate_rates",
]
