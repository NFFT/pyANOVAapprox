#### PERIODIC TEST SOLVER LSQR ####

import os
import sys

src_aa = os.path.abspath(os.path.join(os.getcwd(), "src"))
sys.path.insert(0, src_aa)

import numpy as np
from pyGroupedTransforms import *
from TestFunctionPeriodic import *

import pyANOVAapprox as ANOVAapprox

d = 6
ds = 2
M = 10000  # should be 10000
max_iter = 50
bw = [100, 10]
lambdas = np.array([0.0, 1.0])

rng = np.random.default_rng()
X = rng.random((M, d)) - 0.5
y = np.array([f(X[i, :].T) for i in range(M)], dtype=complex)
X_test = rng.random((M, d)) - 0.5
y_test = np.array([f(X_test[i, :].T) for i in range(M)], dtype=complex)


ads = ANOVAapprox.approx(X, y, ds=ds, basis="per", N=bw)
ads.approximate(lam=lambdas, solver="lsqr")


attr_ranking = ANOVAapprox.get_AttributeRanking(ads, lam=0.0)
print("AR: ", np.sum(attr_ranking))
assert abs(np.sum(attr_ranking) - 1) < 1e-4


bw = ANOVAapprox.get_orderDependentBW(AS, [128, 32])
aU = ANOVAapprox.approx(X, y, U=AS, N=bw, basis="per")
aU.approximate(lam=lambdas, solver="lsqr")


err_L2_ds = ANOVAapprox.get_L2_error(ads, norm(), fc)[0.0]
err_L2_U = ANOVAapprox.get_L2_error(aU, norm(), fc)[0.0]
err_l2_ds = ANOVAapprox.get_l2_error(ads)[0.0]
err_l2_U = ANOVAapprox.get_l2_error(aU)[0.0]
err_l2_rand_ds = ANOVAapprox.get_l2_error(ads, X_test, y_test)[0.0]
err_l2_rand_U = ANOVAapprox.get_l2_error(aU, X_test, y_test)[0.0]


print("== PERIODIC LSQR ==")
print("L2 ds: ", err_L2_ds)
print("L2 U: ", err_L2_U)
print("l2 ds: ", err_l2_ds)
print("l2 U: ", err_l2_U)
print("l2 rand ds: ", err_l2_rand_ds)
print("l2 rand U: ", err_l2_rand_U)

assert err_L2_ds < 0.01
assert err_L2_U < 0.01  # maybe restrict to 0.005
assert err_l2_ds < 0.01
assert err_l2_U < 0.01
assert err_l2_rand_ds < 0.01
assert err_l2_rand_U < 0.01
