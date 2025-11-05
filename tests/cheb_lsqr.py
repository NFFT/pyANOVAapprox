import os
import sys

src_aa = os.path.abspath(os.path.join(os.getcwd(), "src"))
sys.path.insert(0, src_aa)

import numpy as np
import TestFunctionCheb
from pyGroupedTransforms import *

import pyANOVAapprox as ANOVAapprox

d = 8
ds = 2
M = 1000  # eigentlich 10000
max_iter = 50
bw = [20, 4]
lambdas = np.array([0.0, 1.0])

(X, y) = TestFunctionCheb.generateData(M)
(X_test, y_test) = TestFunctionCheb.generateData(M)


#### ####


ads = ANOVAapprox.approx(X.T, y, ds=ds, basis="cheb", N=bw)
ads.approximate(lam=lambdas, solver="lsqr")

# bw = ANOVAapprox.get_orderDependentBW(TestFunctionCheb.AS, bw)

aU = ANOVAapprox.approx(X.T, y, U=TestFunctionCheb.AS, N=bw, basis="cheb")
aU.approximate(lam=lambdas, solver="lsqr")


err_L2_ds = ANOVAapprox.get_L2_error(ads, TestFunctionCheb.norm(), TestFunctionCheb.fc)[
    0.0
]
err_L2_U = ANOVAapprox.get_L2_error(ads, TestFunctionCheb.norm(), TestFunctionCheb.fc)[
    0.0
]
err_l2_ds = ANOVAapprox.get_l2_error(ads)[0.0]
err_l2_U = ANOVAapprox.get_l2_error(aU)[0.0]
err_l2_rand_ds = ANOVAapprox.get_l2_error(ads, X_test.T, y_test)[0.0]
err_l2_rand_U = ANOVAapprox.get_l2_error(aU, X_test.T, y_test)[0.0]

ANOVAapprox.get_mse(ads)
ANOVAapprox.get_mse(ads, X_test.T, y_test)
ANOVAapprox.get_mad(ads)
ANOVAapprox.get_mad(ads, X_test.T, y_test)
ANOVAapprox.get_GSI(ads)
ANOVAapprox.get_GSI(ads, Dict=True)
ads.evaluate()
ads.evaluate(X=X_test.T)

print("== PERIODIC FISTA ==")
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
# assert err_l2_rand_ds < 0.01  stehen nicht in der analogen julia datei
# assert err_l2_rand_U < 0.01
