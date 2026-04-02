import os
import sys

repo_root = os.getcwd()
src_aa = os.path.abspath(os.path.join(repo_root, "src"))
src_gt = os.path.abspath(os.path.join(repo_root, "..", "pyGroupedTransforms", "src"))
src_nfft3 = os.path.abspath(os.path.join(repo_root, "..", "pyNFFT3", "src"))

sys.path.insert(0, src_aa)
sys.path.insert(0, src_gt)
sys.path.insert(0, src_nfft3)

import numpy as np
from TestFunctionPeriodic import AS, f, fc, norm

import pyANOVAapprox as ANOVAapprox

d = 6
ds = 2
M = 10000
max_iter = 50
bw = [100, 10]
lambdas = np.array([0.0, 1.0])

basis_vect = ["exp", "exp", "exp", "exp", "exp", "exp"]

rng = np.random.default_rng(1234)
X = rng.random((M, d)) / 2.0
y = np.array([f(X[i, :]) for i in range(M)], dtype=complex)
X_test = rng.random((M, d)) / 2.0
y_test = np.array([f(X_test[i, :]) for i in range(M)], dtype=complex)

ads = ANOVAapprox.approx(X, y, ds=ds, N=bw, basis="mixed", basis_vect=basis_vect)
ads.approximate(lam=lambdas, solver="lsqr", max_iter=max_iter)

attr_ranking = ads.get_AttributeRanking(lam=0.0)
print("AR: ", np.sum(attr_ranking))
assert abs(np.sum(attr_ranking) - 1.0) < 1e-4

bw = ANOVAapprox.get_orderDependentBW(AS, [128, 32])

aU = ANOVAapprox.approx(X, y, U=AS, N=bw, basis="mixed", basis_vect=basis_vect)
aU.approximate(lam=lambdas, solver="lsqr", max_iter=max_iter)

err_L2_ds = ads.get_L2_error(norm(), fc)[0.0]
err_L2_U = aU.get_L2_error(norm(), fc)[0.0]
err_l2_ds = ads.get_l2_error()[0.0]
err_l2_U = aU.get_l2_error()[0.0]
err_l2_rand_ds = ads.get_l2_error(X=X_test, y=y_test)[0.0]
err_l2_rand_U = aU.get_l2_error(X=X_test, y=y_test)[0.0]

print("== MIXED LSQR ==")
print("L2 ds: ", err_L2_ds)
print("L2 U: ", err_L2_U)
print("l2 ds: ", err_l2_ds)
print("l2 U: ", err_l2_U)
print("l2 rand ds: ", err_l2_rand_ds)
print("l2 rand U: ", err_l2_rand_U)

assert err_l2_ds < 0.01
assert err_l2_U < 0.005
assert err_l2_rand_ds < 0.01
assert err_l2_rand_U < 0.005
