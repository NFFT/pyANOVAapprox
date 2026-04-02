# pip install pyANOVAapprox

# Example for approximating a function with a mixed basis (exp, alg, cos, alg)
#
# Test function (4-dimensional):
#
#   f(x) = exp(sin(2π x₁) · x₂)  +  cos(π x₃) · x₄²
#          + 1/10 · sin²(2π x₁)  +  5 · √(x₂ x₄ + 1)
#
# Basis assignment per dimension:
#   x₁  →  "exp"  (exponential),  x₁ ∈ [-0.5, 0.5)
#   x₂  →  "alg"  (Chebyshev / algebraic),  x₂ ∈ [0, 1]
#   x₃  →  "cos"  (cosine),  x₃ ∈ [0, 1]
#   x₄  →  "alg"  (Chebyshev / algebraic),  x₄ ∈ [0, 1]

import math

import numpy as np

import pyANOVAapprox as ANOVAapprox


def TestFunction(x):
    return (
        np.exp(np.sin(2 * np.pi * x[0]) * x[1])
        + np.cos(np.pi * x[2]) * x[3] ** 2
        + 0.1 * np.sin(2 * np.pi * x[0]) ** 2
        + 5 * np.sqrt(x[1] * x[3] + 1)
    )


rng = np.random.default_rng(1234)

##################################
## Definition of the parameters ##
##################################

d = 4  # dimension
basis_vect = ["exp", "alg", "cos", "alg"]  # basis per dimension

M = 10000       # number of training points
M_test = 100000  # number of test points
max_iter = 50   # maximum number of LSQR iterations

# there are 3 possibilities with varying degree of freedom to define the number of used frequencies
########### Variant 1:
ds = 2  # superposition dimension
num = np.sum([math.comb(d, k) for k in np.arange(1, ds + 1)])  # number of used subsets
b = M / (
    math.log10(M) * num
)  # number for the number of frequencies if we use logarithmic oversampling and distribute it evenly to all subsets
bw = [
    math.floor(b / 2) * 2,
    math.floor(math.sqrt(b) / 2) * 2,
]  # bandwidths (use even numbers)
# Use all subsets up to ds and use bw[0] many frequencies in subsets with one element,
# bw[1]^2 many for subsets with two elements and so on.
#
########### Variant 2:
# used subsets:
# U = [(), (0,), (1,), (2,), (3,),
#      (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
# Bandwidths for these subsets:
# N = [0, 100, 100, 100, 100,
#      10, 10, 10, 10, 10, 10]
# Use the subsets U with the bandwidths N. The bandwidth N[i] corresponds to the subset U[i].
# For subsets with more than one direction the same bandwidth is used in all directions.
#
########### Variant 3:
# used subsets:
# U = [(), (0,), (1,), (2,), (3,),
#      (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
# Bandwidths for these subsets:
# N = [(), (100,), (100,), (100,), (100,),
#      (10,10), (10,10), (10,10), (10,10), (10,10), (10,10)]
# Use the subsets U with the bandwidths N. The bandwidth N[i][j] corresponds to direction U[i][j].

lambdas = np.array([0.0, 1.0])  # used regularisation parameters λ

############################
## Generation of the data ##
############################

# Nodes are drawn from the correct domain for each basis,
# with Chebyshev-weight sampling for the algebraic (alg) dimensions:
#
#   x₁ (exp):  uniform on [-0.5, 0.5)  →  u ∈ [0,1], shift by -0.5
#   x₂ (alg):  arcsine distribution on [0, 1]  →  x = (1 − cos(π u)) / 2
#   x₃ (cos):  uniform on [0, 1]  (cosine series, no special weight needed)
#   x₄ (alg):  arcsine distribution on [0, 1]  →  x = (1 − cos(π u)) / 2
#
# The transformation (1 − cos(πu))/2 is the analogue of sin(π(u−0.5))
# used in exampleCheb.py, adapted to the domain [0, 1].


def cheb_nodes(rng, size):
    """Sample from the arcsine (Chebyshev-weight) distribution on [0, 1]."""
    return (1.0 - np.cos(np.pi * rng.random(size))) / 2.0


X = rng.random((M, d))          # initially all columns uniform in [0, 1]
X[:, 0] -= 0.5                  # x₁ (exp):  shift to [-0.5, 0.5)
X[:, 1] = cheb_nodes(rng, M)    # x₂ (alg):  Chebyshev-weight on [0, 1]
X[:, 3] = cheb_nodes(rng, M)    # x₄ (alg):  Chebyshev-weight on [0, 1]
y = np.array(
    [TestFunction(X[i, :]) for i in range(M)], dtype=complex
)  # function values (complex because the mixed basis with exp uses complex coefficients)

X_test = rng.random((M_test, d))
X_test[:, 0] -= 0.5
X_test[:, 1] = cheb_nodes(rng, M_test)
X_test[:, 3] = cheb_nodes(rng, M_test)
y_test = np.array(
    [TestFunction(X_test[i, :]) for i in range(M_test)], dtype=complex
)

##########################
## Do the approximation ##
##########################

ads = ANOVAapprox.approx(X, y, ds=ds, basis="mixed", N=bw, basis_vect=basis_vect)
ads.approximate(lam=lambdas, max_iter=max_iter, solver="lsqr")

################################
## get approximation accuracy ##
################################

mse = ads.get_mse(X=X_test, y=y_test)  # get mse error at the test points
λ_min = min(
    mse, key=mse.get
)  # get the regularisation parameter which leads to the minimal error
mse_min = mse[λ_min]

print("mse = " + str(mse_min))

###############################################
## Analyze the model to improve the accuracy ##
###############################################

# Attribute ranking: how much does each dimension contribute?
attr_ranking = ads.get_AttributeRanking(lam=λ_min)
print("Attribute ranking:")
for i, r in enumerate(attr_ranking):
    print(f"  x{i+1} ({basis_vect[i]}): {r:.4f}")

# Global sensitivity indices grouped by ANOVA term
gsi = ads.get_GSI(lam=λ_min, Dict=True)
print("\nGlobal sensitivity indices (top terms):")
for u, v in sorted(gsi.items(), key=lambda item: item[1], reverse=True)[:8]:
    if v > 1e-4:
        print(f"  u = {u}: {v:.4f}")
