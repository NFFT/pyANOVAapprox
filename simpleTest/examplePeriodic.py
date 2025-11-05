# pip install pyANOVAapprox

# Example for approximating an periodic function

import math

import matplotlib.pyplot as plt
import numpy as np

import pyANOVAapprox as ANOVAapprox


def TestFunction(x):
    return (
        np.sin(4 * np.pi * x[0] * x[4])
        + 2
        - np.exp(np.sin(2 * np.pi * x[3]))
        + np.sin(4 * np.pi * x[5] * x[1]) ** 2
    )


rng = np.random.default_rng(1234)

##################################
## Definition of the parameters ##
##################################

d = 6  # dimension

M = 10000  # number of used evaluation points to train the model
M_test = 100000  # number of used evaluation points to test the accuracity the model

max_iter = 50  # maximum number of iterations

# there are 3 possibilities with varying degree of freedom to define the number of used frequencies
########### Variant 1:
ds = 2  # superposition dimension
num = np.sum([math.comb(6, k) for k in np.arange(1, 2 + 1)])  # number of used subsets
b = M / (
    math.log10(M) * num
)  # number for the number of frequencies if we use logarithmic oversampling and distribute it evenly to all subsets
bw = np.array(
    [math.floor(b / 2) * 2, math.floor(math.sqrt(b) / 2) * 2]
)  # bandwidths (use even numbers)
# Use all subsets up to ds and use bw[1] many frequences in the the subsets with one element, b[2]^2 many for subsets with two elements and so on
#
########### Variant 2:
# used subsets:
# U = [Int[], [1], [2], [3], [4], [5], [6],
#     [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 3], [2, 4], [2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5], [4, 6], [5, 6]]
# Bandwidths for these subsets:
# N = [[0]  , 100, 100, 100, 100, 100, 100,
#         10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10,     10]
# Use the subsets U with the bandwiths N. The bandwith N[i] corresponds to the subset U[i]. For subsets with more then one direction is the same bandwidth in all directions used
#
########### Variant 3:
# used subsets:
# U = [Int[],   [1],   [2],   [3],   [4],   [5],   [6],
#      [ 1, 2],[ 1, 3],[ 1, 4],[ 1, 5],[ 1, 6],[ 2, 3],[ 2, 4],[ 2, 5],[ 2, 6],[ 3, 4],[ 3, 5],[ 3, 6],[ 4, 5],[ 4, 6],[ 5, 6]]
# Bandwidths for these subsets:
# N = [Int[], [100], [100], [100], [100], [100], [100],
#      [10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10],[10,10]]
# Use the subsets U with the bandwiths N. The bandwith N[i] corresponds to the subset U[i]. The bandwidth N[i][j] corresponds to the direction U[i][j]

lambdas = np.array([0.0, 1.0])  # used regularisation parameters λ

############################
## Generation of the data ##
############################
X = rng.random((M, d)) - 0.5  # construct the evaluation points for training
y = np.array(
    [TestFunction(X[i, :].T) for i in range(M)], dtype=complex
)  # evaluate the function at these points
X_test = rng.random((M_test, d)) - 0.5  #
y_test = np.array(
    [TestFunction(X_test[i, :].T) for i in range(M_test)], dtype=complex
)  # the same for the test points

##########################
## Do the approximation ##
##########################

ads = ANOVAapprox.approx(X, y, ds=ds, basis="per", N=bw)
ads.approximate(lam=lambdas, solver="lsqr")

################################
## get approximation accuracy ##
################################

# mse = ANOVAapprox.get_mse(ads) # get mse error at the given training points
mse = ANOVAapprox.get_mse(ads, X_test, y_test)  # get mse error at the test points
λ_min = min(
    mse, key=mse.get
)  # get the regularisation parameter which leads to the minimal error
mse_min = mse[λ_min]

print("mse = " + str(mse_min))

###############################################
## Analyze the model to improve the accuracy ##
###############################################


ar = ANOVAapprox.get_AttributeRanking(ads, λ_min)  # get the attrbute ranking

plt.figure()
(markers, stemlines, baseline) = plt.stem(
    np.arange(1, d + 1),  # x-Werte (1:d)
    ar,  # y-Werte (ar)
    linefmt="C0-",  # Stil der Stiele
    markerfmt="C0^",  # Stil der Markierung (C0 = erste Farbe, ^ = up-triangle)
    basefmt="k:",  # Stil der Basislinie (optional)
)
plt.setp(markers, markersize=8)
plt.yscale("log")
y_min_calc = 10 ** (np.min(np.log10(ar)) - 0.5)
plt.ylim(y_min_calc, 1)
plt.title("Attribute Ranking")
plt.xlabel("Attribut-Index")
plt.xlim(0.5, d + 0.5)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()  # plot the arrtibute ranking in an logplot
print("active dimensions: " + str(ar[ar > 1e-2]))

gsis = ANOVAapprox.get_GSI(ads, λ_min)
label = list(ads.U[1:])
l = len(label)
plt.figure()
x_values = np.arange(1, l + 1)
(markers, stemlines, baseline) = plt.stem(
    x_values,  # X-Werte: 1 bis l
    gsis,  # Y-Werte: gsis
    linefmt="C0-",  # Stil der Stiele
    markerfmt="C0^",  # Stil der Markierung (^ = up-triangle)
    basefmt="k:",  # Stil der Basislinie
)
plt.setp(markers, markersize=8)
plt.xticks(x_values, label, rotation=45, ha="right")
plt.xlabel("Input Dimension")
plt.yscale("log")
plt.ylim(y_min_calc, 1)
plt.title("Global sensitivity indices")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()  # Stellt sicher, dass die Labels sichtbar sind
plt.show()
print(
    "important dimensional interactions: : "
    + str([label[i] for i in np.arange(0, l)[gsis > 1e-2]])
)

################################################
## Approximation with better suited index set ##
################################################

Umask = np.append(np.array([True]), gsis > 1e-2)
U = [ads.U[i] for i in np.arange(0, len(Umask))[Umask]]  # get important subsets
bws = M / (math.log10(M) * (len(U) - 1))  # calculate frequencies per subset
N = np.array(
    [math.floor(bws ** (1 / max(1, len(u))) / 2) * 2 for u in U]
)  # distribute the frequencies evenly and make them even
N[0] = 0

a = ANOVAapprox.approx(
    X, y, U, N, "per"
)  # generate the data structure for the approximation
a.approximate(
    lam=lambdas, solver="lsqr"
)  # do the approximation for all specified regularisation parameters

mse = ANOVAapprox.get_mse(a, X_test, y_test)  # get mse error at the test points
λ_min = min(
    mse, key=mse.get
)  # get the regularisation parameter which leads to the minimal error
mse_min = mse[λ_min]
print("mse = " + str(mse_min))

########################
## Evaluate the model ##
########################

# y_approx = a.evaluate(lam=λ_min) # evaluate the approximation at the training points for the regularisation λ_min
# y_approx = a.evaluate(X=X_test, lam=λ_min) # evaluate the approximation at the points X_test for the regularisation λ_min

# In the following we plot the real and the approximated anova term for the subset u=[3]

y_eval_anova = a.evaluateANOVAterms(X=X_test, lam=λ_min) # evaluate all of the ANOVA terms 
pos = a.U.index((3,)) # find the index for the subset u=[3]
y_eval_anova_3 = y_eval_anova.T[pos]

perm = np.argsort(X_test.T[3])
X_plot = X_test.T[3][perm]
y_eval_anova_3_plot = np.real(y_eval_anova_3[perm])
y_anova_3_plot = -np.exp(np.sin(2*np.pi*X_plot))+1.26607

plt.figure()
plt.plot(
    X_plot,
    y_eval_anova_3_plot,
    label="approximation"
)
plt.plot(
    X_plot,
    y_anova_3_plot,
    label="ANOVA term" # ... "ANOVA term"]
)
plt.title("Approximation of the ANOVA term 4") # title = "..."
plt.legend() # Zeigt die Labels/Legende an
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
