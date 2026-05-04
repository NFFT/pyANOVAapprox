# pip install pyANOVAapprox

# Example for approximating an periodic function

import math

import matplotlib.pyplot as plt
import numpy as np
from TestFunctionPeriodic import *

import pyANOVAapprox as ANOVAapprox


def TestFunction(x):
    return b_spline_2(x[0]) * b_spline_4(x[1]) * b_spline_6(x[2])


rng = np.random.default_rng(1234)

##################################
## Definition of the parameters ##
##################################

d = 3  # dimension

M = 100000  # number of used evaluation points to train the model
M_test = 100000  # number of used evaluation points to test the accuracity the model

U = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

lambdas = np.array([0.0])  # used regularisation parameters λ

############################
## Generation of the data ##
############################

X = rng.random((M, d))  # construct the evaluation points for training
y = np.array(
    [TestFunction(X[i, :].T) for i in range(M)], dtype=complex
)  # evaluate the function at these points
X = X - 0.5
X_test = rng.random((M_test, d))
y_test = np.array(
    [TestFunction(X_test[i, :].T) for i in range(M_test)], dtype=complex
)  # the same for the test points
X_test = X_test - 0.5

##########################
## Do the approximation ##
##########################

ads = ANOVAapprox.approx(X, y, U=U, basis="per")
ads.autoapproximate()

################################
## get approximation accuracy ##
################################

# mse = ads.get_mse() # get mse error at the given training points
mse = ads.get_mse(X=X_test, y=y_test)  # get mse error at the test points
λ_min = min(
    mse, key=mse.get
)  # get the regularisation parameter which leads to the minimal error
mse_min = mse[λ_min]

print("mse = " + str(mse_min))
