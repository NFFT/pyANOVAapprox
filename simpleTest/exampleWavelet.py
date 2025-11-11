# pip install pyANOVAapprox

# Example for approximating an non periodic function

import math

import matplotlib.pyplot as plt
import numpy as np

import pyANOVAapprox as ANOVAapprox

basis = "chui3"
# for 'cos' the samples have to be in [0,1]^d
# for a periodic function use 'per' or wavelets  'chui2', 'chui3', 'chui4' (samples have to be in [-0.5,0.5]^d here, 'chuim' are the Chui-Wang wavelets of order m)

def TestFunction(x):  # this function is of the form f_0 + f_1 + f_2 + f_3 + f_4 + f_2,3
    return (
        2 * abs(x[0])
        + abs(math.sin(math.pi * x[1] * x[2]))
        + np.cos(3 + x[3])
    )


rng = np.random.default_rng(1234)

##################################
## Definition of the parameters ##
##################################

d = 8  # dimension
q = 2      # superposition dimension
M = 10000  # number of used evaluation points to train the model
M_test = 10000  # number of used evaluation points to test the accuracity the model
N = [5,2]  # number of parameters, should be vector of length q: 
# for wavelets the total number of parameters scales exponentially, i.e.:
# for q = 1 and N = [N1] the total number of parameters scales like ~O(d*2^N1)
# for q = 2 and N = [N1 , N2] the total number of parameters scales like ~O(d*2^N1) + O(d^2 * N2*2^N2)

lambdas = np.array([0.0])  # used regularisation parameters λ

############################
## Generation of the data ##
############################

if basis =="chui2" or basis =="chui3" or basis =="chui4" or basis =="per":
    X = rng.random((M, d)) -0.5        # for perioidic approximation samples have to be in [-0.5,0.5]^d
elif basis == "cos":
    X = rng.random((M, d))
y = np.array(
    [TestFunction(X[i, :].T) for i in range(M)]
)  # evaluate the function at these points

if basis == "chui1" or basis =="chui2" or basis =="chui3" or basis =="chui4" or basis =="per":
    X_test = rng.random((M_test, d)) - 0.5      # for perioidic approximation samples have to be in [-0.5,0.5]^d
elif basis == "cos":
    X_test = rng.random((M_test, d))     
y_test = np.array(
    [TestFunction(X_test[i, :].T) for i in range(M_test)]
)  # the same for the test points

##########################
## Do the approximation ##
##########################

####  construct model for ANOVAapprox ####
anova_model = ANOVAapprox.approx(X, y, ds=q, basis=basis, N=N)

####  Do approximation by least-squares ###
anova_model.approximate(lam=lambdas, solver="lsqr")
print("Total number of used parameters = " + str(len(anova_model.fc[lambdas[0]].vec())))

#######################
## Analyze the model ##
#######################

### Do sensitivity analysis ####
gsis = ANOVAapprox.get_GSI(anova_model,0.0)                     #calculates indices for importance of terms (gsis is vector, with indices belonging to terms in anova_model.U)
#gsis_as_dict = ANOVAapprox.get_GSI(anova_model,0.0,dict=true) 

y_min_calc = 10 ** (np.min(np.log10(gsis)) - 0.5)
label = list(anova_model.U[1:])
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

################################
## get approximation accuracy ##
################################

### error analysis ###
mse_train = ANOVAapprox.get_mse(anova_model,lam=0.0)
mse_test = ANOVAapprox.get_mse(anova_model,X_test,y_test, lam=0.0)

print("MSE on test points: " + str(mse_test)) 

################################################
## Approximation with better suited index set ##
################################################

U = ANOVAapprox.get_ActiveSet(anova_model, [0.01, 0.01], lam=0.0)
print("Found index-set U: " + str(U) )
anova_model = ANOVAapprox.approx(X, y, U=U, N=[i+2 for i in N] , basis=basis)               # increase number of paramers in N for the important terms
anova_model.approximate(lam=lambdas)
print("Total number of used parameters = " + str(len(anova_model.fc[lambdas[0]].vec())))
mse_train = ANOVAapprox.get_mse(anova_model,lam=0.0)
mse_test = ANOVAapprox.get_mse(anova_model,X_test,y_test, lam=0.0)
print("MSE on test points after ANOVA truncation: " + str(mse_test)) 


