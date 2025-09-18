import numpy as np
from scipy.stats import iqr, norm

##### fuctions that trasforms data X dimension-wise:

#### transform data to [-1/2,1/2]^d ####
### input:
# X     .... Matrix M x d     (X is transposed with respect to the "X" from julia)
# sigma .... if sigma should be output
# X_test .... if test data should be transformed: Insert X_test in format M_test x d


def transform_cube(X, sigma=False, X_test=[]):
    M, d = X.shape
    X_trafo = np.zeros_like(X_test if isinstance(X_test, np.ndarray) else X)
    mm = 3
    sigmas = {}

    for i in range(d):
        s = 1.06 * min(np.std(X[:, i]), iqr(X[:, i]) / 1.34) * M ** (-1 / 5)
        sigmas[i] = s

        if len(X_test) != 0:
            X_trafo[:, i] = Rho_circ(X_test[:, i], X[:, i].reshape(1, -1), mm, s) - 0.5
        else:
            X_trafo[:, i] = Rho_circ(X[:, i], X[:, i].reshape(1, -1), mm, s) - 0.5

    return (X_trafo, sigmas) if sigma else X_trafo


#### transform data to [-1/2,1/2]^d ####
# input: ####
# X         .... Matrix M x d      (X is transposed with respect to the "X" from julia)
# optional: ####
# sigma     .... if sigma should be output
# X_test    .... if test data should be transformed: Insert X_test in format M_test x d
# KDE       .... choice of parameter selction: ROT or DPI


def transform_R(X, sigma=False, X_test=False, KDE="ROT"):

    M, d = X.shape
    use_test = isinstance(X_test, np.ndarray)
    X_trafo = np.zeros_like(X_test if use_test else X)
    sigmas = {}

    for i in range(d):
        Xi = X[:, i]

        if KDE == "ROT":
            std_i = np.std(Xi)
            iqr_i = iqr(Xi)
            s = 1.06 * min(std_i, iqr_i / 1.34) * M ** (-1 / 5)

        elif KDE == "DPI":
            std_i = np.std(Xi)
            Ψ_8 = 105 / (32 * np.sqrt(np.pi) * std_i**9)
            g1 = (-(-30 / np.sqrt(2 * np.pi)) / (Ψ_8 * M)) ** (1 / 9)

            X_diff = (Xi[:, None] - Xi[None, :]) / g1
            Ψ_6 = (1 / (M**2 * g1**7)) * np.sum(rho_diff6(X_diff))

            g2 = ((-6 / np.sqrt(2 * np.pi)) / (Ψ_6 * M)) ** (1 / 7)
            X_diff = (Xi[:, None] - Xi[None, :]) / g2
            Ψ_4 = (1 / (M**2 * g2**5)) * np.sum(rho_diff4(X_diff))

            s = (1 / (2 * np.sqrt(np.pi) * Ψ_4 * M)) ** (1 / 5)

        else:
            raise ValueError("KDE not defined. Choose from ROT or DPI")

        sigmas[i] = s
        components = [norm(loc=mu, scale=s) for mu in Xi]

        def mixture_cdf(val):
            return np.mean([c.cdf(val) for c in components])

        if use_test:
            Xi_test = X_test[:, i]
            X_trafo[:, i] = np.array([mixture_cdf(xi) for xi in Xi_test]) - 0.5
        else:
            X_trafo[:, i] = np.array([mixture_cdf(xi) for xi in Xi]) - 0.5

    return (X_trafo, sigmas) if sigma else X_trafo


# helpfunctions:
def Rho1(x, m):
    if m != 3:
        raise NotImplementedError("Only m=3 is implemented")

    result = np.zeros_like(x)
    result[x >= 1.5] = 1
    result[(x >= 0.5) & (x < 1.5)] = (
        5 / 6
        + 9 / 8 * x[(x >= 0.5) & (x < 1.5)]
        - 3 / 4 * x[(x >= 0.5) & (x < 1.5)] ** 2
        + 1 / 6 * x[(x >= 0.5) & (x < 1.5)] ** 3
        - 19 / 48
    )
    result[(x >= -0.5) & (x < 0.5)] = (
        1 / 6
        + 3 / 4 * x[(x >= -0.5) & (x < 0.5)]
        - 1 / 3 * x[(x >= -0.5) & (x < 0.5)] ** 3
        + 1 / 3
    )
    result[(x >= -1.5) & (x < -0.5)] = (
        9 / 8 * x[(x >= -1.5) & (x < -0.5)]
        + 3 / 4 * x[(x >= -1.5) & (x < -0.5)] ** 2
        + 1 / 6 * x[(x >= -1.5) & (x < -0.5)] ** 3
        + 9 / 16
    )
    result[x < -1.5] = 0
    return result


def Rho_circ(x: np.ndarray, X: np.ndarray, m: int, sigma: float) -> np.ndarray:
    M = X.shape[1]
    return (1 / M) * np.sum(Rho1((x[:, None] - X) / sigma, m), axis=1)


def rho_diff6(X: np.ndarray) -> np.ndarray:
    return norm.pdf(X) * (X**6 - 15 * X**4 + 45 * X**2 - 15)


def rho_diff4(X: np.ndarray) -> np.ndarray:
    return norm.pdf(X) * (X**4 - 6 * X**2 + 3)
