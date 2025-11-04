from pyANOVAapprox import *


def _l2error(a, lam, X, y):  # helpfunction for get_l2error
    if y is None:
        y = a.y

    if X is not None:
        y_eval = a.evaluate(lam, X)
    else:
        y_eval = a.evaluate(lam)

    return np.linalg.norm(y_eval - y) / np.linalg.norm(y)


def get_l2_error(a, X=None, y=None, lam=None):
    """
    Computes the relative ``\ell_2`` error for an approx object a.

    - If only `a` and `lam` are provided,
      this function computes the relative ``\ell_2`` error on the training nodes for the specific regularization parameter `lam`.

    - If `a`, `X`, `y`, and `lam` are provided,
      this function computes the relative ``\ell_2`` error on the given data `X` and `y` for the specified regularization parameter `lam`.

    - If only `a` is provided,
      this function computes the relative ``\ell_2`` error on the training nodes for all available regularization parameters.

    - If `a`, `X`, and `y` are provided,
      this function computes the relative ``\ell_2`` error on the given data `X` and `y` for all available regularization parameters.

    Returns either a float (for a single lam) or a dictionary mapping lam values to errors.
    """

    if lam is not None:
        return _l2error(a, lam, X, y)
    else:
        return {l: _l2error(a, l, X, y) for l in list(a.fc)}


def _mse(a, lam, X, y):  # helpfunction for get_mse
    if y is None:
        y = a.y

    if X is not None:
        y_eval = a.evaluate(lam, X)
    else:
        y_eval = a.evaluate(lam)

    return 1 / len(y) * (np.linalg.norm(y_eval - y) ** 2)


def get_mse(a, X=None, y=None, lam=None):
    """
    Computes the mean square error (mse) for an approx object a.

    - If only `a` and `lam` are provided,
      this function computes the mean square error on the training nodes for a specific regularization parameter `lam`.

    - If `a`, `X`, `y`, and `lam` are provided,
      this function computes the mean square error on the given data `X` and `y` for the specified regularization parameter `lam`.

    - If only `a` is provided,
      this function computes the mean square error on the training nodes for all available regularization parameters.

    - If `a`, `X`, and `y` are provided,
      this function computes the mean square error on the given data `X` and `y` for all available regularization parameters.

    Returns either a float (for a single `lam`) or a dictionary mapping each `lam` to its corresponding MSE value.
    """

    if lam is not None:
        return _mse(a, lam, X, y)
    else:
        return {l: _mse(a, l, X, y) for l in list(a.fc)}


def _mad(a, lam, X, y):  # helpfunction for get_mad
    if y is None:
        y = a.y

    if X is not None:
        y_eval = a.evaluate(lam, X)
    else:
        y_eval = a.evaluate(lam)

    return 1 / len(y) * np.linalg.norm(y_eval - y, ord=1)


def get_mad(a, X=None, y=None, lam=None):
    """
    Computes the mean absolute deviation (mad) for an approx object a.

    - If only `a` and `lam` are provided,
      this function computes the mean absolute deviation on the training nodes for a specific regularization parameter `lam`.

    - If `a`, `X`, `y`, and `lam` are provided,
      this function computes the mean absolute deviation on the given data `X` and `y` for the specified regularization parameter `lam`.

    - If only `a` is provided,
      this function computes the mean absolute deviation on the training nodes for all available regularization parameters.

    - If `a`, `X`, and `y` are provided,
      this function computes the mean absolute deviation on the given data `X` and `y` for all available regularization parameters.

    Returns a single MAD value (float), if `lam` is provided, or a dictionary mapping each `lam` to its corresponding MAD.
    """

    if lam is not None:
        return _mad(a, lam, X, y)
    else:
        return {l: _mad(a, l, X, y) for l in list(a.fc)}


def _L2error(a, norm, bc_fun, lam):

    if a.basis in {"per", "cos", "cheb", "std", "mixed"}:
        error = norm**2
        index_set = get_IndexSet(a.trafo.settings, a.X.shape[1])

        for i in range(index_set.shape[1]):
            k = index_set[:, i]
            error += abs(bc_fun(k) - a.fc[lam][i]) ** 2 - abs(bc_fun(k)) ** 2

        return np.sqrt(error) / norm
    else:
        raise NotImplementedError("L2 error is not implemented for this basis.")


def get_L2_error(a, norm, bc_fun, lam=None):
    """
    Computes the relative L2 error of a function approximation for an `approx` object `a`.

    - If `a`, `norm`, `bc_fun`, and `lam` are provided,
      this function computes the relative L2 error for a specific regularization parameter `lam`.

    - If only `a`, `norm`, and `bc_fun` are provided,
    this function computes the relative L2 error for all available regularization parameters.
    """

    if lam is not None:
        return _L2error(a, norm, bc_fun, lam)
    else:
        return {l: _L2error(a, norm, bc_fun, l) for l in list(a.fc)}


def _acc(a, lam, X, y):  # helpfunction for get_acc
    if y is None:
        y = a.y

    if X is not None:
        y_eval = a.evaluate(lam, X)
    else:
        y_eval = a.evaluate(lam)

    return np.sum(np.sign(y_eval) == y) / len(y) * 100.0


def get_acc(a, X=None, y=None, lam=None):

    if lam is not None:
        return _acc(a, lam, X, y)
    else:
        return {l: _acc(a, l, X, y) for l in list(a.fc)}


# def _auc(a, lam, X, y):  # helpfunction for get_auc    #TODO: implement ROC AUC Score
#    if y is None:
#        y = a.y
#
#    if X is not None:
#        y_eval = a.evaluate(lam, X)
#    else:
#        y_eval = a.evaluate(lam)
#
#    y_sc = (y_eval - np.min(y_eval)) / (np.max(y_eval) - np.min(y_eval))
#    y = np.where(y == -1.0, 0, y)
#    y = np.where(y == 1.0, 1, y)
#    y_int = y.astype(np.int64)
#
#    return roc_auc_score(y_int, y_sc)


def get_auc(
    a, X=None, y=None, lam=None
):  # gab einen Error in julia und in Python, als ich es mit komplexen werten getestet habe...

    if lam is not None:
        return _auc(a, lam, X, y)
    else:
        return {l: _auc(a, l, X, y) for l in list(a.fc)}
