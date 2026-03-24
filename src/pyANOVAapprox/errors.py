# from pyANOVAapprox import *
import numpy as np
import pyGroupedTransforms as gt


def _l2_error(self, settingnr, lam, X, y):  # helpfunction for get_l2error
    if y is None:
        y = self.y

    if X is not None:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam, X=X)
    else:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam)

    return np.linalg.norm(y_eval - y) / np.linalg.norm(y)


def get_l2_error(self, settingnr=None, X=None, y=None, lam=None):
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
        return self._l2_error(settingnr, lam, X, y)
    else:
        return {
            l: self._l2_error(settingnr, l, X, y)
            for l in self.lam.keys()
        }


def _mse(self, settingnr, lam, X, y):  # helpfunction for get_mse
    if y is None:
        y = self.y

    if X is not None:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam, X=X)
    else:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam)

    return 1 / len(y) * (np.linalg.norm(y_eval - y) ** 2)


def get_mse(self, settingnr=None, X=None, y=None, lam=None):
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
        return self._mse(settingnr, lam, X, y)
    else:
        return {
            l: self._mse(settingnr, l, X, y) for l in self.lam.keys()
        }


def _mad(self, settingnr, lam, X, y):  # helpfunction for get_mad
    if y is None:
        y = self.y

    if X is not None:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam, X=X)
    else:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam)

    return 1 / len(y) * np.linalg.norm(y_eval - y, ord=1)


def get_mad(self, settingnr=None, X=None, y=None, lam=None):
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
        return self._mad(settingnr, lam, X, y)
    else:
        return {
            l: self._mad(settingnr, l, X, y) for l in self.lam.keys()
        }


def _L2_error(self, norm, bc_fun, settingnr, lam):
    # print(settingnr)
    if self.getSetting(settingnr, lam).basis in {"per", "cos", "cheb", "std", "mixed"}:
        error = norm**2
        index_set = gt.get_IndexSet(self.getTrafo(settingnr, lam).settings, self.X.shape[1])

        for i in range(index_set.shape[1]):
            k = index_set[:, i]
            error += (
                abs(bc_fun(k) - self.getFc(settingnr, lam)[lam][i]) ** 2
                - abs(bc_fun(k)) ** 2
            )

        return np.sqrt(error) / norm
    else:
        raise NotImplementedError("L2 error is not implemented for this basis.")


def get_L2_error(self, norm, bc_fun, settingnr=None, lam=None):
    """
    Computes the relative L2 error of a function approximation for an `approx` object `a`.

    - If `a`, `norm`, `bc_fun`, and `lam` are provided,
      this function computes the relative L2 error for a specific regularization parameter `lam`.

    - If only `a`, `norm`, and `bc_fun` are provided,
    this function computes the relative L2 error for all available regularization parameters.
    """
    # print(settingnr)
    if lam is not None:
        return self._L2_error(norm, bc_fun, settingnr, lam)
    else:
        return {
            l: self._L2_error(norm, bc_fun, settingnr, l)
            for l in self.lam.keys()
        }


def _acc(self, settingnr, lam, X, y):  # helpfunction for get_acc
    if y is None:
        y = self.y

    if X is not None:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam, X=X)
    else:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam)

    return np.sum(np.sign(y_eval) == y) / len(y) * 100.0


def get_acc(self, settingnr=None, X=None, y=None, lam=None):

    if lam is not None:
        return self._acc(settingnr, lam, X, y)
    else:
        return {
            l: self._acc(settingnr, l, X, y) for l in self.lam.keys()
        }


def auc_score(y_true, y_pred_proba):
    combined_data = sorted(zip(y_pred_proba, y_true), key=lambda x: x[0], reverse=True)

    P = sum(y_true)
    N = len(y_true) - P

    tp, fp = 0, 0
    tpr, fpr = 0, 0
    auc = 0.0

    for i in range(len(combined_data)):
        score, label = combined_data[i]

        current_tp = 0
        current_fp = 0
        j = i
        while j < len(combined_data) and combined_data[j][0] == score:
            if combined_data[j][1] == 1:
                current_tp += 1
            else:
                current_fp += 1
            j += 1

        i = j - 1

        tp += current_tp
        fp += current_fp

        new_tpr = tp / P
        new_fpr = fp / N

        auc += (new_fpr - fpr) * (tpr + new_tpr) * 0.5

        tpr, fpr = new_tpr, new_fpr
    return auc


def _auc(self, settingnr, lam, X, y):
    if y is None:
        y = self.y

    if X is not None:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam, X=X)
    else:
        y_eval = self.evaluate(settingnr=settingnr, lam=lam)

    y_sc = (y_eval - np.min(y_eval)) / (np.max(y_eval) - np.min(y_eval))
    y = np.where(y == -1.0, 0, y)
    y = np.where(y == 1.0, 1, y)
    y_int = y.astype(np.int64)

    return auc_score(y_int, y_sc)


def get_auc(self, settingnr=None, X=None, y=None, lam=None):

    if lam is not None:
        return self._auc(settingnr, lam, X, y)
    else:
        return {
            l: self._auc(settingnr, l, X, y) for l in self.lam.keys()
        }


__all__ = [
    "_l2_error",
    "get_l2_error",
    "_mse",
    "get_mse",
    "_mad",
    "get_mad",
    "_L2_error",
    "get_L2_error",
    "_acc",
    "get_acc",
    "_auc",
    "get_auc",
]
