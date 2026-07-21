"""Utilities for analyzing an ANOVA approximation.

This module provides functions for computing variances, global sensitivity
indices, attribute rankings, active sets and Shapley values from an
:py:class:`~pyANOVAapprox.approx.approx` object.

The functions are bound as methods onto :py:class:`~pyANOVAapprox.approx.approx`,
so their first argument ``self`` is the approximation object. They are also
available as plain functions from the package root, where the same argument is
named ``a``::

    import pyANOVAapprox as anova

    anova.get_variances(a, lam=1e-3)   # equivalent to a.get_variances(lam=1e-3)
"""

# from pyANOVAapprox import *
import numpy as np


def _variances(self, settingnr, lam, Dict):  # helpfunction for get_variances
    setting = self.getSetting(settingnr, lam)
    if setting.basis.startswith("chui"):
        variances = self.getFc(settingnr, lam)[lam].norms(
            Dict=False, m=int(setting.basis[-1])
        )
    else:
        variances = self.getFc(settingnr, lam)[lam].norms()

    variances = variances[1:]
    if Dict:
        if a.basis.startswith("chui"):
            variances = self.getFc(settingnr, lam)[lam].norms(
                Dict=True, m=int(setting.basis[-1])
            )
        else:
            variances = self.getFc(settingnr, lam)[lam].norms(Dict=True)

        return {u: variances[u] for u in list(variances)}
    else:
        return variances


def get_variances(self, settingnr=None, lam=None, Dict=False):
    r"""Compute the variances of all ANOVA terms in the approximation.

    The variance of an ANOVA term measures how much that term contributes to
    the total variance of the approximated function.

    Args:
        self: The approximation object.
        settingnr (int, optional): Index of the approximation setting to use.
            Defaults to the only setting if just one is present.
        lam (float, optional): Regularization parameter. If omitted, the
            variances are computed for every available :math:`\lambda`.
        Dict (bool): If ``True``, return the variances as a dictionary indexed
            by ANOVA terms. Otherwise return them as an array.
            Defaults to ``False``.

    Returns:
        dict or numpy.ndarray: Variances of the ANOVA terms for the requested
        ``lam``. If ``lam`` is omitted, a dictionary mapping each
        :math:`\lambda` to the corresponding result.
    """
    if isinstance(lam, float):
        return self._variances(
            settingnr, lam, Dict
        )  # get_variances(a::approx, λ::Float64; dict::Bool=false,)::Union{Vector{Float64},Dict{Vector{Int},Float64}}

    elif (
        lam == None
    ):  # get_variances( a::approx; dict::Bool = false )::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}
        return {l: self._variances(settingnr, l, Dict) for l in self.lam.keys()}


def _GSI(self, settingnr, lam, Dict):  # helpfunction for get_GSI
    setting = self.getSetting(settingnr, lam)
    if setting.basis.startswith("chui"):
        variances = np.square(
            self.getFc(settingnr, lam)[lam].norms(Dict=False, m=int(setting.basis[-1]))
        )
    else:
        variances = np.square(self.getFc(settingnr, lam)[lam].norms())

    variances = variances[1:]
    variance_f = sum(variances)

    if Dict:
        if setting.basis.startswith("chui"):
            variances = self.getFc(settingnr, lam)[lam].norms(
                Dict=True, m=int(setting.basis[-1])
            )
        else:
            variances = self.getFc(settingnr, lam)[lam].norms(Dict=True)
        return {u: (variances[u] ** 2) / variance_f for u in list(variances)}

    else:
        return variances / variance_f


def get_GSI(self, settingnr=None, lam=None, Dict=False):
    r"""Compute the Global Sensitivity Indices (GSIs) of the approximation.

    A Global Sensitivity Index represents the fraction of the total variance
    explained by each ANOVA term.

    Args:
        self: The approximation object.
        settingnr (int, optional): Index of the approximation setting to use.
        lam (float, optional): Regularization parameter. If omitted, the GSIs
            are computed for every available :math:`\lambda`.
        Dict (bool): If ``True``, return the indices as a dictionary indexed by
            ANOVA terms. Otherwise return them as an array. Defaults to
            ``False``.

    Returns:
        dict or numpy.ndarray: Global Sensitivity Indices of the ANOVA terms
        for the requested ``lam``. If ``lam`` is omitted, a dictionary mapping
        each :math:`\lambda` to the corresponding result.
    """

    if (
        lam is not None
    ):  # get_GSI(a::approx, λ::Float64; dict::Bool = false,)::Union{Vector{Float64},Dict{Vector{Int},Float64}}
        return self._GSI(settingnr, lam, Dict)
    else:  # get_GSI( a::approx; dict::Bool = false )::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}
        return {l: self._GSI(settingnr, l, Dict) for l in self.lam.keys()}


def _AttributeRanking(self, settingnr, lam):  # helpfunction for get_AttributeRanking

    d = self.X.shape[1]
    gsis = self.get_GSI(settingnr, lam, Dict=True)
    U = list(gsis)
    lengths = [len(u) for u in U]
    ds = max(lengths)

    factors = np.zeros((d, ds), dtype=int)

    for i in range(d):
        for j in range(ds):
            for v in U:
                if (i in v) and (len(v) == j + 1):
                    factors[i, j] += 1

    r = np.zeros(d, dtype=float)
    nf = 0.0

    for u in U:
        weights = 0.0
        for s in u:
            contribution = gsis[u] * (1.0 / factors[s, len(u) - 1])
            r[s] += contribution
            weights += 1.0 / factors[s, len(u) - 1]
        nf += weights * gsis[u]

    return r / nf


def get_AttributeRanking(self, settingnr=None, lam=None):
    r"""Rank the input variables by their influence on the approximation.

    Higher-ranked variables contribute more strongly to the approximated
    function. The ranking is derived from the global sensitivity indices by
    distributing each ANOVA term's index over the variables it involves.

    Args:
        self: The approximation object.
        settingnr (int, optional): Index of the approximation setting to use.
        lam (float, optional): Regularization parameter. If omitted, the
            ranking is computed for every available :math:`\lambda`.

    Returns:
        dict or numpy.ndarray: For a given ``lam``, an array of length ``d``.
        If ``lam`` is omitted, a dictionary mapping each :math:`\lambda` to
        such an array.
    """
    if (
        lam is None
    ):  # get_AttributeRanking( a::approx, λ::Float64 )::Dict{Float64,Vector{Float64}}
        return {l: self._AttributeRanking(settingnr, l) for l in self.lam.keys()}
    else:  # get_AttributeRanking( a::approx, λ::Float64 )::Vector{Float64}
        return self._AttributeRanking(settingnr, lam)


def _ActiveSet(self, eps, settingnr, lam):  # helpfunction for get_ActiveSet

    U = self.getSetting(settingnr, lam).U[1:]
    lengths = [len(u) for u in U]
    ds = max(lengths)

    if len(eps) != ds:
        raise ValueError("Entries in vector eps have to be ds.")

    gsi = self.get_GSI(settingnr, lam)

    n = 0

    for i in range(len(gsi)):
        if gsi[i] > eps[len(U[i]) - 1]:
            n += 1

    U_active = [None] * (n + 1)
    U_active[0] = ()

    idx = 1
    for i in range(len(gsi)):
        if gsi[i] > eps[len(U[i]) - 1]:
            U_active[idx] = U[i]
            idx += 1

    return U_active


def get_ActiveSet(self, eps, settingnr=None, lam=None):
    r"""Select the ANOVA terms whose sensitivity exceeds a per-order threshold.

    A term ``u`` is kept when its global sensitivity index is larger than
    ``eps[len(u) - 1]``, i.e. the threshold is chosen according to the
    interaction order of the term. The returned set always starts with the
    empty term ``()``.

    Args:
        self: The approximation object.
        eps (numpy.ndarray): Thresholds, one per interaction order. Its length
            must equal the maximum interaction order ``ds`` of the setting.
        settingnr (int, optional): Index of the approximation setting to use.
        lam (float, optional): Regularization parameter. If omitted, the active
            set is computed for every available :math:`\lambda`.

    Returns:
        dict or list: For a given ``lam``, the list of retained ANOVA terms.
        If ``lam`` is omitted, a dictionary mapping each :math:`\lambda` to
        such a list.

    Raises:
        ValueError: If ``len(eps)`` does not match the maximum interaction
            order of the setting.
    """
    if (
        lam is None
    ):  # get_ActiveSet(a::approx, eps::Vector{Float64})::Dict{Float64,Vector{Vector{Int}}}
        return {l: self._ActiveSet(eps, settingnr, l) for l in self.lam.keys()}
    else:  # get_ActiveSet(a::approx, eps::Vector{Float64}, λ::Float64)::Vector{Vector{Int}}
        return self._ActiveSet(eps, settingnr, lam)


def _ShapleyValues(self, settingnr, lam):  # helpfunction for get_ShapleyValues

    d = self.X.shape[1]
    vars_dict = get_variances(self, settingnr, lam, Dict=True)
    U = list(vars_dict)
    r = np.zeros(d)

    for i in range(d):
        for v in U:
            if i in v:
                r[i] += (vars_dict[v] ** 2) / len(v)

    return r


def get_ShapleyValues(self, settingnr=None, lam=None):
    r"""Compute the Shapley values of the approximation.

    Shapley values provide a fair allocation of the total contribution of the
    approximation among the individual input variables, by distributing the
    contributions of all ANOVA interaction terms.

    Args:
        self: The approximation object.
        settingnr (int, optional): Index of the approximation setting to use.
        lam (float, optional): Regularization parameter. If omitted, the
            Shapley values are computed for every available :math:`\lambda`.

    Returns:
        dict or numpy.ndarray: For a given ``lam``, an array of length ``d``.
        If ``lam`` is omitted, a dictionary mapping each :math:`\lambda` to
        such an array.
    """
    if lam is None:  # get_ShapleyValues(a::approx)::Dict{Float64,Vector{Float64}}
        return {l: self._ShapleyValues(settingnr, l) for l in self.lam.keys()}
    else:  # get_ShapleyValues(a::approx, λ::Float64)::Vector{Float64}
        return self._ShapleyValues(settingnr, lam)


_all = [
    "_variances",
    "get_variances",
    "_GSI",
    "get_GSI",
    "_AttributeRanking",
    "get_AttributeRanking",
    "_ActiveSet",
    "get_ActiveSet",
    "_ShapleyValues",
    "get_ShapleyValues",
]
