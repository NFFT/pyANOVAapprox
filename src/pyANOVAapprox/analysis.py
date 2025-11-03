from pyANOVAapprox import *


def get_variance(a, lam, Dict):  # helpfunction for get_variances

    if a.basis.startswith("chui"):
        variances = a.fc[lam].norms(Dict=False, m=int(a.basis[-1]))
    else:
        variances = a.fc[lam].norms()

    variances = variances[1:]
    if Dict:
        if a.basis.startswith("chui"):
            variances = a.fc[lam].norms(Dict=True, m=int(a.basis[-1]))
        else:
            variances = a.fc[lam].norms(Dict=True)

        return {u: variances[u] for u in list(variances)}
    else:
        return variances


def get_variances(a, lam=None, Dict=False):
    """
    This function returns the variances of the approximated ANOVA terms for all ``\lambda``, if lam == None. Otherwise for the provided lam. Depending on Dict, it returns the approximated ANOVA terms as a vector or as a dict.
    """
    if isinstance(lam, float):
        return get_variance(
            a=a, lam=lam, Dict=Dict
        )  # get_variances(a::approx, λ::Float64; dict::Bool=false,)::Union{Vector{Float64},Dict{Vector{Int},Float64}}

    elif (
        lam == None
    ):  # get_variances( a::approx; dict::Bool = false )::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}
        return {l: get_variance(a=a, lam=l, Dict=Dict) for l in list(a.fc)}


def _GSI(a, lam, Dict):  # helpfunction for get_GSI

    if a.basis.startswith("chui"):
        variances = np.square(a.fc[lam].norms(Dict=False, m=int(a.basis[-1])))
    else:
        variances = np.square(a.fc[lam].norms())

    variances = variances[1:]
    variance_f = sum(variances)

    if Dict:
        if a.basis.startswith("chui"):
            variances = np.square(a.fc[lam].norms(Dict=True, m=int(a.basis[-1])))
        else:
            variances = a.fc[lam].norms(Dict=True)
        return {u: (variances[u] ** 2) / variance_f for u in list(variances)}

    else:
        return variances / variance_f


def get_GSI(a, lam=None, Dict=False):
    """
    This function returns the global sensitivity indices of the approximation for all ``\lambda``, if lam == None. Otherwise for the provided lam. Depending on Dict, it returns the approximated ANOVA terms as a vector or as a dict.
    """

    if (
        lam is not None
    ):  # get_GSI(a::approx, λ::Float64; dict::Bool = false,)::Union{Vector{Float64},Dict{Vector{Int},Float64}}
        return _GSI(a=a, lam=lam, Dict=Dict)
    else:  # get_GSI( a::approx; dict::Bool = false )::Dict{Float64,Union{Vector{Float64},Dict{Vector{Int},Float64}}}
        return {l: _GSI(a, l, Dict) for l in list(a.fc)}


def lam_AttributeRanking(a, lam):  # helpfunction foor get_AttributeRanking

    d = a.X.shape[1]
    gsis = get_GSI(a, lam, Dict=True)
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


def get_AttributeRanking(a, lam=None):
    """
    This function returns the attribute ranking of the approximation for all reg. parameters ``\lambda``, if lam == None, as a dictionary of vectors of length `a.d`. Otherwise for the provided lam as a vector of length `a.d`.
    """
    if (
        lam == None
    ):  # get_AttributeRanking( a::approx, λ::Float64 )::Dict{Float64,Vector{Float64}}
        return {l: get_AttributeRanking(a, l) for l in list(a.fc)}
    else:  # get_AttributeRanking( a::approx, λ::Float64 )::Vector{Float64}
        return lam_AttributeRanking(a=a, lam=lam)


def lam_ActiveSet(a, eps, lam):  # helpfunction for get_ActiveSet

    U = a.U[1:]
    lengths = [len(u) for u in U]
    ds = max(lengths)

    if len(eps) != ds:
        raise ValueError("Entries in vector eps have to be ds.")

    gsi = get_GSI(a, lam)

    n = 0

    for i in range(len(gsi)):
        if gsi[i] > eps[len(U[i]) - 1]:
            n += 1

    U_active = [None] * (n + 1)
    U_active[0] = []

    idx = 1
    for i in range(len(gsi)):
        if gsi[i] > eps[len(U[i]) - 1]:
            U_active[idx] = U[i]
            idx += 1

    return U_active


def get_ActiveSet(a, eps, lam=None):

    if (
        lam == None
    ):  # get_ActiveSet(a::approx, eps::Vector{Float64})::Dict{Float64,Vector{Vector{Int}}}
        return {l: lam_ActiveSet(a=a, eps=eps, lam=l) for l in list(a.fc)}
    else:  # get_ActiveSet(a::approx, eps::Vector{Float64}, λ::Float64)::Vector{Vector{Int}}
        return lam_ActiveSet(a=a, eps=eps, lam=lam)


def lam_ShapleyValues(a, lam):  # helpfunction for get_ShapleyValues

    d = a.X.shape[1]
    vars_dict = get_variances(a, lam, Dict=True)
    U = list(vars_dict)
    r = np.zeros(d)

    for i in range(d):
        for v in U:
            if i in v:
                r[i] += (vars_dict[v] ** 2) / len(v)

    return r


def get_ShapleyValues(a, lam=None):
    """
    This function returns the Shapley values of the approximation for all reg. parameters ``\lambda``, if lam == None, as a dictionary of vectors of length `a.d`. Otherwise for the provided lam as a vector of length `a.d`.
    """
    if lam == None:  # get_ShapleyValues(a::approx)::Dict{Float64,Vector{Float64}}
        return {l: lam_ShapleyValues(a=a, lam=l) for l in list(a.fc)}
    else:  # get_ShapleyValues(a::approx, λ::Float64)::Vector{Float64}
        return lam_ShapleyValues(a=a, lam=lam)
