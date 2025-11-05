from pyGroupedTransforms.GroupedTransform import *  # TODO: Kann wahrscheinlich weg sobald in pyGroupedTransform GreoupedTransform exportiert wird

from pyANOVAapprox import *
from pyANOVAapprox.fista import *

bases = ["per", "cos", "cheb", "std", "chui1", "chui2", "chui3", "chui4", "mixed"]
types = {
    "per": complex,
    "cos": float,
    "cheb": float,
    "std": float,
    "chui1": float,
    "chui2": float,
    "chui3": float,
    "chui4": float,
    "mixed": complex,
}
gt_systems = {
    "per": "exp",
    "cos": "cos",
    "cheb": "cos",
    "std": "cos",
    "chui1": "chui1",
    "chui2": "chui2",
    "chui3": "chui3",
    "chui4": "chui4",
    "mixed": "mixed",
}

vtypes = {
    "per": complex,
    "cos": float,
    "cheb": float,
    "std": float,
    "chui1": float,
    "chui2": float,
    "chui3": float,
    "chui4": float,
    "mixed": complex,
}


def get_orderDependentBW(U, N):

    N_bw = [0] * len(U)

    for i in range(len(U)):
        if len(U[i]) == 0:
            N_bw[i] = 0
        else:
            N_bw[i] = N[len(U[i]) - 1]

    return N_bw


### approx:
# A struct to hold the scattered data function approximation.

## Fields
# * `basis::String` - basis of the function space; currently choice of `"per"` (exponential functions), `"cos"` (cosine functions), `"cheb"` (Chebyshev basis),`"std"`(transformed exponential functions), `"chui1"` (Haar wavelets), `"chui2"` (Chui-Wang wavelets of order 2),`"chui3"`  (Chui-Wang wavelets of order 3) ,`"chui4"` (Chui-Wang wavelets of order 4)
# * `X::Matrix{Float64}` - scattered data nodes with d rows and M columns
# * `y::Union{Vector{ComplexF64},Vector{Float64}}` - M function values (complex for `basis = "per"`, real ortherwise)
# * `U::Vector{Vector{Int}}` - a vector containing susbets of coordinate indices
# * `N::Vector{Int}` - bandwdiths for each ANOVA term
# * `trafo::GroupedTransform` - holds the grouped transformation
# * `fc::Dict{Float64,GroupedCoefficients}` - holds the GroupedCoefficients after approximation for every different regularization parameters

## Constructor
#    approx( X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, U::Vector{Vector{Int}}, N::Vector{Int}, basis::String = "cos" )

## Additional Constructor
#    approx( X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, ds::Int, N::Vector{Int}, basis::String = "cos" )


class approx:
    def __init__(
        self,
        X,
        y,
        U=None,
        N=None,
        basis="cos",
        classification=False,
        basis_vect=[],
        fastmult=None,
        ds=None,
    ):

        if N == None or len(N) == 0:
            ValueError("please define N")

        if (
            U == None or len(U) == 0
        ):  # setting U   #approx(X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, ds::Int, N::Vector{Int}, basis::String = "cos"; classification::Bool = false, basis_vect::Vector{String} = Vector{String}([]), fastmult::Bool = classification ? true : false,)
            U = get_superposition_set(X.shape[1], ds)

        if not isinstance(
            N[0], tuple
        ):  # setting N    #approx( X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, U::Vector{Vector{Int}}, N::Vector{Int}, basis::String = "cos"; classification::Bool = false, basis_vect::Vector{String} = Vector{String}([]), fastmult::Bool = classification ? true : false,)
            ds = max(len(u) for u in U)

            if len(N) != len(U) and len(N) != ds:
                raise ValueError("N needs to have |U| or max |u| entries.")
            if len(N) == ds:
                bw = get_orderDependentBW(U, N)
            else:
                bw = N

            bws = [None] * len(U)

            for i in range(len(U)):
                u = U[i]
                if len(u) == 0:
                    bws[i] = np.array([0] * len(u), np.int32)
                else:
                    bws[i] = np.array([bw[i]] * len(u), np.int32)

            N = bws

        else:
            N = [np.array(u, dtype=np.int32) for u in N]

        if basis_vect is None:
            basis_vect = []
        if fastmult is None:
            fastmult = True if classification else False
        if basis not in bases:
            raise ValueError("Basis not found.")
        if y[0].dtype != vtypes[basis]:
            raise TypeError(
                "Periodic functions require complex vectors, nonperiodic functions real vectors."
            )

        M = X.shape[0]

        if len(y) != M:
            raise ValueError("y needs as many entries as X has rows.")

        for i in range(len(U)):
            u = U[i]
            if len(u) > 0:
                if len(N[i]) != len(u):
                    raise ValueError(
                        f"Vector N has for the set {u} not the right length"
                    )

        if basis == "mixed":
            raise Error("mixed is not implemented yet")

        # if basis == "mixed":
        #    if len(basis_vect) == 0:
        #        raise ValueError("please call approx with basis_vect for a NFMT transform.")
        #    if len(basis_vect) < max(max(u) for u in U) +1:
        #        raise ValueError("basis_vect must have an entry for every dimension.")

        min_X = np.min(X)
        max_X = np.max(X)

        if basis in {"per", "chui1", "chui2", "chui3", "chui4"}:
            if min_X < -0.5 or max_X >= 0.5:
                raise ValueError("Nodes need to be between -0.5 and 0.5.")
        elif basis == "cos":
            if min_X < 0 or max_X > 1:
                raise ValueError("Nodes need to be between 0 and 1.")
        elif basis == "cheb":
            if min_X < -1 or max_X > 1:
                raise ValueError("Nodes need to be between -1 and 1.")

        Xt = X.copy()

        if basis == "cos":
            Xt /= 2
        elif basis == "cheb":
            Xt = np.arccos(Xt)
            Xt /= 2 * np.pi
        elif basis == "std":
            Xt /= sqrt(2)
            Xt = erf(Xt)
            Xt += 1
            Xt /= 4

        trafo = GroupedTransform(
            system=gt_systems[basis],
            U=U,
            N=N,
            X=Xt,
            fastmult=fastmult,
            basis_vect=basis_vect,
        )

        self.basis = basis
        self.X = X
        self.y = y
        self.U = U
        self.N = N
        self.trafo = trafo
        self.fc = {}
        self.classification = classification
        self.basis_vect = basis_vect
        self.fastmult = fastmult

    def Approximate(
        self, lam, max_iter=50, weights=None, verbose=False, solver=None, tol=1e-8
    ):  # helpfuntion for approximate
        M = self.X.shape[0]
        nf = get_NumFreq(self.trafo.settings)
        w = np.ones(nf, "float")

        if weights is not None:
            if len(weights) != nf or np.min(weights) < 1:
                raise ValueError("Weight requirements not fulfilled.")
            else:
                w = weights

        if self.basis in {"per", "mixed"}:
            what = GroupedCoefficients(self.trafo.settings, np.array(w, "complex"))
        else:
            what = GroupedCoefficients(self.trafo.settings, w)

        lambda_keys = list(self.fc)
        tmp = np.zeros(nf, dtype=types[self.basis])

        if solver is None:
            solver = "fista" if self.classification else "lsqr"

        if self.classification and solver != "fista":
            raise ValueError(
                "Classification is only implemented with the fista solver."
            )

        if solver == "lsqr":
            diag_w_sqrt = np.sqrt(lam) * np.sqrt(w)

            def matv(fhat):
                return np.concatenate(
                    (
                        self.trafo @ GroupedCoefficients(self.trafo.settings, fhat),
                        diag_w_sqrt * fhat,
                    )
                )

            def rmatv(f):
                return (self.trafo.H @ f[:M]).vec() + (diag_w_sqrt * f[M:])

            F_vec = DeferredLinearOperator(
                mfunc=matv, rmfunc=rmatv, shape=(M + nf, nf), dtype=types[self.basis]
            )

            tmp = lsqr(
                F_vec,
                np.concatenate((self.y, np.zeros(nf, dtype=types[self.basis]))),
                atol=tol,
                btol=tol,
                iter_lim=max_iter,
                show=verbose,
            )
            self.fc[lam] = GroupedCoefficients(self.trafo.settings, tmp[0])

        elif solver == "fista":
            ghat = GroupedCoefficients(self.trafo.settings, tmp)
            fista(
                ghat,
                self.trafo,
                self.y,
                lam,
                what,
                max_iter=max_iter,
                classification=self.classification,
            )
            self.fc[lam] = ghat
        else:
            raise ValueError("Solver not found.")

    def approximate(
        self, lam, max_iter=50, weights=None, verbose=False, solver=None, tol=1e-8
    ):
        """
        If lam is a np.ndarray of dtype float, this function computes the approximation for the regularization parameters contained in lam.
        If lam is a float, this function computes the approximation for the regularization parameter lam.
        """
        if (
            isinstance(lam, np.ndarray) and lam.dtype == float
        ):  # approximate( a::approx; lambda::Vector{Float64} = exp.(range(0, 5, length = 5)), max_iter::Int = 50, weights::Union{Vector{Float64},Nothing} = nothing, verbose::Bool = false, solver::String = "lsqr" )::Nothing
            lam = np.sort(lam)[::-1]
            for l in lam:
                self.Approximate(
                    l,
                    max_iter=max_iter,
                    weights=weights,
                    verbose=verbose,
                    solver=solver,
                    tol=tol,
                )

        elif isinstance(
            lam, float
        ):  # approximate( a::approx, λ::Float64; max_iter::Int = 50, weights::Union{Vector{Float64},Nothing} = nothing, verbose::Bool = false, solver::String = "lsqr", tol:.Float64b= 1e-8 )::Nothing
            self.Approximate(
                lam,
                max_iter=max_iter,
                weights=weights,
                verbose=verbose,
                solver=solver,
                tol=tol,
            )

        else:
            raise ValueError(
                lam, "has to be a float or a numpy array with float as dtype"
            )

    def evaluate(self, lam=None, X=None):
        """
        This function evaluates the approximation with optional node matrix X and regularization lam.

        - If both X and lam are given: evaluate at X for specific lam.
        - If only X is given: evaluate at X for all lam.
        - If only lam is given: evaluate at self.X for specific lam.
        - If neither are given: evaluate at self.X for all lam.
        """

        if X is not None:
            if self.basis == "per" and (np.min(X) < -0.5 or np.max(X) >= 0.5):
                raise ValueError("Nodes need to be between -0.5 and 0.5.")
            elif self.basis == "cos" and (np.min(X) < 0 or np.max(X) > 1):
                raise ValueError("Nodes need to be between 0 and 1.")
            elif self.basis == "cheb" and (np.min(X) < -1 or np.max(X) > 1):
                raise ValueError("Nodes need to be between -1 and 1.")

            Xt = np.copy(X)

            if self.basis == "cos":
                Xt /= 2
            elif self.basis == "cheb":
                Xt = np.arccos(Xt)
                Xt /= 2 * np.pi
            elif self.basis == "std":
                Xt /= np.sqrt(2)
                Xt = erf(Xt)
                Xt += 1
                Xt /= 4

            trafo = GroupedTransform(
                system=gt_systems[self.basis],
                U=self.U,
                N=self.N,
                X=Xt,
                basis_vect=self.basis_vect,
            )

            if (
                lam is not None
            ):  # evaluate( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Vector{ComplexF64},Vector{Float64}}
                return trafo @ self.fc[lam]
            else:  # evaluate( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}
                return {λ: trafo @ self.fc[λ] for λ in list(self.fc)}

        else:
            if (
                lam is not None
            ):  # evaluate( a::approx; λ::Float64 )::Union{Vector{ComplexF64},Vector{Float64}}
                return self.trafo @ self.fc[lam]
            else:  # evaluate( a::approx )::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}
                return {λ: self.trafo @ self.fc[λ] for λ in list(self.fc)}

    def evaluateANOVAterms(self, X, lam=None):
        """
        This function evaluates the single ANOVA terms of the approximation on the nodes of matrix X and regularization lam.

        - If lam is given: evaluate at X for specific lam.
        - If lam is not given: evaluate at X for all lam.
        """

        if self.basis == "per" and (np.min(X) < -0.5 or np.max(X) >= 0.5):
            raise ValueError("Nodes need to be between -0.5 and 0.5.")
        elif self.basis == "cos" and (np.min(X) < 0 or np.max(X) > 1):
            raise ValueError("Nodes need to be between 0 and 1.")
        elif self.basis == "cheb" and (np.min(X) < -1 or np.max(X) > 1):
            raise ValueError("Nodes need to be between -1 and 1.")

        Xt = np.copy(X)

        if self.basis == "cos":
            Xt /= 2
        elif self.basis == "cheb":
            Xt = np.arccos(Xt)
            Xt /= 2 * pi
        elif self.basis == "std":
            Xt /= sqrt(2)
            Xt = erf(Xt)
            Xt += 1
            Xt /= 4

        if self.basis == "per":
            values = np.zeros((Xt.shape[0], len(self.U)), "complex")
        else:
            values = np.zeros((Xt.shape[0], len(self.U)), "float")

        trafo = GroupedTransform(
            system=gt_systems[self.basis],
            U=self.U,
            N=self.N,
            X=Xt,
            basis_vect=self.basis_vect,
        )

        if (
            lam is not None
        ):  # evaluateANOVAterms( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Matrix{ComplexF64},Matrix{Float64}}
            for j, u in enumerate(self.U):
                values[:, j] = trafo[u] @ self.fc[lam][u]
            return values
        else:  # evaluateANOVAterms( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}
            results = {}
            for λ in list(self.fc):
                vals = np.zeros_like(values)
                for j, u in enumerate(self.U):
                    vals[:, j] = trafo[u] @ self.fc[λ][u]
                results[λ] = vals
            return results

    def evaluateSHAPterms(self, X, lam=None):
        """
        This function evaluates for each dimension the Shapley contribution to the overall approximation on the nodes of matrix X and regularization lam.

        - If lam is given: evaluate at X for specific lam.
        - If lam is not given: evaluate at X for all lam.
        """

        if self.basis == "per" and (np.min(X) < -0.5 or np.max(X) >= 0.5):
            raise ValueError("Nodes need to be between -0.5 and 0.5.")
        elif self.basis == "cos" and (np.min(X) < 0 or np.max(X) > 1):
            raise ValueError("Nodes need to be between 0 and 1.")
        elif self.basis == "cheb" and (np.min(X) < -1 or np.max(X) > 1):
            raise ValueError("Nodes need to be between -1 and 1.")

        d = X.shape[1]
        M = X.shape[0]

        if (
            lam is not None
        ):  # evaluateSHAPterms( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Matrix{ComplexF64},Matrix{Float64}}
            terms = self.evaluateANOVAterms(X, lam)

            Dtype = np.complex128 if self.basis == "per" else np.float64
            values = np.zeros((M, d), dtype=Dtype)

            for i in range(d):
                for j, u in enumerate(self.U):
                    if i in u:
                        values[:, i] += terms[:, j] / len(u)
            return values

        else:  # evaluateSHAPterms( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}
            results = {}
            for l in list(self.fc):
                terms = self.evaluateANOVAterms(X, l)

                Dtype = np.complex128 if self.basis == "per" else np.float64
                values = np.zeros((M, d), dtype=Dtype)

                for i in range(d):
                    for j, u in enumerate(self.U):
                        if i in u:
                            values[:, i] += terms[:, j] / len(u)
                results[l] = values

            return results
