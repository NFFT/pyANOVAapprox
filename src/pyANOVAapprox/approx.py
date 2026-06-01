import copy
import warnings

import pyANOVAapprox.analysis as ANOVAanalysis
import pyANOVAapprox.bandwidth as ANOVAbandwidth
import pyANOVAapprox.errors as ANOVAerrors
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


def transformX(X, basis):
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
    return Xt


compute_bandwidth = ANOVAbandwidth.compute_bandwidth


class approx_setting:
    def __init__(
        self,
        parent,
        U=None,
        N=None,
        basis="cos",
        classification=False,
        basis_vect=[],
        fastmult=None,
        parallel=True,
        ds=None,
        lam={0.0},
    ):

        # if N == None or len(N) == 0:
        #    ValueError("please define N")

        if (
            U == None or len(U) == 0
        ):  # setting U   #approx(X::Matrix{Float64}, y::Union{Vector{ComplexF64},Vector{Float64}}, ds::Int, N::Vector{Int}, basis::String = "cos"; classification::Bool = false, basis_vect::Vector{String} = Vector{String}([]), fastmult::Bool = classification ? true : false,)
            U = get_superposition_set(parent.X.shape[1], ds)

        if N is not None and not isinstance(
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

        elif N is not None and isinstance(N[0], tuple):
            N = [np.array(u, dtype=np.int32) for u in N]

        if basis_vect is None:
            basis_vect = []
        if fastmult is None:
            fastmult = False if classification else True
        if basis not in bases:
            raise ValueError("Basis not found.")
        # if y[0].dtype != vtypes[basis]:
        #    raise TypeError(
        #        "Periodic functions require complex vectors, nonperiodic functions real vectors."
        #    )

        # if basis == "mixed":
        #    if len(basis_vect) == 0:
        #        raise ValueError("please call approx with basis_vect for a NFMT transform.")
        #    if len(basis_vect) < max(max(u) for u in U) +1:
        #        raise ValueError("basis_vect must have an entry for every dimension.")

        self.basis = basis
        self.U = U
        self.N = N
        self.classification = classification
        self.basis_vect = basis_vect
        self.fastmult = fastmult
        self.parallel = parallel
        self.lam = lam
        self.parent = parent


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
        parallel=True,
        ds=None,
        lam={0.0},
    ):

        if type(lam) is np.ndarray:
            lam = {float(i) for i in lam}
        if type(lam) is list:
            lam = set(lam)

        self.X = X

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different lengths.")

        setting = approx_setting(
            parent=self,
            U=U,
            N=N,
            basis=basis,
            classification=classification,
            basis_vect=basis_vect,
            fastmult=fastmult,
            parallel=parallel,
            ds=ds,
            lam=lam,
        )

        min_X = np.min(X)
        max_X = np.max(X)

        if setting.basis in {"per", "chui1", "chui2", "chui3", "chui4"}:
            if min_X < -0.5 or max_X >= 0.5:
                raise ValueError("Nodes need to be between -0.5 and 0.5.")
        elif setting.basis == "cos":
            if min_X < 0 or max_X > 1:
                raise ValueError("Nodes need to be between 0 and 1.")
        elif setting.basis == "cheb":
            if min_X < -1 or max_X > 1:
                raise ValueError("Nodes need to be between -1 and 1.")

        self.y = y
        self.setting = [setting]
        self.trafo = [None]
        self.fc = [{}]
        self.aktsetting = 0
        self.lam = {}
        for i in lam:
            self.lam[i] = [0]

        if setting.N is not None:
            self.addTrafo()

    def addSetting(self, setting):
        settingnr = len(self.setting)
        self.setting.append(copy.copy(setting))
        self.aktsetting = settingnr
        self.trafo.append(None)
        self.fc.append({})

    def getSettingNr(self, settingnr=None, lam=None, warn=True):
        if lam is None:
            return self.aktsetting if settingnr is None else settingnr
        else:
            if lam in self.lam.keys():
                return self.lam[lam][-1] if settingnr is None else settingnr
            else:
                warnings.warn("Not yet approximated for this lambda")
                return self.aktsetting if settingnr is None else settingnr

    def getSetting(self, settingnr=None, lam=None):
        return self.setting[self.getSettingNr(settingnr, lam)]

    def getTrafo(self, settingnr=None, lam=None):
        return self.trafo[self.getSettingNr(settingnr, lam)]

    def getFc(self, settingnr=None, lam=None):
        return self.fc[self.getSettingNr(settingnr, lam)]

    def addTrafo(self, settingnr=None):
        setting = self.getSetting(settingnr)
        if settingnr is None:
            settingnr = self.aktsetting
        self.trafo[settingnr] = GroupedTransform(
            system=gt_systems[setting.basis],
            U=setting.U,
            N=setting.N,
            X=transformX(self.X, setting.basis),
            fastmult=setting.fastmult,
            parallel=setting.parallel,
            basis_vect=setting.basis_vect,
        )

    def _approximate(
        self, lam, settingnr, max_iter, weights, verbose, solver, tol
    ):  # helpfuntion for approximate
        setting = self.getSetting(settingnr)

        if setting.N is None:
            raise ValueError("N is not set. You may want to use autoapproximate")

        M = self.X.shape[0]
        nf = get_NumFreq(self.getTrafo(settingnr).settings)
        w = np.ones(nf, "float")

        if weights is not None:
            if len(weights) != nf or np.min(weights) < 1:
                raise ValueError("Weight requirements not fulfilled.")
            else:
                w = weights

        if setting.basis in {"per", "mixed"}:
            what = GroupedCoefficients(
                self.getTrafo(settingnr).settings, np.array(w, "complex")
            )
        else:
            what = GroupedCoefficients(self.getTrafo(settingnr).settings, w)

        tmp = np.zeros(nf, dtype=types[setting.basis])

        if solver is None:
            solver = "fista" if setting.classification else "lsqr"

        if setting.classification and solver != "fista":
            raise ValueError(
                "Classification is only implemented with the fista solver."
            )

        if solver == "lsqr":
            diag_w_sqrt = np.sqrt(lam) * np.sqrt(w)

            def matv(fhat):
                return np.concatenate(
                    (
                        self.getTrafo(settingnr)
                        @ GroupedCoefficients(self.getTrafo(settingnr).settings, fhat),
                        diag_w_sqrt * fhat,
                    )
                )

            def rmatv(f):
                return (self.getTrafo(settingnr).H @ f[:M]).vec() + (
                    diag_w_sqrt * f[M:]
                )

            F_vec = DeferredLinearOperator(
                mfunc=matv, rmfunc=rmatv, shape=(M + nf, nf), dtype=types[setting.basis]
            )

            tmp = lsqr(
                F_vec,
                np.concatenate((self.y, np.zeros(nf, dtype=types[setting.basis]))),
                atol=tol,
                btol=tol,
                iter_lim=max_iter,
                show=verbose,
            )
            self.getFc(settingnr)[lam] = GroupedCoefficients(
                self.getTrafo(settingnr).settings, tmp[0]
            )

        elif solver == "fista":
            ghat = GroupedCoefficients(self.getTrafo(settingnr).settings, tmp)
            fista(
                ghat,
                self.getTrafo(settingnr),
                self.y,
                lam,
                what,
                max_iter=max_iter,
                classification=setting.classification,
            )
            self.getFc(settingnr)[lam] = ghat
        else:
            raise ValueError("Solver not found.")

    def approximate(
        self,
        lam=None,
        settingnr=None,
        max_iter=50,
        weights=None,
        verbose=False,
        solver=None,
        tol=1e-8,
    ):
        """
        If lam is a np.ndarray of dtype float, this function computes the approximation for the regularization parameters contained in lam.
        If lam is a float, this function computes the approximation for the regularization parameter lam.
        """

        setting = self.getSetting(settingnr)

        if lam is None:
            lam = setting.lam
        else:
            if isinstance(lam, np.ndarray):
                lam = {float(l) for l in lam}
            if not isinstance(lam, set):
                lam = {lam}
            setting.lam = lam

        for l in lam:
            self._approximate(
                l,
                settingnr=settingnr,
                max_iter=max_iter,
                weights=weights,
                verbose=verbose,
                solver=solver,
                tol=tol,
            )

    estimate_rates = ANOVAbandwidth.estimate_rates

    def _autoapproximate(
        self,
        lam,
        settingnr,
        B,
        maxiter,
        solver,
        verbosity,
        solver_max_iter,
        solver_weights,
        solver_verbose,
        solver_tol,
    ):
        settingnrs = []
        setting = self.getSetting(settingnr)

        n = len(self.y)
        if B is None:
            B = bisect(lambda x: x * math.log(x) - n, 1, n)

        D = dict([(u, tuple([1.0] * len(u))) for u in setting.U])
        t = dict([(u, tuple([1.0] * len(u))) for u in setting.U])
        fix = dict([(u, tuple([False] * len(u))) for u in setting.U])
        bw =  [tuple([0] * len(u)) for u in setting.U]

        if verbosity > 3:
            if not os.path.exists("log"):
                os.mkdir("log")
            with open("log/log.csv", "w", newline="") as csvfile:
                csv.writer(csvfile, delimiter=",").writerow(["it"] + setting.U)
        if verbosity > 2:
            print("B =", B)
        for idx in range(maxiter):
            if verbosity > 0:
                print("===== Iteration ", str(idx + 1), " =====")
            bw, fix = compute_bandwidth(B, D, t, fix, bw)
            if setting.N is not None:
                self.addSetting(setting)
                settingnr = self.aktsetting
                settingnrs = settingnrs + [settingnr]
                setting = self.getSetting(settingnr)

            setting.N = [bw[i] for i in setting.U]
            self.addTrafo()

            if verbosity > 0:
                for i in setting.U:
                    print("bw in", str(i), ":", bw[i])
                # print("bw in iteration", str(idx + 1), "are", str(bw))
                print()
            if verbosity > 3:
                with open("log/log.csv", "a", newline="") as csvfile:
                    csv.writer(csvfile, delimiter=",").writerow(
                        ["bw in it" + str(idx + 1)] + [str(bw[i]) for i in setting.U]
                    )
            self.approximate(
                lam=lam,
                solver=solver,
                max_iter=solver_max_iter,
                weights=solver_weights,
                verbose=solver_verbose,
                tol=solver_tol,
            )

            D, t = self.estimate_rates(lam=lam, verbosity=verbosity)
            if verbosity > 1:
                for i in setting.U:
                    print(
                        "estimated rates for",
                        str(i),
                        ": D = ",
                        str(D[i]),
                        "and t = ",
                        str(t[i]),
                    )
                print()

                # print(
                #    "estimated rates in iteration",
                #    str(idx + 1),
                #    "are D =",
                #    str(D),
                #    "and t =",
                #    str(t),
                # )
            if verbosity > 3:
                with open("log/log.csv", "a", newline="") as csvfile:
                    wr = csv.writer(csvfile, delimiter=",")
                    wr.writerow(
                        ["D in it" + str(idx + 1)] + [str(D[i]) for i in setting.U]
                    )
                    wr.writerow(
                        ["t in it" + str(idx + 1)] + [str(t[i]) for i in setting.U]
                    )

            if all([math.isnan(j) for j in sum([D[i] for i in D], [])]) and all(
                [math.isnan(j) for j in sum([t[i] for i in D], [])]
            ):
                break
        return settingnrs

    def autoapproximate(
        self,
        lam=None,
        settingnr=None,
        B=None,
        maxiter=2,
        solver="lsqr",
        verbosity=0,
        solver_max_iter=50,
        solver_weights=None,
        solver_verbose=False,
        solver_tol=1e-8,
    ):
        settingnr = self.getSettingNr(settingnr)
        setting = self.getSetting(settingnr)

        if lam is None:
            lam = setting.lam
        else:
            if isinstance(lam, np.ndarray):
                lam = {float(l) for l in lam}
            if not isinstance(lam, set):
                lam = {lam}
            setting.lam = lam

        for l in lam:
            settingnrs = self._autoapproximate(
                l,
                settingnr=settingnr,
                B=B,
                maxiter=maxiter,
                solver=solver,
                verbosity=verbosity,
                solver_max_iter=solver_max_iter,
                solver_weights=solver_weights,
                solver_verbose=solver_verbose,
                solver_tol=solver_tol,
            )
            self.lam[l] = self.lam[l] + settingnrs

    def evaluate(self, settingnr=None, lam=None, X=None):
        """
        This function evaluates the approximation with optional node matrix X and regularization lam.

        - If both X and lam are given: evaluate at X for specific lam.
        - If only X is given: evaluate at X for all lam.
        - If only lam is given: evaluate at self.X for specific lam.
        - If neither are given: evaluate at self.X for all lam.
        """
        setting = self.getSetting(settingnr, lam)

        if X is not None:
            if setting.basis == "per" and (np.min(X) < -0.5 or np.max(X) >= 0.5):
                raise ValueError("Nodes need to be between -0.5 and 0.5.")
            elif setting.basis == "cos" and (np.min(X) < 0 or np.max(X) > 1):
                raise ValueError("Nodes need to be between 0 and 1.")
            elif setting.basis == "cheb" and (np.min(X) < -1 or np.max(X) > 1):
                raise ValueError("Nodes need to be between -1 and 1.")

            trafo = GroupedTransform(
                system=gt_systems[setting.basis],
                U=setting.U,
                N=setting.N,
                X=transformX(X, setting.basis),
                basis_vect=setting.basis_vect,
            )
        else:
            trafo = self.getTrafo(settingnr, lam)

        if (
            lam is not None
        ):  # evaluate( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Vector{ComplexF64},Vector{Float64}}
            return trafo @ self.getFc(settingnr, lam)[lam]
        else:  # evaluate( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Vector{ComplexF64},Vector{Float64}}}
            return {λ: trafo @ self.getFc(settingnr, lam)[λ] for λ in self.lam.keys()}

    def evaluateANOVAterms(self, settingnr=None, X=None, lam=None):
        """
        This function evaluates the single ANOVA terms of the approximation on the nodes of matrix X and regularization lam.

        - If lam is given: evaluate at X for specific lam.
        - If lam is not given: evaluate at X for all lam.
        """
        setting = self.getSetting(settingnr, lam)
        if X is None:
            X = self.X

        if setting.basis == "per" and (np.min(X) < -0.5 or np.max(X) >= 0.5):
            raise ValueError("Nodes need to be between -0.5 and 0.5.")
        elif setting.basis == "cos" and (np.min(X) < 0 or np.max(X) > 1):
            raise ValueError("Nodes need to be between 0 and 1.")
        elif setting.basis == "cheb" and (np.min(X) < -1 or np.max(X) > 1):
            raise ValueError("Nodes need to be between -1 and 1.")

        Xt = transformX(X, setting.basis)

        if self.getSetting(settingnr).basis == "per":
            values = np.zeros(
                (Xt.shape[0], len(self.getSetting(settingnr).U)), "complex"
            )
        else:
            values = np.zeros((Xt.shape[0], len(self.getSetting(settingnr).U)), "float")

        trafo = GroupedTransform(
            system=gt_systems[setting.basis],
            U=setting.U,
            N=setting.N,
            X=Xt,
            basis_vect=setting.basis_vect,
        )

        if (
            lam is not None
        ):  # evaluateANOVAterms( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Matrix{ComplexF64},Matrix{Float64}}
            for j, u in enumerate(setting.U):
                values[:, j] = trafo[u] @ self.getFc(settingnr, lam)[lam][u]
            return values
        else:  # evaluateANOVAterms( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}
            results = {}
            for λ in self.lam.keys():
                vals = np.zeros_like(values)
                for j, u in enumerate(setting.U):
                    vals[:, j] = trafo[u] @ self.getFc(settingnr, lam)[λ][u]
                results[λ] = vals
            return results

    def evaluateSHAPterms(self, settingnr=None, X=None, lam=None):
        """
        This function evaluates for each dimension the Shapley contribution to the overall approximation on the nodes of matrix X and regularization lam.

        - If lam is given: evaluate at X for specific lam.
        - If lam is not given: evaluate at X for all lam.
        """
        setting = self.getSetting(settingnr, lam)

        if X is None:
            X = self.X
        if setting.basis == "per" and (np.min(X) < -0.5 or np.max(X) >= 0.5):
            raise ValueError("Nodes need to be between -0.5 and 0.5.")
        elif setting.basis == "cos" and (np.min(X) < 0 or np.max(X) > 1):
            raise ValueError("Nodes need to be between 0 and 1.")
        elif setting.basis == "cheb" and (np.min(X) < -1 or np.max(X) > 1):
            raise ValueError("Nodes need to be between -1 and 1.")

        d = X.shape[1]
        M = X.shape[0]

        if (
            lam is not None
        ):  # evaluateSHAPterms( a::approx; X::Matrix{Float64}, λ::Float64 )::Union{Matrix{ComplexF64},Matrix{Float64}}
            terms = self.evaluateANOVAterms(X, lam)

            Dtype = np.complex128 if setting.basis == "per" else np.float64
            values = np.zeros((M, d), dtype=Dtype)

            for i in range(d):
                for j, u in enumerate(setting.U):
                    if i in u:
                        values[:, i] += terms[:, j] / len(u)
            return values

        else:  # evaluateSHAPterms( a::approx; X::Matrix{Float64} )::Dict{Float64,Union{Matrix{ComplexF64},Matrix{Float64}}}
            results = {}
            for l in self.lam.keys():
                terms = self.evaluateANOVAterms(X, l)

                Dtype = np.complex128 if setting.basis == "per" else np.float64
                values = np.zeros((M, d), dtype=Dtype)

                for i in range(d):
                    for j, u in enumerate(setting.U):
                        if i in u:
                            values[:, i] += terms[:, j] / len(u)
                results[l] = values

            return results

    _l2_error = ANOVAerrors._l2_error
    get_l2_error = ANOVAerrors.get_l2_error
    _mse = ANOVAerrors._mse
    get_mse = ANOVAerrors.get_mse
    _mad = ANOVAerrors._mad
    get_mad = ANOVAerrors.get_mad
    _L2_error = ANOVAerrors._L2_error
    get_L2_error = ANOVAerrors.get_L2_error
    _acc = ANOVAerrors._acc
    get_acc = ANOVAerrors.get_acc
    _auc = ANOVAerrors._auc
    get_auc = ANOVAerrors.get_auc

    _variances = ANOVAanalysis._variances
    get_variances = ANOVAanalysis.get_variances
    _GSI = ANOVAanalysis._GSI
    get_GSI = ANOVAanalysis.get_GSI
    _AttributeRanking = ANOVAanalysis._AttributeRanking
    get_AttributeRanking = ANOVAanalysis.get_AttributeRanking
    _ActiveSet = ANOVAanalysis._ActiveSet
    get_ActiveSet = ANOVAanalysis.get_ActiveSet
    _ShapleyValues = ANOVAanalysis._ShapleyValues
    get_ShapleyValues = ANOVAanalysis.get_ShapleyValues
