from pyGroupedTransforms.GroupedCoefficients import *  # TODO: Kann wahrscheinlich weg sobald in pyGroupedTransform GreoupedTransform exportiert wird

from pyANOVAapprox import *


def bisection(
    fun, fval, left, right, fleft, fright, max_iter=10, tol=1e-15, verbose=False
):

    fright -= fval
    fleft -= fval

    for it in range(max_iter):
        middle = (left + right) / 2
        fmiddle = fun(middle) - fval

        if (fmiddle >= 0 and fleft >= 0) or (fmiddle < 0 and fleft < 0):
            left = middle
            fleft = fmiddle
        else:
            right = middle
            fright = fmiddle

        if verbose:
            print(f"residual for Bisection {abs(fmiddle)}")

        if abs(fmiddle) < tol:
            break

    return middle


def newton(fun, dfun, fval, x, max_iter=10, tol=1e-15, verbose=False):
    for it in range(max_iter):
        f = fun(x)
        df = dfun(x)

        if isnan(f) or isnan(df) or abs(df) < 1e-15:
            break

        x += (fval - f) / df

        if verbose:
            print(
                f"residual for Newton: {abs(f - fval)}\n"
                f"f: {f} df: {df} fval: {fval} x {x}"
            )

        if abs(f - fval) < tol:
            break

    return x


def λ2ξ(lam, what, y, verbose=False):

    def fun(xi):
        return np.sum(np.abs(what * (y / ((1 / xi) + what)) ** 2))

    def dfun(xi):
        return (
            2
            * np.sum(np.abs(what * (y / ((1 / xi) + what)) ** 2 / ((1 / xi) + what)))
            * xi**-2
        )

    fright = np.sum(np.abs(what * (y / (1 + what)) ** 2))

    if lam**2 < fright:
        fleft = 0
        xi = bisection(
            fun,
            lam**2,
            1e-10,
            1.0,
            fleft,
            fright,
            max_iter=25,
            tol=1e-10,
            verbose=verbose,
        )
    else:
        fleft = np.sum(what * (y / what) ** 2)

        def inv_fun(xi):
            return fun(1 / xi)

        xi = 1 / bisection(
            inv_fun,
            lam**2,
            1e-10,
            1.0,
            fleft,
            fright,
            max_iter=25,
            tol=1e-16,
            verbose=verbose,
        )

        if xi > 100:
            return xi

    if xi <= 0 or np.isnan(xi) or np.isinf(xi):  # später wieder rausnehmen
        raise RuntimeError(f"λ2ξ: invalid xi before Newton: xi = {xi}")

    # apply Newton on f(exp(x)). Small solutions can be found more accurate this way and we work our way around negative solutions.
    xi = math.exp(
        newton(
            lambda x: fun(math.exp(x)),
            lambda x: dfun(math.exp(x)) * math.exp(x),
            lam**2,
            math.log(xi),
            max_iter=50,
            tol=1e-16,
            verbose=verbose,
        )
    )

    if abs(fun(xi) - lam**2) > 1:
        raise RuntimeError(
            f"λ2ξ: something went wrong minimizing. (residual: {abs(fun(xi) - lam **2)})"
        )

    return xi


def loss2_function(x):
    for i in range(len(x)):
        if x[i] > 1:
            x[i] = 0
        else:
            x[i] = (1 - x[i]) ** 2

    return x


def nabla_loss2_function(x):
    # derivative of quadratic loss:
    for i in range(len(x)):
        if x[i] > 1:
            x[i] = 0
        else:
            x[i] = (1 - x[i]) * -2

    return x


def fista(
    ghat,
    F,
    y,
    lam,
    what,
    L="adaptive",
    max_iter=25,
    classification=False,
    verbose=False,
):

    adaptive = L == "adaptive"

    if adaptive:
        L = 1.0
        eta = 2.0
    else:
        L = float(L)

    U = [s.u for s in ghat.settings]

    hhat = GroupedCoefficients(ghat.settings, np.copy(ghat.vec()))
    t = 1.0

    def loss_val(coef):
        if classification:
            return np.sum(loss2_function(y * (F @ coef))) * (1 / len(y)) + lam * np.sum(
                np.abs(coef.data)
            )
        else:
            return 0.5 * (np.linalg.norm((F @ coef) - y) ** 2) + lam * np.sum(
                coef.norms(Dict=False, other=what)
            )

    val = [loss_val(hhat)]

    for k in range(max_iter - 1):
        k = k + 1

        ghat_old = GroupedCoefficients(ghat.settings, np.copy(ghat.vec()))
        t_old = t

        if classification:
            fgrad = F.H @ (
                (y * nabla_loss2_function(y * (F @ hhat))) / len(y)
            )  # TODO: ghat or hhat
        else:
            fgrad = F.H @ (F @ hhat - y)

        while True:
            # p_L(hhat)
            if classification:
                for k in range(len(ghat.data)):
                    # fhat > 0:
                    if L * hhat[k] - fgrad[k] > lam:
                        ghat[k] = hhat[k] - fgrad[k] / L - lam / L
                        # fhat < 0:
                    elif fgrad[k] - L * hhat[k] > lam:
                        ghat[k] = hhat[k] - fgrad[k] / L + lam / L
                    else:
                        ghat[k] = 0.0
            else:
                ghat.set_data((hhat - (1 / L) * fgrad).vec())
                mask = [
                    (lam / L) ** 2 < np.sum(np.abs((ghat[u] ** 2) / what[u])) for u in U
                ]
                U_masked = [u for u, m in zip(U, mask) if m]

                # code mit threads:

                threads = []
                h = 0
                H = len(U_masked)
                xis = [None] * H

                def worker(i, u):
                    xis[i] = λ2ξ(lam / L, what[u], ghat[u], verbose=verbose)

                for u in U_masked:
                    T = threading.Thread(target=worker, args=(h, u))
                    T.start()
                    threads.append(T)
                    h = h + 1

                for T in threads:
                    T.join()

                if H == h + 1:
                    print("okay!")

                # alternativer code ohne threads:
                # xis = [λ2ξ(lam / L, what[u], ghat[u], verbose =verbose) for u in U_masked]

                for u, xi in zip(U_masked, xis):
                    if np.isinf(xi):
                        ghat[u] = 0 * ghat[u]
                    else:
                        ghat[u] = ghat[u] / (1 + xi * what[u])

            if not adaptive:
                val.append(loss_val(hhat))
                break

            # F
            Fvalue = loss_val(ghat)

            # Q
            if classification:
                Q = (
                    (1 / len(y)) * np.sum(loss2_function(y * (F * hhat)))
                    + np.vdot((ghat - hhat).vec(), fgrad.vec()).real
                    + L / 2 * np.linalg.norm((ghat - hhat).vec()) ** 2
                    + lam * np.sum(np.abs(ghat.vec()))
                )

            else:
                Q = (
                    np.linalg.norm((F * hhat) - y) ** 2 / 2
                    + np.vdot((ghat - hhat).vec(), fgrad.vec()).real
                    + L / 2 * np.linalg.norm((ghat - hhat).vec()) ** 2
                    + lam * np.sum(ghat.norms(Dict=False, other=what))
                )

            if Fvalue.real < Q + 1e-10 or L >= 2**32:
                val.append(Fvalue)
                break
            else:
                L *= eta

        # update t
        t = (1 + math.sqrt(1 + 4 * t**2)) / 2

        # update hhat
        hhat = ghat + (t_old - 1) / t * (ghat - ghat_old)

        # stoping criteria
        resnorm = np.linalg.norm((ghat_old - ghat).vec())
        if resnorm < 1e-16:
            break
        if abs(val[-1] - val[-2]) < 1e-16:
            break
