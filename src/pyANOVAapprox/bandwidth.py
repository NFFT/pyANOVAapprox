from pyANOVAapprox import *


def getfcu(ghat, u):
    idx = [s.u for s in ghat.settings].index(u)
    bws = ghat.settings[idx].bandwidths

    #fcu = ghat[u].reshape(bws[::-1] - 1)
    #fcu = np.permute_dims(fcu, range(len(bws))[::-1])
    fcu = ghat[u].reshape(bws - 1)
    fcu = np.permute_dims(fcu, range(len(bws)))
    
    return fcu


def getaxissum(ghat, u, j, system):
    fcu = getfcu(ghat, u)
    idx = [s.u for s in ghat.settings].index(u)
    bws = ghat.settings[idx].bandwidths

    fcuj = np.sum(abs(fcu) ** 2, axis=tuple([i for i in range(len(u)) if i != j]))

    if system == "exp":
        fcuj = (
            fcuj[np.mod(range(int(bws[j] / 2), bws[j]), bws[j] - 1)]
            + fcuj[range(int(bws[j] / 2) - 1, -1, -1)]
        )
    elif system == "cos":
        pass
    else:
        raise ValueError("For this basis is estimate rates not implemented")
    # fcuj = np.concatenate(fcuj[math.ceil(bws[j]/2):] + [fcuj[0]]) + fcuj[math.ceil(bws[j]/2)-1::-1]
    return fcuj


def compute_bandwidth(B, D, t):
    us = set(D.keys()) - {()}
    bw = {u: [6] * len(u) for u in us}
    bw[()] = []
    
    minfreqs = sum(math.prod((bw[u][j] - 1) for j in range(len(u))) for u in us)
    if B < minfreqs:
        raise ValueError(f"Budget too small: {B} < {minfreqs}")

    def fun_lmbda_u(lmbda, u):
        p1 = math.prod((bw[u][j] - 1) for j in range(len(u)) if math.isnan(t[u][j]))
        p2 = math.prod(
            ((2 * t[u][j] * D[u][j]) / lmbda) ** (0.5 / t[u][j])
            for j in range(len(u))
            if not math.isnan(t[u][j])
        )
        denom_exp = 1 + sum(
            0.5 / t[u][j] for j in range(len(u)) if not math.isnan(t[u][j])
        )
        return (p1 * p2) ** (1 / denom_exp)

    def fun_lmbda(lmbda, B):
        total = 0.0
        for u in us:
            total += fun_lmbda_u(math.exp(lmbda), u)
        return total - B

    lmbda = math.exp(bisect(lambda t: fun_lmbda(t, B), -100, 100))

    for u in us:
        sizeIu = fun_lmbda_u(lmbda, u)
        for j in range(len(u)):
            if not math.isnan(t[u][j]):
                bwuj = ((2 * t[u][j] * D[u][j]) / (lmbda * sizeIu)) ** (
                    0.5 / t[u][j]
                ) + 1
                bw[u][j] = 2 * round(0.5 * bwuj)

    return bw


def fitrate_log(y):
    n = len(y)
    lx = [math.log(i) for i in range(1, n + 1)]
    ly = [math.log(val) for val in y]
    omega = [1.0 / i for i in range(1, n + 1)]

    wslx = sum(o * x for o, x in zip(omega, lx))
    wsly = sum(o * y for o, y in zip(omega, ly))
    wslxly = sum(o * x * y for o, x, y in zip(omega, lx, ly))
    ws2lx = sum(o * (x**2) for o, x in zip(omega, lx))
    so = sum(omega)
    denom = so * ws2lx - (wslx**2)

    a = (ws2lx * wsly - wslxly * wslx) / denom
    b = (so * wslxly - wslx * wsly) / denom

    return math.exp(a), b


def most_common_value(data):
    mn, mx = min(data), max(data)
    n_points = 1000
    t = [mn + (mx - mn) * i / (n_points - 1) for i in range(n_points)]

    N = len(data)
    cdf = [sum(1 for d in data if d < x) / N for x in t]

    df = [0.0] * n_points
    df[0] = (cdf[1] - cdf[0]) / (t[1] - t[0])
    df[-1] = (cdf[-1] - cdf[-2]) / (t[-1] - t[-2])
    for i in range(1, n_points - 1):
        df[i] = (cdf[i + 1] - cdf[i - 1]) / (t[i + 1] - t[i - 1])

    max_index = max(range(n_points), key=lambda i: df[i])
    return t[max_index]


def estimate_rates(self, lam, settingnr=None, verbosity=0):
    us = [s.u for s in self.getTrafo(settingnr).settings]
    nhat = GroupedCoefficients(
        self.getTrafo(settingnr).settings,
        [1.0 + 0.0j] * len(self.getFc(settingnr)[lam].data),
    )

    D = dict([(u, [None] * len(u)) for u in us])
    t = dict([(u, [None] * len(u)) for u in us])

    mcl = np.exp(most_common_value(np.log(abs(self.getFc(settingnr)[lam].data))))
    threshold = 100 * mcl**2
    
    if verbosity>5:
        if not os.path.exists(os.path.join("log","figures")):
            os.mkdir(os.path.join("log","figures"))
        num = 0
        
    system = self.getTrafo(settingnr).system

    for u in us:
        if len(u) == 0:
            continue

        if verbosity>5:
            fig, ax = plt.subplots()
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(str(u))

        for j in range(len(u)):
            axissum = getaxissum(self.getFc(settingnr)[lam], u, j, system)

            axissumnum = getaxissum(nhat, u, j, system)
            idx = next(
                (
                    i
                    for i, v in enumerate((axissum > threshold * axissumnum)[::-1])
                    if v
                ),
                None,
            )

            if verbosity>5:

                y = np.cumsum(axissum[::-1])[::-1]
                ax.plot(
                    range(1, len(y) + 1),
                    y,
                    label=u[j],
                    marker="o",
                    #                    color=j
                )

            if (idx is None) or idx >= len(axissum):
                D[u][j] = math.nan
                t[u][j] = math.nan
            else:
                idx = min(len(axissum), len(axissum) - idx + 2)
                Duj, tuj = fitrate_log(np.cumsum((axissum[0:idx])[::-1])[::-1])
                D[u][j] = Duj
                t[u][j] = -tuj / 2

                if verbosity>5:
                    x = np.arange(1, idx + 1)
                    ax.plot(
                        x,
                        D[u][j] * x ** (-2 * t[u][j]),
                        linewidth=2,
                        #                        color=j
                    )
        if verbosity>5:
            #plt.figure(figsize=(12, 9))
            #for ax in ps:
            #    plt.sca(ax)
            fig.savefig(os.path.join("log","figures", str(num).strip()+ "_rates_" + str(u).strip() + ".png"))
            num = num + 1
            plt.close(fig) 

    return D, t
