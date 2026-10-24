"""
coxfast — minimal Cox proportional-hazards fitter used for the bootstrap analyses (Plan v1.1, Stage 2).
Efron ties, Newton-Raphson with step halving (the default full Newton step overflows on this design),
optional strata and ridge penalty (0.5 * lam * ||beta||^2 on standardised covariates, as in lifelines).
Baseline cumulative hazard: Breslow estimator (as in lifelines). Validated against lifelines in stage-2 log.
"""
import numpy as np

def _stratum_terms(time, event, X, beta):
    o = np.argsort(-time, kind='mergesort'); t, e, Xs = time[o], event[o], X[o]
    xb = Xs @ beta; w = np.exp(xb)
    ll = 0.0; p = X.shape[1]; g = np.zeros(p); H = np.zeros((p, p))
    ev_idx = np.where(e == 1)[0]
    if len(ev_idx) == 0: return ll, g, H
    ut = np.unique(t[ev_idx])[::-1]                                   # descending, matches sort order
    lasts = np.searchsorted(-t, -ut, side='right') - 1                  # last index with t >= tk
    Xw = Xs * w[:, None]
    cw = np.cumsum(w); cxw = np.cumsum(Xw, axis=0)
    r2 = np.zeros((p, p)); start = 0
    for tk, last in zip(ut, lasts):
        seg = slice(start, last + 1); r2 = r2 + Xw[seg].T @ Xs[seg]; start = last + 1
        r0, r1 = cw[last], cxw[last]
        D = ev_idx[t[ev_idx] == tk]; d = len(D)
        wd = w[D]; t0 = wd.sum(); t1 = Xw[D].sum(0); t2 = Xw[D].T @ Xs[D]
        ll += xb[D].sum(); g += Xs[D].sum(0)
        for l in range(d):
            c = l / d; den = r0 - c * t0; a1 = r1 - c * t1; a2 = r2 - c * t2
            ll -= np.log(den); g -= a1 / den; H -= a2 / den - np.outer(a1, a1) / den ** 2
    return ll, g, H

def _total(time, event, X, strata, beta, lam, scale):
    ll = 0.0; p = X.shape[1]; g = np.zeros(p); H = np.zeros((p, p))
    for s in np.unique(strata):
        m = strata == s; a, b, c = _stratum_terms(time[m], event[m], X[m], beta); ll += a; g += b; H += c
    if lam > 0:
        pen = lam / scale ** 2
        ll -= 0.5 * np.sum(pen * beta ** 2); g -= pen * beta; H -= np.diag(pen)
    return ll, g, H

class Cox:
    def __init__(self, lam=0.0, tol=1e-9, max_iter=100):
        self.lam, self.tol, self.max_iter = lam, tol, max_iter
    def fit(self, time, event, X, strata=None):
        time = np.asarray(time, float); event = np.asarray(event, int); X = np.asarray(X, float)
        strata = np.zeros(len(time), int) if strata is None else np.asarray(strata)
        self.mu = X.mean(0); Xc = X - self.mu; scale = X.std(0); scale[scale == 0] = 1
        beta = np.zeros(X.shape[1]); ll, g, H = _total(time, event, Xc, strata, beta, self.lam, scale)
        for _ in range(self.max_iter):
            step = np.linalg.solve(-H, g); s = 1.0
            while True:
                nb = beta + s * step; nll, ng, nH = _total(time, event, Xc, strata, nb, self.lam, scale)
                if np.isfinite(nll) and nll >= ll - 1e-12: break
                s /= 2
                if s < 1e-8: break
            conv = abs(nll - ll) < self.tol
            beta, ll, g, H = nb, nll, ng, nH
            if conv: break
        self.beta, self.loglik, self.var = beta, ll, np.linalg.inv(-H)
        # Breslow baseline per stratum on centred covariates
        self.base = {}
        for s in np.unique(strata):
            m = strata == s; t, e, w = time[m], event[m], np.exp(Xc[m] @ beta)
            ut = np.unique(t[e == 1]); H0 = []
            o = np.argsort(t); ts, ws = t[o], w[o]; rev = np.cumsum(ws[::-1])[::-1]
            for tk in ut:
                idx = np.searchsorted(ts, tk, side='left'); H0.append((e[t == tk] == 1).sum() / rev[idx])
            self.base[s] = (ut, np.cumsum(H0))
        return self
    def lp(self, X):
        return (np.asarray(X, float) - self.mu) @ self.beta
    def risk(self, X, horizon, strata=None):
        lp = self.lp(X); strata = np.zeros(len(lp), int) if strata is None else np.asarray(strata)
        out = np.empty(len(lp))
        for s in np.unique(strata):
            ut, H0 = self.base[s]; k = np.searchsorted(ut, horizon, side='right') - 1
            h = H0[k] if k >= 0 else 0.0; m = strata == s
            out[m] = 1 - np.exp(-h * np.exp(lp[m]))
        return out
