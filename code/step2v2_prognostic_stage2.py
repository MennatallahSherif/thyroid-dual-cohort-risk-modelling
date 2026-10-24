"""
STEP 2 v2 — SEER prognostic modelling, stage 2 (Analysis Plan v1.1, Sections 3.6-3.12; deviation-log entry for
the post hoc histology-stratified sensitivity).

Usage  : python step2v2_prognostic_stage2.py <processed_dir> <results_dir> <part>
         part = main   : primary + FTC-stratified models; Harrell bootstrap B=500 (optimism, paired ΔC, calibration)
                sens   : prespecified sensitivity analyses (apparent; paired ΔC with B=200 refit bootstrap)
                desc   : histology / special-code descriptive table
                dca    : decision-curve analysis (run only if the main part shows eligibility)
                report : assemble final tables from the parts
Engine : coxfast.Cox (Efron ties; validated against lifelines: identical HRs, 24-m risks within 1e-9)

Bootstrap conventions (Harrell 1996; Steyerberg 2019):
  optimism_b = metric(model_b on bootstrap sample b) - metric(model_b on original data)
  corrected  = apparent - mean(optimism_b)
  paired ΔC  : same resamples for all models; corrected Δ = apparent Δ - mean(Δ optimism);
               95% CI = percentile interval of the bootstrap-sample Δ, shifted by the mean Δ optimism;
               two-sided bootstrap p = 2*min(P(Δ*<=0), P(Δ*>=0)) on the shifted distribution (floor 1/B); Holm for the
               2 primary comparisons (M2-M1, M3-M1).
Uno's C uses IPCW from the evaluation data, truncated at tau = 24 months.
Calibration slope: Cox regression of the outcome (administratively censored at 24 m) on cloglog(predicted 24-m risk).
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from lifelines import KaplanMeierFitter, AalenJohansenFitter
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coxfast import Cox
warnings.filterwarnings('ignore')

IN, OUT, PART = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
SEED, B_MAIN, B_SENS, H = 20261024, 500, 200, 24
if len(sys.argv) > 4: B_MAIN = B_SENS = int(sys.argv[4])   # testing only

s = pd.read_csv(os.path.join(IN, 'seer_clean_v2.csv'), low_memory=False)

def cohort(dtc_only=True, unknown_cod='exclude', drop_ex90=False, min_D=None):
    d = s[(s.nodal_quant == 1) & s.surv_months.notna()]
    if dtc_only: d = d[d.dtc == 1]
    if unknown_cod == 'exclude': d = d[d.cod_unknown == 0]
    d = d[d[['age_mid', 'male', 'T', 'N', 'M']].notna().all(axis=1)]
    if drop_ex90: d = d[d.nodes_examined_90plus == 0]
    if min_D: d = d[d.D >= min_D]
    d = d.reset_index(drop=True).copy()
    d['time'] = d.surv_months.where(d.surv_months > 0, 0.5)
    d['event'] = d.css_event_sensA if unknown_cod == 'as_css' else d.css_event
    return d

def design(d, extra=None, all_hist=False, rcs=False):
    X = pd.DataFrame({'age': d.age_mid, 'male': d.male, 'FTC': d.ftc,
                      'T2': (d['T'] == 'T2') * 1, 'T3': (d['T'] == 'T3') * 1, 'T4': (d['T'] == 'T4') * 1, 'M1': (d.M == 'M1') * 1,
                      'N1a': (d.N == 'N1a') * 1, 'N1b': (d.N == 'N1b') * 1, 'N1_NOS': (d.N == 'N1 NOS') * 1,
                      'LNR': d.LNR, 'LODDS': d.LODDS})
    if rcs:  # restricted cubic spline, 3 knots at 10/50/90th percentiles (Harrell), one non-linear term
        k = np.percentile(d.age_mid, [10, 50, 90]); x = d.age_mid.values
        p3 = lambda u: np.maximum(u, 0) ** 3
        X['age_rcs1'] = (p3(x - k[0]) - p3(x - k[1]) * (k[2] - k[0]) / (k[2] - k[1]) + p3(x - k[2]) * (k[1] - k[0]) / (k[2] - k[1])) / (k[2] - k[0]) ** 2
    if all_hist:
        for h in ['ATC', 'MTC', 'Mixed medullary', 'Other']:
            X['H_' + h.replace(' ', '_')] = (d.histology == h) * 1
    if extra is not None:
        for c in extra.columns: X[c] = extra[c].values
    return X.astype(float)

CORE = ['age', 'male', 'FTC', 'T2', 'T3', 'T4', 'M1']
NOD = {'M0 core': [], 'M1 core + N': ['N1a', 'N1b', 'N1_NOS'], 'M2 core + LNR': ['LNR'], 'M3 core + LODDS': ['LODDS']}
def specs(core):
    return {k: core + v for k, v in NOD.items()}
PAIRS = [('M2 core + LNR', 'M1 core + N', 'primary'), ('M3 core + LODDS', 'M1 core + N', 'primary'),
         ('M1 core + N', 'M0 core', 'secondary'), ('M2 core + LNR', 'M0 core', 'secondary'), ('M3 core + LODDS', 'M0 core', 'secondary')]

def km_risk(t, e, h=H):
    km = KaplanMeierFitter().fit(t, e); return 1 - float(km.survival_function_at_times(h).iloc[0])

def cal_slope(t, e, risk):
    r = np.clip(risk, 1e-10, 1 - 1e-10); cll = np.log(-np.log(1 - r))[:, None]
    t24 = np.minimum(t, H); e24 = ((e == 1) & (t <= H)).astype(int)
    return Cox().fit(t24, e24, cll).beta[0]

def metrics(model, X, t, e, strata, full=True):
    lp = model.lp(X); y = Surv.from_arrays(e.astype(bool), t)
    out = {'uno24': concordance_index_ipcw(y, y, lp, tau=H)[0]}
    if full:
        out['harrell'] = concordance_index_censored(e.astype(bool), t, lp)[0]
        a, _ = cumulative_dynamic_auc(y, y, lp, [12, 24]); out['td12'], out['td24'] = a
        r = model.risk(X, H, strata); out['slope'] = cal_slope(t, e, r); out['OE'] = km_risk(t, e) / r.mean()
    return out

def fit_all(d, X, sp, strata, lam=None):
    st = d.ftc.values if strata else None
    return {k: Cox(lam=(lam or {}).get(k, 0.0)).fit(d.time.values, d.event.values, X[c].values, strata=st) for k, c in sp.items()}

# =====================================================================
if PART == 'main':
    d = cohort(); X = design(d); t, e = d.time.values, d.event.values
    assert len(d) == 30649 and e.sum() == 243
    VARIANTS = {'primary (unstratified)': (specs(CORE), False),
                'post hoc: stratified by histology': (specs([c for c in CORE if c != 'FTC']), True)}
    app = {}
    for vn, (sp, strat) in VARIANTS.items():
        st = d.ftc.values if strat else None
        ms = fit_all(d, X, sp, strat)
        app[vn] = {k: metrics(m, X[sp[k]].values, t, e, st) for k, m in ms.items()}
        # apparent calibration deciles
        rows = []
        for k, m in ms.items():
            r = m.risk(X[sp[k]].values, H, st); dec = pd.qcut(pd.Series(r).rank(method='first'), 10, labels=False).values
            for q in range(10):
                mm = dec == q; km = KaplanMeierFitter().fit(t[mm], e[mm]); ci = km.confidence_interval_survival_function_
                kk = ci.index[ci.index <= H][-1]
                rows.append(dict(variant=vn, model=k, decile=q + 1, n=int(mm.sum()), events_24m=int(((e == 1) & (t <= H))[mm].sum()),
                                 mean_predicted=r[mm].mean(), observed_KM=1 - km.survival_function_at_times(H).iloc[0],
                                 obs_low=1 - ci.loc[kk].iloc[1], obs_high=1 - ci.loc[kk].iloc[0]))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f'cal_deciles_{"strat" if strat else "primary"}.csv'), index=False)
    json.dump(app, open(os.path.join(OUT, 'main_apparent.json'), 'w'), indent=1)
    rng = np.random.default_rng(SEED); n = len(d)
    path = os.path.join(OUT, 'main_boot.csv')
    done = pd.read_csv(path).b.max() + 1 if os.path.exists(path) else 0
    for b in range(B_MAIN):
        idx = rng.integers(0, n, n)
        if b < done: continue
        db = d.iloc[idx].reset_index(drop=True); Xb = X.iloc[idx].reset_index(drop=True)
        rows = []
        for vn, (sp, strat) in VARIANTS.items():
            stb = db.ftc.values if strat else None; sto = d.ftc.values if strat else None
            ms = fit_all(db, Xb, sp, strat)
            for k, m in ms.items():
                mb = metrics(m, Xb[sp[k]].values, db.time.values, db.event.values, stb)
                mo = metrics(m, X[sp[k]].values, t, e, sto)
                rows.append(dict(b=b, variant=vn, model=k, **{f'boot_{q}': v for q, v in mb.items()}, **{f'orig_{q}': v for q, v in mo.items()}))
        pd.DataFrame(rows).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        if b % 10 == 0: print(f'main bootstrap {b + 1}/{B_MAIN}', time.strftime('%H:%M:%S'), flush=True)

# =====================================================================
if PART == 'sens':
    surg = s.surgery_code
    def treat(d):
        sc = d.surgery_code
        return pd.DataFrame({'surg_neartotal': sc.isin([30, 40]) * 1, 'surg_less': sc.isin([13, 20, 21, 22, 23, 25, 26, 27]) * 1,
                             'surg_other_unknown': sc.isin([0, 80, 90, 99]) * 1,
                             'RAI': d.radiation.isin(['Radioisotopes (1988+)', 'Combination of beam with implants or isotopes']) * 1})
    SENS = {
        'S-COD-A: unknown COD counted as CSS': dict(c=dict(unknown_cod='as_css')),
        'S-COD-B: unknown COD censored at death': dict(c=dict(unknown_cod='censor')),
        'S-EX90: examined = 90 excluded': dict(c=dict(drop_ex90=True)),
        'S-TREAT: + surgery type + RAI': dict(c={}, treat=True),
        'S-RCS: age as RCS (3 knots)': dict(c={}, rcs=True),
        'S-D5: nodes examined >= 5': dict(c=dict(min_D=5)),
        'S-ALLHIST: all histologies (reference)': dict(c=dict(dtc_only=False), allh=True),
        'S-RIDGE: ridge, lambda by 5-fold CV': dict(c={}, ridge=True),
    }
    LAMS = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    def cv_lambda(d, X, cols):
        from sklearn.model_selection import StratifiedKFold
        best, res = None, {}
        for lam in LAMS:   # cross-validated partial likelihood (Verweij & van Houwelingen 1993)
            cvl = 0.0
            for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, d.event):
                m_tr = Cox(lam=lam).fit(d.time.values[tr], d.event.values[tr], X[cols].values[tr])
                from coxfast import _total
                Xc = X[cols].values - m_tr.mu; sc = np.ones(len(cols))
                ll_all = _total(d.time.values, d.event.values, Xc, np.zeros(len(d)), m_tr.beta, 0, sc)[0]
                ll_tr = _total(d.time.values[tr], d.event.values[tr], Xc[tr], np.zeros(len(tr)), m_tr.beta, 0, sc)[0]
                cvl += ll_all - ll_tr
            res[lam] = cvl
        return max(res, key=res.get), res
    path = os.path.join(OUT, 'sens_results.csv'); out = []; lam_log = {}
    for name, cfg in SENS.items():
        d = cohort(**cfg['c'])
        extra = treat(d) if cfg.get('treat') else None
        X = design(d, extra=extra, all_hist=cfg.get('allh', False), rcs=cfg.get('rcs', False))
        core = CORE + (['age_rcs1'] if cfg.get('rcs') else []) + (list(extra.columns) if extra is not None else []) + \
               ([c for c in X.columns if c.startswith('H_')] if cfg.get('allh') else [])
        sp = specs(core); lam = None
        if cfg.get('ridge'):
            lam = {}
            for k, c in sp.items():
                lam[k], res = cv_lambda(d, X, c); lam_log[k] = {'selected': lam[k], 'cv_partial_loglik': res}
        t, e = d.time.values, d.event.values
        ms = fit_all(d, X, sp, False, lam)
        appm = {k: metrics(m, X[sp[k]].values, t, e, None) for k, m in ms.items()}
        rng = np.random.default_rng(SEED); bs = {k: [] for k in sp}
        for b in range(B_SENS):
            idx = rng.integers(0, len(d), len(d)); db = d.iloc[idx].reset_index(drop=True); Xb = X.iloc[idx].reset_index(drop=True)
            mb = fit_all(db, Xb, sp, False, lam)
            for k, m in mb.items(): bs[k].append(metrics(m, Xb[sp[k]].values, db.time.values, db.event.values, None, full=False)['uno24'])
        for k in sp:
            out.append(dict(analysis=name, kind='model', model=k, n=len(d), events=int(e.sum()), lam=(lam or {}).get(k, 0),
                            Uno_C_24=appm[k]['uno24'], Harrell_C=appm[k]['harrell'], tdAUC_24=appm[k]['td24'],
                            cal_slope=appm[k]['slope'], OE=appm[k]['OE']))
        for a, b_, role in PAIRS:
            dlt = np.array(bs[a]) - np.array(bs[b_])
            out.append(dict(analysis=name, kind='paired', model=f'{a} - {b_}', role=role, delta=appm[a]['uno24'] - appm[b_]['uno24'],
                            CI_low=np.percentile(dlt, 2.5), CI_high=np.percentile(dlt, 97.5)))
        if name.startswith('S-TREAT'):
            m1 = ms['M1 core + N']; se = np.sqrt(np.diag(m1.var))
            for j, c in enumerate(sp['M1 core + N']):
                out.append(dict(analysis=name, kind='HR in M1', model=c, delta=np.exp(m1.beta[j]),
                                CI_low=np.exp(m1.beta[j] - 1.96 * se[j]), CI_high=np.exp(m1.beta[j] + 1.96 * se[j])))
        pd.DataFrame(out).to_csv(path, index=False); print('done', name, time.strftime('%H:%M:%S'), flush=True)
    json.dump(lam_log, open(os.path.join(OUT, 'ridge_cv.json'), 'w'), indent=1, default=str)

# =====================================================================
if PART == 'desc':
    a = s[s.surv_months.notna() & (s.cod_unknown == 0)].copy(); a['time'] = a.surv_months.where(a.surv_months > 0, 0.5)
    rows = []
    def add(label, g, why=''):
        r = dict(group=label, n=len(g), css_events=int(g.css_event.sum()), css_pct=100 * g.css_event.mean(),
                 km_css_risk_24m_pct=100 * km_risk(g.time, g.css_event) if len(g) else np.nan, note=why)
        rows.append(r)
    for h in ['PTC', 'FTC', 'MTC', 'ATC', 'Mixed medullary', 'Other']:
        add(f'All records: {h}', a[a.histology == h])
    for h in ['PTC', 'FTC', 'MTC', 'ATC', 'Mixed medullary', 'Other']:
        add(f'Quantitative nodal cohort: {h}', a[(a.histology == h) & (a.nodal_quant == 1)])
    sp_ = a[a.nodes_special_code == 1]
    add('Special nodal codes 95-97 (all histologies)', sp_, 'Examined 95/96/97 or positive 95/97: aspiration only or number unknown; LNR/LODDS not computable')
    add('Special nodal codes 95-97 (DTC)', sp_[sp_.dtc == 1], 'Excluded from the primary cohort')
    comp = sp_.histology.value_counts().rename('n').to_frame(); comp['css_events'] = sp_.groupby('histology').css_event.sum()
    comp['pct_of_group'] = 100 * comp.n / comp.n.sum()
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'desc_histology.csv'), index=False)
    comp.reset_index().to_csv(os.path.join(OUT, 'desc_special_code_composition.csv'), index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False)); print(comp)

# =====================================================================
if PART == 'dca':   # 10-fold cross-validated 24-m risks (internally validated); exploratory
    from sklearn.model_selection import StratifiedKFold
    d = cohort(); X = design(d); t, e = d.time.values, d.event.values; sp = specs(CORE)
    cvr = {k: np.zeros(len(d)) for k in sp}
    for tr, va in StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, e):
        for k, c in sp.items():
            m = Cox().fit(t[tr], e[tr], X[c].values[tr]); cvr[k][va] = m.risk(X[c].values[va], H)
    ths = np.round(np.arange(0.0025, 0.1001, 0.0025), 4); rows = []
    prev = km_risk(t, e)
    for th in ths:
        nb_all = prev - (1 - prev) * th / (1 - th)
        for k in sp:
            pos = cvr[k] >= th; npos = pos.mean()
            nb = (km_risk(t[pos], e[pos]) * npos - (1 - km_risk(t[pos], e[pos])) * npos * th / (1 - th)) if pos.sum() > 0 else 0.0
            rows.append(dict(model=k, threshold=th, pct_positive=100 * npos, net_benefit=nb, nb_treat_all=nb_all))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'dca_24m_cv.csv'), index=False)
    pd.DataFrame({'time': t, 'event': e, **cvr}).to_csv(os.path.join(OUT, 'cv_risks_24m.csv'), index=False)
    print('dca done')
