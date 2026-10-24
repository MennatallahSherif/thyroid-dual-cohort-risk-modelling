"""
STEP 2 v2 — SEER prognostic modelling, stage 1 (Analysis Plan v1.1, Sections 3.5-3.9). Apparent performance only;
optimism correction, paired ΔC and sensitivity analyses are in stage 2.

Usage  : python step2v2_prognostic_stage1.py <processed_dir> <results_dir>
Input  : seer_clean_v2.csv (from step1v2_cleaning.py)
Cohort : DTC (corrected map), quantitative nodal data, survival time available, known cause of death,
         complete age, sex, T, N, M  (primary analytic cohort; expected 30,649 / 243 CSS events)
Models : M0 core = age (continuous, per year) + male + FTC + T2/T3/T4 (ref T1) + M1
         M1 = M0 + N1a/N1b/N1 NOS (ref N0);  M2 = M0 + LNR;  M3 = M0 + LODDS
         Unpenalised Cox (EPP 24.3 >= 20, Plan 3.6)
Time   : survival months; 0 months set to 0.5 (deaths/censoring within the diagnosis month)
Outputs: prog_hr.csv, prog_discrimination.csv, prog_calibration_summary.csv, prog_calibration_deciles.csv,
         prog_ph_tests.csv, prog_calibration_24m.png, prog_stage1_log.txt
"""
import os, sys, warnings
import numpy as np, pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

IN = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'
os.makedirs(OUT, exist_ok=True)
LOG = []
def log(m=''):
    print(m); LOG.append(str(m))

s = pd.read_csv(os.path.join(IN, 'seer_clean_v2.csv'), low_memory=False)
d = s[(s.dtc == 1) & (s.nodal_quant == 1) & s.surv_months.notna() & (s.cod_unknown == 0)]
d = d[d[['age_mid', 'male', 'T', 'N', 'M']].notna().all(axis=1)].reset_index(drop=True)
log(f'Primary cohort: n = {len(d):,}; CSS events = {int(d.css_event.sum())}')
assert len(d) == 30649 and d.css_event.sum() == 243

d['time'] = d.surv_months.where(d.surv_months > 0, 0.5)
d['event'] = d.css_event
X = pd.DataFrame({'age': d.age_mid, 'male': d.male, 'FTC': d.ftc,
                  'T2': (d['T'] == 'T2').astype(int), 'T3': (d['T'] == 'T3').astype(int), 'T4': (d['T'] == 'T4').astype(int),
                  'M1': (d.M == 'M1').astype(int),
                  'N1a': (d.N == 'N1a').astype(int), 'N1b': (d.N == 'N1b').astype(int), 'N1_NOS': (d.N == 'N1 NOS').astype(int),
                  'LNR': d.LNR, 'LODDS': d.LODDS})
CORE = ['age', 'male', 'FTC', 'T2', 'T3', 'T4', 'M1']
SPEC = {'M0 core': CORE, 'M1 core + N': CORE + ['N1a', 'N1b', 'N1_NOS'],
        'M2 core + LNR': CORE + ['LNR'], 'M3 core + LODDS': CORE + ['LODDS']}
for k, v in SPEC.items():
    log(f'{k}: {len(v)} parameters; EPP = {d.event.sum() / len(v):.1f}')

y_all = Surv.from_arrays(d.event.astype(bool), d.time)
TAUS = [12, 24, 36]
# Newton-Raphson step 0.3: the default full step overflows on this design; the step size changes the path, not the MLE
FIT = dict(step_size=0.3)

HR, DISC, CALS, CALD, PH = [], [], [], [], []
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)

def km_risk(t, e, h=24):
    km = KaplanMeierFitter().fit(t, e)
    sv = km.survival_function_at_times(h).iloc[0]
    ci = km.confidence_interval_survival_function_
    k = ci.index[ci.index <= h][-1]
    return 1 - sv, 1 - ci.loc[k].iloc[1], 1 - ci.loc[k].iloc[0]

for ax, (name, cols) in zip(axes, SPEC.items()):
    df = pd.concat([X[cols], d[['time', 'event']]], axis=1)
    cph = CoxPHFitter(penalizer=0.0).fit(df, 'time', 'event', fit_options=FIT)
    sm_ = cph.summary
    for c in cols:
        HR.append(dict(model=name, predictor=c, HR=sm_.loc[c, 'exp(coef)'], CI_low=sm_.loc[c, 'exp(coef) lower 95%'],
                       CI_high=sm_.loc[c, 'exp(coef) upper 95%'], p=sm_.loc[c, 'p']))
    lp = cph.predict_log_partial_hazard(df[cols]).values.ravel()

    # ---- discrimination (apparent)
    harrell = concordance_index_censored(d.event.astype(bool).values, d.time.values, lp)[0]
    uno24 = concordance_index_ipcw(y_all, y_all, lp, tau=24)[0]
    uno36 = concordance_index_ipcw(y_all, y_all, lp, tau=36)[0]
    tmax = d.time[d.event == 1].max()
    taus = [t for t in TAUS if t < d.time.max()]
    auc_t, _ = cumulative_dynamic_auc(y_all, y_all, lp, taus)
    r = dict(model=name, n=len(d), events=int(d.event.sum()), parameters=len(cols),
             Harrell_C=harrell, Uno_C_tau24=uno24, Uno_C_tau36_exploratory=uno36)
    for t, a in zip(taus, auc_t): r[f'tdAUC_{t}m' + ('_exploratory' if t == 36 else '')] = a
    DISC.append(r)

    # ---- calibration at 24 months
    S24 = cph.predict_survival_function(df[cols], times=[24]).values.ravel()
    pred = np.clip(1 - S24, 1e-8, 1 - 1e-8)
    obs, olo, ohi = km_risk(d.time, d.event)
    # calibration slope: Cox regression of the outcome (administratively censored at 24 m) on cloglog(predicted risk)
    t24 = np.minimum(d.time, 24); e24 = ((d.event == 1) & (d.time <= 24)).astype(int)
    cl = pd.DataFrame({'cll': np.log(-np.log(1 - pred)), 't': t24, 'e': e24})
    cs = CoxPHFitter().fit(cl, 't', 'e', fit_options=FIT).summary.loc['cll']
    CALS.append(dict(model=name, mean_predicted_24m=pred.mean(), observed_KM_24m=obs, OE_ratio=obs / pred.mean(),
                     max_predicted_24m=pred.max(), calibration_slope=cs['coef'],
                     slope_CI=f"{cs['coef lower 95%']:.3f}-{cs['coef upper 95%']:.3f}"))
    dec = pd.qcut(pd.Series(pred).rank(method='first'), 10, labels=False)
    xs, ys, lo_, hi_ = [], [], [], []
    for q in range(10):
        m = (dec == q).values
        o, l, h = km_risk(d.time[m], d.event[m])
        CALD.append(dict(model=name, decile=q + 1, n=int(m.sum()), css_events_24m=int(e24[m].sum()),
                         mean_predicted_24m=pred[m].mean(), observed_KM_24m=o, obs_CI_low=l, obs_CI_high=h))
        xs.append(pred[m].mean()); ys.append(o); lo_.append(l); hi_.append(h)
    xs, ys = np.array(xs) * 100, np.array(ys) * 100
    ax.errorbar(xs, ys, yerr=[ys - np.array(lo_) * 100, np.array(hi_) * 100 - ys], fmt='o', ms=4, capsize=2)
    lim = max(xs.max(), (np.array(hi_) * 100).max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8); ax.set_title(f'{name}\nslope {cs["coef"]:.2f}, O:E {obs / pred.mean():.2f}', fontsize=10)
    ax.set_xlabel('Mean predicted 24-month CSS risk (%)')
    if ax is axes[0]: ax.set_ylabel('Observed 24-month CSS risk, KM (%)')

    # ---- proportional hazards
    pht = proportional_hazard_test(cph, df, time_transform='rank')
    for c in cols:
        PH.append(dict(model=name, predictor=c, test_statistic=pht.summary.loc[c, 'test_statistic'], p=pht.summary.loc[c, 'p']))
    log(f'{name}: Harrell {harrell:.4f}  Uno24 {uno24:.4f}  slope {cs["coef"]:.3f}  O:E {obs / pred.mean():.3f}')

plt.tight_layout(); plt.savefig(os.path.join(OUT, 'prog_calibration_24m.png'), dpi=200); plt.close()
for nm, rows in [('prog_hr', HR), ('prog_discrimination', DISC), ('prog_calibration_summary', CALS),
                 ('prog_calibration_deciles', CALD), ('prog_ph_tests', PH)]:
    pd.DataFrame(rows).to_csv(os.path.join(OUT, nm + '.csv'), index=False)
pd.set_option('display.width', 250)
for nm in ['prog_discrimination', 'prog_calibration_summary', 'prog_hr', 'prog_ph_tests']:
    log(f'\n== {nm}\n' + pd.read_csv(os.path.join(OUT, nm + '.csv')).round(4).to_string(index=False))
open(os.path.join(OUT, 'prog_stage1_log.txt'), 'w', encoding='utf-8').write('\n'.join(LOG))
