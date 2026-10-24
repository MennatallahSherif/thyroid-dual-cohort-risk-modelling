"""
Bridge feasibility check — Analysis Plan v1.1, Section 4 (prespecified). Decides D1.
Usage  : python step_bridge_feasibility.py <processed_dir> <results_dir>
Inputs : thyroidxl_clean_v2.csv, seer_clean_v2.csv (from step1v2_cleaning.py)
Outputs: bridge_strata.csv, bridge_slopes.csv, bridge_age_sex.csv, bridge_decision.txt

SEER population: DTC (PTC + FTC, corrected map) with survival time; unknown cause of death excluded (primary endpoint rule).
Not restricted to the nodal cohort: the bridge concerns age, not nodal metrics.
"""
import os, sys, warnings
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from lifelines import CoxPHFitter, AalenJohansenFitter
warnings.filterwarnings('ignore')

D = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else D
os.makedirs(OUT, exist_ok=True)
xl = pd.read_csv(os.path.join(D, 'thyroidxl_clean_v2.csv'))
se = pd.read_csv(os.path.join(D, 'seer_clean_v2.csv'), low_memory=False)
xl = xl[xl.in_setA == 1].copy()
se = se[se.surv_months.notna() & (se.cod_unknown == 0)].copy()
se['event_type'] = np.where(se.css_event == 1, 1, np.where(se.other_death == 1, 2, 0))
H = 24  # months

# ---------- strata (4.2): start from 6 groups; merge toward the middle until both rules hold
EDGES = [0, 35, 45, 55, 65, 75, 200]
def seer_lower(b):
    b = str(b)
    return 0 if b.startswith('00') else (90 if '90+' in b else int(b.split('-')[0]))
se['age_lower'] = se.age_bin.apply(seer_lower)       # stratify on the raw bin, not the midpoint
dtc = se[se.dtc == 1]

def counts(edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s_ = dtc[(dtc.age_lower >= lo) & (dtc.age_lower < hi)]
        x_ = xl[(xl.age >= lo) & (xl.age < hi)]
        out.append((lo, hi, int(s_.css_event.sum()), len(x_)))
    return out
edges = EDGES[:]
while True:
    c = counts(edges)
    bad = [i for i, (_, _, e, nx) in enumerate(c) if e < 10 or nx < 30]
    if not bad: break
    i = bad[0]
    mid = (len(c) - 1) / 2
    # merge toward the middle: a lower-half stratum merges with its upper neighbour, an upper-half one with its lower neighbour
    j = i + 1 if i < mid else i - 1
    drop = max(i, j)                 # remove the shared edge
    edges.pop(drop)
final = counts(edges)
def label(lo, hi): return f'<{hi}' if lo == 0 else (f'>={lo}' if hi == 200 else f'{lo}-{hi-1}')

# ---------- per-stratum estimates (4.3)
def aj_cif(d, horizon=H):
    if d.event_type.eq(1).sum() == 0: return 0.0, 0.0, np.nan
    aj = AalenJohansenFitter(calculate_variance=True, seed=1)
    aj.fit(d.surv_months.clip(lower=0.01), d.event_type, event_of_interest=1)
    cif = aj.cumulative_density_; ci = aj.confidence_interval_cumulative_density_
    idx = cif.index[cif.index <= horizon]
    k = idx[-1]
    return float(cif.loc[k].iloc[0]), float(ci.loc[k].iloc[0]), float(ci.loc[k].iloc[1])

rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    x_ = xl[(xl.age >= lo) & (xl.age < hi)]
    k = int(x_.malignant.sum()); n = len(x_)
    wl, wh = proportion_confint(k, n, method='wilson')
    r = dict(stratum=label(lo, hi), xl_n=n, xl_malignant=k, xl_pct=100 * k / n, xl_ci=f'{100*wl:.1f}-{100*wh:.1f}')
    for name, pop in [('dtc', dtc), ('ptc', se[se.histology == 'PTC']), ('all', se)]:
        s_ = pop[(pop.age_lower >= lo) & (pop.age_lower < hi)]
        p, l, h = aj_cif(s_)
        r.update({f'seer_{name}_n': len(s_), f'seer_{name}_css_events': int(s_.css_event.sum()),
                  f'seer_{name}_cif24_pct': 100 * p, f'seer_{name}_cif24_ci': f'{100*l:.2f}-{100*h:.2f}'})
    rows.append(r)
strata = pd.DataFrame(rows)

# ---------- within-cohort slopes per 10 years, sex-adjusted (4.4)
def xl_or(d, ycol):
    d = d[d[ycol].notna()]
    X = sm.add_constant(pd.DataFrame({'age10': d.age / 10, 'male': (d.gender == 1).astype(int)}))
    m = sm.Logit(d[ycol].astype(int), X).fit(disp=0)
    b, se_ = m.params['age10'], m.bse['age10']
    return np.exp(b), np.exp(b - 1.96 * se_), np.exp(b + 1.96 * se_), len(d), int(d[ycol].sum())
def seer_hr(d):
    df = pd.DataFrame({'T': d.surv_months.clip(lower=0.01), 'E': d.css_event, 'age10': d.age_mid / 10, 'male': d.male})
    cph = CoxPHFitter().fit(df, 'T', 'E')
    s = cph.summary.loc['age10']
    return s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%'], len(df), int(df.E.sum())
sl = []
for nm, f in [('ThyroidXL full cohort (primary)', lambda: xl_or(xl, 'malignant')),
              ('ThyroidXL S-D1 (sensitivity)', lambda: xl_or(xl[xl.sd1 == 1], 'sd1_malignant')),
              ('SEER DTC (primary)', lambda: seer_hr(dtc)),
              ('SEER PTC-only (sensitivity)', lambda: seer_hr(se[se.histology == 'PTC'])),
              ('SEER all histologies (reference)', lambda: seer_hr(se))]:
    est, lo, hi, n, e = f()
    sl.append(dict(analysis=nm, measure='OR per 10 y' if 'ThyroidXL' in nm else 'HR per 10 y', estimate=est, ci_low=lo, ci_high=hi, n=n, events=e))
slopes = pd.DataFrame(sl)

# ---------- age x sex (descriptive)
axs = []
for sex_xl, sex_se, lab in [(2, 0, 'Female'), (1, 1, 'Male')]:
    for cond, a_lab in [(lambda a: a < 55, '<55'), (lambda a: a >= 55, '>=55')]:
        x_ = xl[(xl.gender == sex_xl) & cond(xl.age)]
        k, n = int(x_.malignant.sum()), len(x_); wl, wh = proportion_confint(k, n, method='wilson')
        s_ = dtc[(dtc.male == sex_se) & cond(dtc.age_lower)]
        p, l, h = aj_cif(s_)
        axs.append(dict(group=f'{lab}, {a_lab}', xl_n=n, xl_pct=100 * k / n, xl_ci=f'{100*wl:.1f}-{100*wh:.1f}',
                        seer_dtc_n=len(s_), seer_dtc_css_events=int(s_.css_event.sum()), seer_dtc_cif24_pct=100 * p, seer_dtc_cif24_ci=f'{100*l:.2f}-{100*h:.2f}'))
age_sex = pd.DataFrame(axs)

# ---------- decision rule (4.5)
a_ok = all(e >= 10 and nx >= 30 for _, _, e, nx in final)
def sign(v): return 1 if v > 1 else (-1 if v < 1 else 0)
g = slopes.set_index('analysis')
b1_xl = sign(g.loc['ThyroidXL full cohort (primary)', 'estimate']) == sign(g.loc['ThyroidXL S-D1 (sensitivity)', 'estimate'])
b1_se = sign(g.loc['SEER DTC (primary)', 'estimate']) == sign(g.loc['SEER PTC-only (sensitivity)', 'estimate'])
def b2(row):
    # CI limit on the opposite side of 1 from the point estimate must lie within 0.91-1.10
    if row.estimate < 1: return row.ci_high <= 1.10
    return row.ci_low >= 1 / 1.10
b2_xl = b2(g.loc['ThyroidXL full cohort (primary)']); b2_se = b2(g.loc['SEER DTC (primary)'])
lines = [f'Final strata: {[label(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]}',
         f'(a) Estimability (>=10 SEER DTC CSS events and >=30 ThyroidXL patients per stratum): {a_ok}',
         f'(b1) Direction consistent ThyroidXL full vs S-D1: {b1_xl}; SEER DTC vs PTC-only: {b1_se}',
         f'(b2) No reverse effect >=10%/decade admitted: ThyroidXL {b2_xl}; SEER DTC {b2_se}',
         f'(b) Stability overall: {b1_xl and b1_se and b2_xl and b2_se}',
         '(c) Framing: author judgement (not computed)']
strata.to_csv(os.path.join(OUT, 'bridge_strata.csv'), index=False)
slopes.to_csv(os.path.join(OUT, 'bridge_slopes.csv'), index=False)
age_sex.to_csv(os.path.join(OUT, 'bridge_age_sex.csv'), index=False)
open(os.path.join(OUT, 'bridge_decision.txt'), 'w').write('\n'.join(lines))
pd.set_option('display.width', 250)
print(pd.DataFrame(final, columns=['lo', 'hi', 'seer_dtc_css_events', 'xl_n']))
print(strata.round(2).to_string(index=False)); print(slopes.round(3).to_string(index=False)); print(age_sex.round(2).to_string(index=False)); print('\n'.join(lines))
