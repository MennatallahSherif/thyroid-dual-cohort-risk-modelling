"""
verify_outputs.py — v1.1 (frozen results of 24 Oct 2026; Analysis Plan v1.1, Stage-2 Report Section 12).

Usage (from the code folder):
    python verify_outputs.py [--data ..\\data\\processed_ref2024] [--results ..\\results] [--out ..\\results\\verification_report.csv]

Output format kept from v1.0.1: verification_report.csv with columns check, expected, got, tol, result,
plus a sixth column `group` (primary | sensitivity | descriptive/audit).
Every check is required. A missing output file or column is a FAIL, never a skip.
The script exits with status 1 if any check fails.

Check types
  eq   : |got - expected| <= tol (tol = 0 means exact)
  le   : got <= expected
  ge   : got >= expected
  true : boolean condition (expected 1, got 1/0)
"""
import argparse, json, os, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join('..', 'data', 'processed_ref2024'))
ap.add_argument('--results', default=os.path.join('..', 'results'))
ap.add_argument('--out', default=None)
a = ap.parse_args()
R = a.results
OUT = a.out or os.path.join(R, 'verification_report.csv')
P = {'xl': os.path.join(a.data, 'thyroidxl_clean_v2.csv'), 'seer': os.path.join(a.data, 'seer_clean_v2.csv'),
     'diag': {y: os.path.join(R, f'diagnostic_ref{y}') for y in ['2024', '2023', '2025', 'exclude']},
     'stage2': os.path.join(R, 'stage2'), 'prog': os.path.join(R, 'prognostic'), 'bridge': os.path.join(R, 'bridge_ref2024')}

ROWS = []
def add(check, expected, got, tol=0.0, kind='eq', group='primary'):
    try:
        g = float(got)
        if kind == 'eq': ok = abs(g - float(expected)) <= tol + 1e-12
        elif kind == 'le': ok = g <= float(expected) + 1e-12
        elif kind == 'ge': ok = g >= float(expected) - 1e-12
        elif kind == 'true': ok = bool(g) is bool(expected)
        else: raise ValueError(kind)
    except Exception:
        g, ok = np.nan, False
    ROWS.append(dict(check=check, expected=float(expected), got=g, tol=tol, result='PASS' if ok else 'FAIL', group=group))

def fail(check, group, why):
    ROWS.append(dict(check=f'{check} [{why}]', expected=np.nan, got=np.nan, tol=np.nan, result='FAIL', group=group))

def load_csv(path, check, group):
    if not os.path.exists(path):
        fail(check, group, f'missing file {path}'); return None
    return pd.read_csv(path, low_memory=False)

def section(name, group, fn):
    try:
        fn()
    except Exception as e:  # any missing column / row is a FAIL, not a skip
        fail(name, group, f'{type(e).__name__}: {e}')

# ======================================================================
# ThyroidXL — cohort and labels (descriptive/audit)
def xl_checks():
    x = load_csv(P['xl'], 'ThyroidXL cleaned file', 'descriptive/audit')
    if x is None: return
    A = x[x.in_setA == 1]
    g = 'descriptive/audit'
    add('XL Set A n', 4088, len(A), group=g)
    add('XL primary malignant (histopathology over FNAC)', 1231, A.malignant.sum(), group=g)
    add('XL released-label malignant', 1229, A.malignant_released.sum(), group=g)
    ch = set(A.loc[A.label_changed == 1, 'patient_id'].astype(int))
    add('XL labels changed by hierarchy = {256, 366, 370, 509}', 1, int(ch == {256, 366, 370, 509}), kind='true', group=g)
    S = A[A.sd1 == 1]
    add('XL S-D1 n', 796, len(S), group=g)
    add('XL S-D1 malignant', 704, S.sd1_malignant.sum(), group=g)
    add('XL S-D1 benign', 92, (S.sd1_malignant == 0).sum(), group=g)
    add('XL Set B within S-D1 benign', 88, ((S.in_setB == 1) & (S.sd1_malignant == 0)).sum(), group=g)
    add('XL birth-year-coded ages', 34, x.age_birthyear_corrected.sum(), group=g)
    by = x[x.age_birthyear_corrected == 1]
    add('XL A4 reference year 2024 applied (age = 2024 - birth year)', 1,
        int(np.allclose(by.age, 2024 - by.age_raw)), kind='true', group=g)
section('ThyroidXL cohort', 'descriptive/audit', xl_checks)

# ThyroidXL — diagnostic performance (primary, 2024)
def diag_checks():
    d = P['diag']['2024']
    perf = load_csv(os.path.join(d, 'diag_performance.csv'), 'diag_performance 2024', 'primary')
    dl = load_csv(os.path.join(d, 'diag_paired_delta.csv'), 'diag_paired_delta 2024', 'primary')
    th = load_csv(os.path.join(d, 'diag_thresholds.csv'), 'diag_thresholds 2024', 'primary')
    if perf is None or dl is None or th is None: return
    p = perf[perf.analysis.str.startswith('P:')].set_index('model').AUC
    for m, e in [('TIRADS (empirical, fold-wise)', 0.9098), ('Logistic Regression', 0.9324), ('Random Forest', 0.9318), ('XGBoost', 0.9353)]:
        add(f'Diagnostic AUC (primary): {m}', e, p[m], 0.0005)
    q = dl[dl.analysis.str.startswith('P:')].set_index('comparison').delta_AUC
    for m, e in [('Logistic Regression', 0.0226), ('Random Forest', 0.0221), ('XGBoost', 0.0255)]:
        add(f'Diagnostic paired dAUC vs TIRADS: {m}', e, q[f'{m} - TIRADS'], 0.0005)
    spec = th.set_index('rule').Specificity.str.split(' ').str[0].astype(float)
    tir = spec[spec.index.str.startswith('TIRADS')].iloc[0]
    lr = spec[spec.index.str.startswith('Logistic')].iloc[0]
    add('Operating point: TIRADS >= 4 specificity', 0.576, tir, 0.002)
    add('Operating point: LR specificity at matched sensitivity', 0.574, lr, 0.002)
    add('Operating point: no ML model exceeds TIRADS specificity (max ML - TIRADS <= 0)', 0,
        spec[~spec.index.str.startswith('TIRADS')].max() - tir, kind='le')
    sb = perf[perf.analysis.str.startswith('Set B within S-D1')]
    add('Set B within S-D1 reported without CI (descriptive only)', 1, int(len(sb) == 4 and sb.AUC_CI.isna().all()), kind='true',
        group='sensitivity')
section('Diagnostic primary', 'primary', diag_checks)

def a4_checks():
    base = pd.read_csv(os.path.join(P['diag']['2024'], 'diag_performance.csv'))
    b = base[base.analysis.str.startswith('P:')].set_index('model').AUC
    for y in ['2023', '2025', 'exclude']:
        perf = load_csv(os.path.join(P['diag'][y], 'diag_performance.csv'), f'diag_performance {y}', 'sensitivity')
        th = load_csv(os.path.join(P['diag'][y], 'diag_thresholds.csv'), f'diag_thresholds {y}', 'sensitivity')
        if perf is None or th is None: continue
        p = perf[perf.analysis.str.startswith('P:')].set_index('model').AUC
        add(f'A4 sensitivity {y}: max |AUC - AUC(2024)|', 0, (p - b).abs().max(), 0.002, group='sensitivity')
        spec = th.set_index('rule').Specificity.str.split(' ').str[0].astype(float)
        tir = spec[spec.index.str.startswith('TIRADS')].iloc[0]
        add(f'A4 sensitivity {y}: no ML model exceeds TIRADS specificity', 0, spec[~spec.index.str.startswith('TIRADS')].max() - tir,
            kind='le', group='sensitivity')
section('A4 sensitivities', 'sensitivity', a4_checks)

# ======================================================================
# SEER — cohort construction (descriptive/audit)
def seer_checks():
    s = load_csv(P['seer'], 'SEER cleaned file', 'descriptive/audit')
    if s is None: return
    g = 'descriptive/audit'
    dtc = s[s.dtc == 1]
    add('SEER DTC n', 58781, len(dtc), group=g)
    add('SEER DTC CSS events', 564, dtc.css_event.sum(), group=g)
    sp = s[s.nodes_special_code == 1]
    add('SEER special nodal codes, all histologies n', 840, len(sp), group=g)
    add('SEER special nodal codes, all histologies CSS events', 145, sp.css_event.sum(), group=g)
    add('SEER special nodal codes, DTC n', 675, (sp.dtc == 1).sum(), group=g)
    add('SEER special nodal codes, DTC CSS events', 56, sp[sp.dtc == 1].css_event.sum(), group=g)
    add('SEER quantitative nodal DTC n', 31490, ((s.dtc == 1) & (s.nodal_quant == 1)).sum(), group=g)
    d = s[(s.dtc == 1) & (s.nodal_quant == 1) & s.surv_months.notna() & (s.cod_unknown == 0)]
    d = d[d[['age_mid', 'male', 'T', 'N', 'M']].notna().all(axis=1)]
    add('SEER primary prognostic cohort n', 30649, len(d), group=g)
    add('SEER primary prognostic CSS events', 243, d.css_event.sum(), group=g)
    p_m1 = 3 + (d['T'].nunique() - 1) + (d.M.nunique() - 1) + (d.N.nunique() - 1)
    epp = d.css_event.sum() / p_m1
    add('SEER EPP for M1', 24.3, epp, 0.1, group=g)
    add('SEER unpenalised Cox primary (EPP >= 20)', 20, epp, kind='ge', group='primary')
section('SEER cohort', 'descriptive/audit', seer_checks)

# ======================================================================
# Prognostic — optimism-corrected performance (primary)
def prog_checks():
    S2 = P['stage2']
    perf = load_csv(os.path.join(S2, 's2_performance.csv'), 's2_performance', 'primary')
    dl = load_csv(os.path.join(S2, 's2_paired_delta.csv'), 's2_paired_delta', 'primary')
    cal = load_csv(os.path.join(S2, 's2_calibration.csv'), 's2_calibration', 'primary')
    hr = load_csv(os.path.join(P['prog'], 'prog_hr.csv'), 'prog_hr (stage 1)', 'primary')
    if perf is None or dl is None or cal is None or hr is None: return
    pr = perf[(perf.variant == 'primary (unstratified)') & (perf.metric == "Uno's C (tau = 24 m)")].set_index('model')
    add('Bootstrap replicates B (primary)', 500, pr.B.min())
    for m, e in [('M0 core', 0.9094), ('M1 core + N', 0.9181), ('M2 core + LNR', 0.9091), ('M3 core + LODDS', 0.9071)]:
        add(f"Uno's C tau=24, optimism-corrected: {m}", e, pr.loc[m, 'corrected'], 0.001)
    q = dl[(dl.variant == 'primary (unstratified)') & (dl.metric == "Uno's C (tau = 24 m)")].set_index('comparison')
    for c, e in [('M2 core + LNR - M1 core + N', -0.0089), ('M3 core + LODDS - M1 core + N', -0.0110), ('M1 core + N - M0 core', 0.0087)]:
        add(f'Corrected paired dC: {c}', e, q.loc[c, 'corrected_delta'], 0.001)
    for c in ['M2 core + LNR - M1 core + N', 'M3 core + LODDS - M1 core + N']:
        add(f'Holm-adjusted p: {c}', 0.032, q.loc[c, 'p_holm_primary'], 0.01)
    cl = cal[cal.variant == 'primary (unstratified)'].set_index('model')['corrected Calibration slope (24 m)']
    for m, e in [('M0 core', 0.965), ('M1 core + N', 0.960), ('M2 core + LNR', 0.964), ('M3 core + LODDS', 0.964)]:
        add(f'Corrected 24-m calibration slope: {m}', e, cl[m], 0.005)
    l = hr[(hr.model == 'M2 core + LNR') & (hr.predictor == 'LNR')].iloc[0]
    add('LNR HR in M2', 1.85, l.HR, 0.01); add('LNR HR in M2, CI low', 1.27, l.CI_low, 0.01); add('LNR HR in M2, CI high', 2.70, l.CI_high, 0.01)
    # robustness within the primary analysis
    for c in ['M2 core + LNR - M0 core', 'M3 core + LODDS - M0 core']:
        add(f'Primary: {c} CI includes 0 (no incremental discrimination)', 1, int(q.loc[c, 'CI_low'] <= 0 <= q.loc[c, 'CI_high']), kind='true')
    for c in ['M2 core + LNR - M1 core + N', 'M3 core + LODDS - M1 core + N']:
        add(f'Primary: {c} CI below 0 (lower than N)', 1, int(q.loc[c, 'CI_high'] < 0), kind='true')
    add('DCA eligibility: all corrected slopes within 0.8-1.2', 1, int(cl.between(0.8, 1.2).all()), kind='true')
section('Prognostic primary', 'primary', prog_checks)

def bridge_ridge_checks():
    b = load_csv(os.path.join(P['bridge'], 'bridge_slopes.csv'), 'bridge_slopes 2024', 'primary')
    if b is not None:
        b = b.set_index('analysis')
        add('Bridge: ThyroidXL OR per 10 y', 0.766, b.loc['ThyroidXL full cohort (primary)', 'estimate'], 0.002)
        add('Bridge: SEER DTC HR per 10 y', 2.445, b.loc['SEER DTC (primary)', 'estimate'], 0.002)
    rp = os.path.join(P['stage2'], 'ridge_cv.json')
    if not os.path.exists(rp):
        fail('Ridge CV lambda', 'sensitivity', f'missing file {rp}'); return
    r = json.load(open(rp))
    for m, e in [('M0 core', 0.01), ('M1 core + N', 0), ('M2 core + LNR', 0.01), ('M3 core + LODDS', 0.01)]:
        add(f'Ridge lambda selected by CV: {m}', e, float(r[m]['selected']), 0.0, group='sensitivity')
section('Bridge and ridge', 'primary', bridge_ridge_checks)

# ======================================================================
# Robustness (sensitivity)
def robust_checks():
    S2 = P['stage2']; g = 'sensitivity'
    sens = load_csv(os.path.join(S2, 'sens_results.csv'), 'sens_results', g)
    dl = load_csv(os.path.join(S2, 's2_paired_delta.csv'), 's2_paired_delta', g)
    cal = load_csv(os.path.join(S2, 's2_calibration.csv'), 's2_calibration', g)
    dca = load_csv(os.path.join(S2, 'dca_24m_cv.csv'), 'dca_24m_cv', g)
    if sens is None or dl is None or cal is None or dca is None: return
    pr = sens[sens.kind == 'paired']
    exp_analyses = ['S-COD-A', 'S-COD-B', 'S-EX90', 'S-TREAT', 'S-RCS', 'S-D5', 'S-ALLHIST', 'S-RIDGE']
    for an in exp_analyses:
        add(f'Sensitivity present: {an}', 1, int(pr.analysis.str.startswith(an).sum() == 5), kind='true', group=g)
    inc0 = lambda r: int(r.CI_low <= 0 <= r.CI_high)
    for an in exp_analyses:
        sub = pr[pr.analysis.str.startswith(an)].set_index('model')
        for c in ['M2 core + LNR - M0 core', 'M3 core + LODDS - M0 core']:
            add(f'{an}: {c} CI includes 0', 1, inc0(sub.loc[c]), kind='true', group=g)
    for an in ['S-COD-A', 'S-COD-B']:
        sub = pr[pr.analysis.str.startswith(an)].set_index('model')
        for c in ['M2 core + LNR - M1 core + N', 'M3 core + LODDS - M1 core + N']:
            add(f'{an}: {c} CI below 0', 1, int(sub.loc[c, 'CI_high'] < 0), kind='true', group=g)
    d5 = pr[pr.analysis.str.startswith('S-D5')].set_index('model')
    for c in ['M2 core + LNR - M1 core + N', 'M3 core + LODDS - M1 core + N', 'M1 core + N - M0 core']:
        add(f'S-D5 (nodes >= 5): {c} CI includes 0 (no clear difference)', 1, inc0(d5.loc[c]), kind='true', group=g)
    st = dl[(dl.variant == 'post hoc: stratified by histology') & (dl.metric == "Uno's C (tau = 24 m)")].set_index('comparison')
    for c in ['M2 core + LNR - M0 core', 'M3 core + LODDS - M0 core']:
        add(f'FTC-stratified: {c} CI includes 0', 1, inc0(st.loc[c]), kind='true', group=g)
    for c in ['M2 core + LNR - M1 core + N', 'M3 core + LODDS - M1 core + N']:
        add(f'FTC-stratified: {c} CI below 0', 1, int(st.loc[c, 'CI_high'] < 0), kind='true', group=g)
    ms = sens[sens.kind == 'model']
    add('All prespecified sensitivities: min calibration slope >= 0.8', 0.8, ms.cal_slope.min(), kind='ge', group=g)
    add('All prespecified sensitivities: max calibration slope <= 1.2', 1.2, ms.cal_slope.max(), kind='le', group=g)
    cs = cal[cal.variant.str.startswith('post hoc')]['corrected Calibration slope (24 m)']
    add('FTC-stratified: corrected slopes within 0.8-1.2', 1, int(cs.between(0.8, 1.2).all()), kind='true', group=g)
    nb = dca.pivot(index='threshold', columns='model', values='net_benefit')
    add('DCA (Supplementary only): max between-model net-benefit difference <= 0.0005', 0.0005,
        (nb.max(axis=1) - nb.min(axis=1)).max(), kind='le', group=g)
    add('DCA: all models above treat-all at every threshold', 1,
        int((dca.net_benefit > dca.nb_treat_all).all()), kind='true', group=g)
section('Robustness', 'sensitivity', robust_checks)

# ======================================================================
rep = pd.DataFrame(ROWS, columns=['check', 'expected', 'got', 'tol', 'result', 'group'])
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
rep.to_csv(OUT, index=False)
pd.set_option('display.width', 250, 'display.max_rows', 500, 'display.max_colwidth', 90)
print(rep.to_string(index=False))
n_fail = (rep.result != 'PASS').sum()
print(f'\nTotal {len(rep)} | PASS {len(rep) - n_fail} | FAIL {n_fail}')
for gname, gg in rep.groupby('group'):
    print(f'  {gname}: {len(gg)} checks, {int((gg.result == "PASS").sum())} PASS')
sys.exit(1 if n_fail else 0)
