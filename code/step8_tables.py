"""
STEP 8 — Main and supplementary tables from the frozen v1.1 results (24 Oct 2026). No model is fitted here.

Usage (from the code folder):
    python step8_tables.py [--data ..\\data\\processed_ref2024] [--results ..\\results] [--out ..\\tables]

Outputs
  Table1_characteristics.tex, Table2_diagnostic.tex, Table3_prognostic.tex
  Supplementary_Tables.xlsx (one sheet per table) and TableS*.tex
LaTeX conventions: tabular* with booktabs; footnotes as \smallskip\noindent{\footnotesize ...}; no threeparttable.
Captions are short placeholders; final titles are set in the manuscript.
"""
import argparse, os, json, re
import numpy as np, pandas as pd
from lifelines import KaplanMeierFitter

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join('..', 'data', 'processed_ref2024'))
ap.add_argument('--results', default=os.path.join('..', 'results'))
ap.add_argument('--out', default=os.path.join('..', 'tables'))
a = ap.parse_args()
R, OUT = a.results, a.out
os.makedirs(OUT, exist_ok=True)
D = {y: os.path.join(R, f'diagnostic_ref{y}') for y in ['2024', '2023', '2025', 'exclude']}
S2, PR, BR = os.path.join(R, 'stage2'), os.path.join(R, 'prognostic'), os.path.join(R, 'bridge_ref2024')

def rd(path, **kw):
    if not os.path.exists(path): raise FileNotFoundError(f'Required input missing: {path}')
    return pd.read_csv(path, low_memory=False, **kw)
def n_(x): return f'{int(x):,}'
def pct(k, n, d=1): return f'{n_(k)} ({100 * k / n:.{d}f})'
def ci(e, lo, hi, d=3): return f'{e:.{d}f} ({lo:.{d}f} to {hi:.{d}f})'
def sci(e, lo, hi, d=4): return f'{e:+.{d}f} ({lo:+.{d}f} to {hi:+.{d}f})'
def pv(p): return '<0.001' if p < 0.001 else f'{p:.3f}'
def miqr(s, d=0): q = s.quantile([.5, .25, .75]); return f'{q.iloc[0]:.{d}f} ({q.iloc[1]:.{d}f}–{q.iloc[2]:.{d}f})'
def strip_ci(s): return s.replace('-', '–')

TEXMAP = [('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('≥', r'$\geq$'), ('≤', r'$\leq$'),
          ('−', '$-$'), ('–', '--'), ('τ', r'$\tau$'), ('Δ', r'$\Delta$'), ('×', r'$\times$'), ('λ', r'$\lambda$'), ('→', r'$\rightarrow$')]
def tex(s):
    s = str(s)
    s = re.sub(r'(?<![\w.])-0\.0+(?![\d])', '0.00', s)            # negative zero
    s = s.replace('<', '\x00LT').replace('>', '\x00GT')
    s = re.sub(r'(?<![\w.\-])-(?=\d)', '\x00MINUS', s)               # numeric minus sign (hyphen-minus only)
    for a_, b_ in TEXMAP: s = s.replace(a_, b_)
    s = s.replace('—', '---').replace('\x00LT', '$<$').replace('\x00GT', '$>$').replace('\x00MINUS', '$-$')
    return s
def write_tex(df, name, caption, label, notes='', align=None, section_col=None, size='footnotesize', resize=False, colsep=None):
    # header cells may contain '\n' for a two-line header; resize=True (Supplementary only) scales a plain tabular to text width
    cols = list(df.columns); k = len(cols)
    align = align or ('l' + 'c' * (k - 1))
    if resize:
        L = [r'\begin{table}[htbp]', rf'\caption{{{tex(caption)}}}\label{{{label}}}', rf'\{size}',
             r'\resizebox{\textwidth}{!}{%', rf'\begin{{tabular}}{{@{{}}{align}@{{}}}}', r'\toprule']
    else:
        L = [r'\begin{table}[htbp]', rf'\caption{{{tex(caption)}}}\label{{{label}}}', rf'\{size}',
             *([rf'\setlength{{\tabcolsep}}{{{colsep}}}'] if colsep else []),
             rf'\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}}{align}@{{}}}}', r'\toprule']
    parts = [str(c).split('\n') for c in cols]; nl = max(len(p_) for p_ in parts)
    for i in range(nl):
        L.append(' & '.join(tex(p_[i]) if i < len(p_) else '' for p_ in parts) + r' \\')
    L.append(r'\midrule')
    for _, r in df.iterrows():
        if section_col and str(r[cols[0]]).startswith('§'):
            L.append(rf'\multicolumn{{{k}}}{{@{{}}l}}{{\textit{{{tex(str(r[cols[0]])[1:])}}}}} \\')
        else:
            L.append(' & '.join(tex('' if pd.isna(v) else v) for v in r.values) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}}' if resize else r'\end{tabular*}']
    if notes: L.append(rf'\smallskip\noindent{{\footnotesize {tex(notes)}}}')
    L.append(r'\end{table}')
    open(os.path.join(OUT, name + '.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n'); print('saved', name)

# =====================================================================
# Table 1 — cohort characteristics
xl = rd(os.path.join(a.data, 'thyroidxl_clean_v2.csv')); A = xl[xl.in_setA == 1]; nA = len(A)
s = rd(os.path.join(a.data, 'seer_clean_v2.csv'))
d = s[(s.dtc == 1) & (s.nodal_quant == 1) & s.surv_months.notna() & (s.cod_unknown == 0)]
d = d[d[['age_mid', 'male', 'T', 'N', 'M']].notna().all(axis=1)].copy(); nD = len(d)
assert nA == 4088 and nD == 30649
rows = [('§Diagnostic cohort (ThyroidXL)', ''), ('Patients', n_(nA)),
        ('Age, years, median (IQR)', miqr(A.age)), ('Female', pct((A.gender == 2).sum(), nA))]
rows.append(('ACR TI-RADS category', ''))
for t in sorted(A.tirads.dropna().unique()): rows.append((f'   TR{int(t)}', pct((A.tirads == t).sum(), nA)))
rows.append(('Bethesda category', ''))
for b, lab in [(1, 'I'), (2, 'II'), (3, 'III'), (4, 'IV'), (5, 'V'), (6, 'VI')]: rows.append((f'   {lab}', pct((A.fnac == b).sum(), nA)))
rows.append(('   FNAC not available', pct(A.fnac.isna().sum(), nA)))
rows.append(('Malignant (reference standard)', pct(A.malignant.sum(), nA)))
rows.append(('Histopathology available', pct(A.histopathology.notna().sum(), nA)))
rows.append(('§Prognostic cohort (SEER DTC)', '')); rows.append(('Patients', n_(nD)))
def lower(b): b = str(b); return 0 if b.startswith('00') else (90 if '90+' in b else int(b.split('-')[0]))
lw = d.age_bin.apply(lower)
rows.append(('Age group, years', ''))
for lo_, hi_, lab in [(0, 45, '<45'), (45, 55, '45–54'), (55, 65, '55–64'), (65, 75, '65–74'), (75, 200, '≥75')]:
    rows.append((f'   {lab}', pct(((lw >= lo_) & (lw < hi_)).sum(), nD)))
rows += [('Male', pct(d.male.sum(), nD)), ('Histology', ''), ('   Papillary', pct((d.histology == 'PTC').sum(), nD)),
         ('   Follicular (incl. oncocytic)', pct((d.histology == 'FTC').sum(), nD)), ('T category', '')]
for t in ['T1', 'T2', 'T3', 'T4']: rows.append((f'   {t}', pct((d['T'] == t).sum(), nD)))
rows.append(('AJCC N category', ''))
for t in ['N0', 'N1a', 'N1b', 'N1 NOS']: rows.append((f'   {t}', pct((d.N == t).sum(), nD)))
rows.append(('M1', pct((d.M == 'M1').sum(), nD)))
rows += [('Lymph nodes examined, median (IQR)', miqr(d.D)), ('Positive lymph nodes, median (IQR)', miqr(d.NPLN)),
         ('LNR, median (IQR)', miqr(d.LNR, 2)), ('Total thyroidectomy', pct((d.surgery_code == 50).sum(), nD)),
         ('Radioactive iodine', pct(d.radiation.isin(['Radioisotopes (1988+)', 'Combination of beam with implants or isotopes']).sum(), nD))]
t_ = d.surv_months.where(d.surv_months > 0, 0.5)
fu = KaplanMeierFitter().fit(t_, 1 - d.dead).median_survival_time_
rows += [('Follow-up, months, median (reverse Kaplan–Meier)', f'{fu:.0f}'), ('CSS deaths', pct(d.css_event.sum(), nD)),
         ('Other-cause deaths', pct(d.other_death.sum(), nD))]
T1 = pd.DataFrame(rows, columns=['Characteristic', 'Value'])
write_tex(T1, 'Table1_characteristics', 'Characteristics of the diagnostic and prognostic cohorts', 'tab:characteristics',
          'Values are n (%) unless stated. Diagnostic malignancy: histopathology over FNAC. Prognostic cohort: DTC with quantitative '
          'nodal data, known cause of death and complete T, N and M. IQR, interquartile range; LNR, lymph node ratio; CSS, cancer-specific survival.',
          align='lr', section_col=True)

# =====================================================================
# Table 2 — diagnostic performance (primary)
perf = rd(os.path.join(D['2024'], 'diag_performance.csv')); dl = rd(os.path.join(D['2024'], 'diag_paired_delta.csv'))
cal = rd(os.path.join(D['2024'], 'diag_calibration.csv')); th = rd(os.path.join(D['2024'], 'diag_thresholds.csv'))
MOD = [('TIRADS (empirical, fold-wise)', 'ACR TI-RADS'), ('Logistic Regression', 'Logistic regression'),
       ('Random Forest', 'Random forest'), ('XGBoost', 'XGBoost')]
P = perf[perf.analysis.str.startswith('P:')].set_index('model'); Q = dl[dl.analysis.str.startswith('P:')].set_index('comparison')
C = cal[cal.analysis.str.startswith('P:')].set_index('model')
spec = th.set_index('rule').Specificity
def spec_for(m):
    key = 'TIRADS' if m.startswith('TIRADS') else m
    return strip_ci(spec[spec.index.str.startswith(key)].iloc[0])
rows = []
for m, lab in MOD:
    lo_, hi_ = P.loc[m, 'AUC_CI'].split('-')
    r = {'Model': lab, 'AUC\n(95% CI)': f'{P.loc[m, "AUC"]:.3f} ({lo_}–{hi_})'}
    if m.startswith('TIRADS'): r.update({'ΔAUC\n(95% CI)': 'Reference', 'Holm\np': '—'})
    else:
        q = Q.loc[f'{m} - TIRADS']; r.update({'ΔAUC\n(95% CI)': f'{q.delta_AUC:.3f} ({q.CI_low:.3f}–{q.CI_high:.3f})', 'Holm\np': pv(q.delong_p_holm)})
    r.update({'Calibration\nintercept': f'{C.loc[m, "calibration_intercept"]:.2f}', 'Calibration\nslope': f'{C.loc[m, "calibration_slope"]:.2f}',
              'Brier\nscore': f'{C.loc[m, "brier"]:.3f}', 'Specificity*\n(95% CI)': spec_for(m)})
    rows.append(r)
T2 = pd.DataFrame(rows)
write_tex(T2, 'Table2_diagnostic', 'Diagnostic discrimination, calibration and operating-point specificity', 'tab:diagnostic',
          f'n = {n_(P.loc["XGBoost", "n"])}; malignant {n_(P.loc["XGBoost", "malignant"])}. Stratified 5-fold cross-validation (out-of-fold predictions); '
          'AUC CIs and paired ΔAUC CIs from 2,000 patient-level bootstrap resamples; Holm-adjusted paired DeLong tests across the three '
          'co-primary comparisons; ΔAUC versus ACR TI-RADS. Intercept and slope: calibration intercept and slope. *Specificity at the sensitivity of ACR TI-RADS ≥4 (98.5%); model thresholds chosen on the same out-of-fold '
          'predictions; Wilson 95% CIs.', colsep='3pt')

# =====================================================================
# Table 3 — prognostic performance and paired differences
perf2 = rd(os.path.join(S2, 's2_performance.csv')); pdl = rd(os.path.join(S2, 's2_paired_delta.csv')); c2 = rd(os.path.join(S2, 's2_calibration.csv'))
PM = [('M0 core', 'M0 core'), ('M1 core + N', 'M1 core + N category'), ('M2 core + LNR', 'M2 core + LNR'), ('M3 core + LODDS', 'M3 core + LODDS')]
pp = perf2[perf2.variant == 'primary (unstratified)']
UNO, HAR = "Uno's C (tau = 24 m)", "Harrell's C"
def g(m, met, col): return pp[(pp.model == m) & (pp.metric == met)][col].iloc[0]
rows = []
for m, lab in PM:
    rows.append({'Model': lab, "Uno's C\napparent": f'{g(m, UNO, "apparent"):.3f}', "Uno's C\ncorrected": f'{g(m, UNO, "corrected"):.3f}',
                 "Harrell's C\ncorrected": f'{g(m, HAR, "corrected"):.3f}',
                 'tdAUC 12 m\ncorrected': f'{g(m, "tdAUC 12 m", "corrected"):.3f}', 'tdAUC 24 m\ncorrected': f'{g(m, "tdAUC 24 m", "corrected"):.3f}',
                 'Calibration\nslope': f'{g(m, "Calibration slope (24 m)", "corrected"):.2f}', 'O:E\n24 m': f'{g(m, "O:E (24 m)", "apparent"):.2f}'})
T3a = pd.DataFrame(rows)
U = pdl[(pdl.variant == 'primary (unstratified)') & (pdl.metric == "Uno's C (tau = 24 m)")].set_index('comparison')
CMP = [('M2 core + LNR - M1 core + N', 'LNR vs N category', 'Primary'), ('M3 core + LODDS - M1 core + N', 'LODDS vs N category', 'Primary'),
       ('M1 core + N - M0 core', 'N category vs core', 'Secondary'), ('M2 core + LNR - M0 core', 'LNR vs core', 'Secondary'),
       ('M3 core + LODDS - M0 core', 'LODDS vs core', 'Secondary')]
T3b = pd.DataFrame([{'Comparison': lab, 'Role': role, "ΔUno's C (95% CI)": sci(U.loc[c, 'corrected_delta'], U.loc[c, 'CI_low'], U.loc[c, 'CI_high']),
                     'Holm p': pv(U.loc[c, 'p_holm_primary']) if role == 'Primary' else '—'} for c, lab, role in CMP])
write_tex(T3a, 'Table3a_prognostic_performance', 'Prognostic discrimination and 24-month calibration', 'tab:prognostic',
          'n = 30,649; 243 CSS deaths. Unpenalised Cox models; core = age, sex, histology, T and M category. Corrected values are '
          'optimism-corrected (Harrell bootstrap, 500 resamples, full refit); Uno\'s C truncated at τ = 24 months; tdAUC, time-dependent '
          'AUC; calibration slope optimism-corrected; O:E, observed (Kaplan–Meier) to expected 24-month risk, apparent.')
write_tex(T3b, 'Table3b_prognostic_paired', "Paired differences in optimism-corrected Uno's C at 24 months", 'tab:paired',
          'Same bootstrap resamples for all models; 95% CI = percentile interval of bootstrap differences shifted by the mean optimism of '
          'the difference; Holm adjustment across the two primary comparisons.', align='llcc')

# =====================================================================
# Supplementary tables
SUP = {}
A_ = A.copy(); A_['Bethesda'] = A_.fnac.map({1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}).fillna('No FNAC')
A_['Released label (source)'] = A_.released_result + ' (' + A_.label_source + ')'
ct = pd.crosstab(A_.Bethesda, A_['Released label (source)'], margins=True, margins_name='Total')
ct = ct.reindex(['I', 'II', 'III', 'IV', 'V', 'VI', 'No FNAC', 'Total']).reset_index().rename(columns={'Bethesda': 'Bethesda category'})
SUP['S1_label_provenance'] = ('Bethesda category by released label and label source (ThyroidXL, n = 4,088)', ct, '')
disc = A[(A.label_changed == 1) | (A.hist_class == 'uncertain')][['patient_id', 'released_result', 'label_source', 'fnac', 'histopathology', 'malignant']]
disc = disc.assign(fnac=disc.fnac.map({1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}).fillna('—'),
                   malignant=disc.malignant.map({1: 'Malignant', 0: 'Benign'})).rename(columns={
    'patient_id': 'Patient', 'released_result': 'Released label', 'label_source': 'Source', 'fnac': 'Bethesda',
    'histopathology': 'Histopathology text', 'malignant': 'Primary label'})
SUP['S2_label_discordance'] = ('Released labels discordant with histopathology text', disc,
                               'Follicular thyroid tumour (uncertain malignant potential) kept non-malignant and excluded from S-D1.')
rows = []
for y in ['2024', '2023', '2025', 'exclude']:
    pf = rd(os.path.join(D[y], 'diag_performance.csv')); pdd = rd(os.path.join(D[y], 'diag_paired_delta.csv'))
    for an in pf.analysis.unique():
        if y != '2024' and not an.startswith('P:'): continue
        sub = pf[pf.analysis == an].set_index('model')
        for m, lab in MOD:
            r = sub.loc[m]; q = pdd[(pdd.analysis == an) & (pdd.comparison == f'{m} - TIRADS')]
            rows.append({'Age reference': y if y != 'exclude' else 'birth-year ages excluded', 'Analysis': an, 'Model': lab, 'n': n_(r.n),
                         'Malignant': n_(r.malignant), 'AUC (95% CI)': f'{r.AUC:.3f}' + (f' ({strip_ci(r.AUC_CI)})' if isinstance(r.AUC_CI, str) else ''),
                         'ΔAUC vs ACR TI-RADS (95% CI)': sci(q.delta_AUC.iloc[0], q.CI_low.iloc[0], q.CI_high.iloc[0], 3) if len(q) else '—'})
SUP['S3_diagnostic_sensitivity'] = ('Diagnostic sensitivity, supportive and supplementary analyses', pd.DataFrame(rows),
                                    'Set B (FNAC as predictor) is subject to incorporation bias; within S-D1 it is descriptive only (88 benign). Primary age reference year 2024.')
b3 = rd(os.path.join(D['2024'], 'diag_bethesda3.csv')); b3['model'] = b3.model.map(dict(MOD)); b3['AUC'] = b3.AUC.round(3)
SUP['S4_bethesda_III'] = ('Exploratory discrimination within Bethesda III (n = 39)', b3.rename(columns={'model': 'Model', 'n': 'n', 'malignant': 'Malignant'}),
                          'Out-of-fold predictions of the primary models evaluated in the subgroup; no between-model inference.')
thf = th.copy(); thf['rule'] = thf.rule.str.replace(r' \(sensitivity matched.*', ' (sensitivity-matched)', regex=True)
for c in ['Sensitivity', 'Specificity', 'PPV', 'NPV']: thf[c] = thf[c].map(strip_ci)
SUP['S5_operating_points'] = ('Operating-point performance (primary diagnostic cohort)', thf.rename(columns={'rule': 'Rule'}), 'Wilson 95% CIs.')
hr = rd(os.path.join(PR, 'prog_hr.csv'))
hr['HR (95% CI)'] = [ci(h, l, u, 2) for h, l, u in zip(hr.HR, hr.CI_low, hr.CI_high)]; hr['p'] = hr.p.map(pv)
SUP['S6_hazard_ratios'] = ('Hazard ratios, primary prognostic models (apparent)', hr[['model', 'predictor', 'HR (95% CI)', 'p']].rename(
    columns={'model': 'Model', 'predictor': 'Predictor'}), 'Age per year; LNR per unit (0 to 1); LODDS per unit.')
ph = rd(os.path.join(PR, 'prog_ph_tests.csv')); ph['p'] = ph.p.map(pv); ph['test_statistic'] = ph.test_statistic.round(2)
SUP['S7_ph_tests'] = ('Proportional-hazards tests (rank-transformed time)', ph.rename(columns={'model': 'Model', 'predictor': 'Predictor', 'test_statistic': 'Statistic'}), '')
pf2 = perf2.copy(); pf2[['apparent', 'optimism', 'corrected']] = pf2[['apparent', 'optimism', 'corrected']].round(4)
SUP['S8_prognostic_performance_all'] = ('Apparent, optimism and corrected performance, primary and post hoc histology-stratified models',
                                        pf2.drop(columns='B').rename(columns={'variant': 'Variant', 'model': 'Model', 'metric': 'Metric'}), 'B = 500 bootstrap resamples.')
pd2 = pdl.copy(); pd2['Δ (95% CI)'] = [sci(e, l, h) for e, l, h in zip(pd2.corrected_delta, pd2.CI_low, pd2.CI_high)]
pd2['Holm p'] = pd2.p_holm_primary.map(lambda v: '—' if pd.isna(v) else pv(v))
SUP['S9_paired_all_metrics'] = ('Paired optimism-corrected differences, all metrics, primary and histology-stratified models',
                                pd2[['variant', 'metric', 'comparison', 'role', 'Δ (95% CI)', 'Holm p', 'interpretation']].rename(
                                    columns={'variant': 'Variant', 'metric': 'Metric', 'comparison': 'Comparison', 'role': 'Role', 'interpretation': 'Reading'}),
                                'Histology-stratified models were added post hoc after a proportional-hazards violation for FTC.')
sens = rd(os.path.join(S2, 'sens_results.csv'))
sm = sens[sens.kind == 'model'][['analysis', 'model', 'n', 'events', 'lam', 'Uno_C_24', 'tdAUC_24', 'cal_slope', 'OE']].copy()
sm[['Uno_C_24', 'tdAUC_24']] = sm[['Uno_C_24', 'tdAUC_24']].round(3); sm[['cal_slope', 'OE']] = sm[['cal_slope', 'OE']].round(2)
sm['n'] = sm.n.astype(int); sm['events'] = sm.events.astype(int)
SUP['S10_prognostic_sensitivity_models'] = ('Prespecified prognostic sensitivity analyses: model performance (apparent)', sm.rename(columns={
    'analysis': 'Analysis', 'model': 'Model', 'events': 'CSS deaths', 'lam': 'λ', 'Uno_C_24': "Uno's C 24 m", 'tdAUC_24': 'tdAUC 24 m',
    'cal_slope': 'Calibration slope', 'OE': 'O:E'}), '')
sp = sens[sens.kind == 'paired'].copy(); sp['ΔUno\'s C (95% CI)'] = [sci(e, l, h) for e, l, h in zip(sp.delta, sp.CI_low, sp.CI_high)]
SUP['S11_prognostic_sensitivity_paired'] = ('Prespecified prognostic sensitivity analyses: paired differences (apparent)',
                                            sp[['analysis', 'model', 'role', "ΔUno's C (95% CI)"]].rename(columns={'analysis': 'Analysis', 'model': 'Comparison', 'role': 'Role'}),
                                            'Refit bootstrap percentile 95% CIs, 200 resamples.')
tr = sens[sens.kind == 'HR in M1'].copy(); tr['HR (95% CI)'] = [ci(h, l, u, 2) for h, l, u in zip(tr.delta, tr.CI_low, tr.CI_high)]
SUP['S12_treatment_model_HR'] = ('Hazard ratios of the treatment-adjusted M1 model', tr[['model', 'HR (95% CI)']].rename(columns={'model': 'Predictor'}),
                                 'Treatment variables were added as a sensitivity analysis only; RAI estimate reflects confounding by indication and is not causal.')
dh = rd(os.path.join(S2, 'desc_histology.csv')); comp = rd(os.path.join(S2, 'desc_special_code_composition.csv'))
dh['css_pct'] = dh.css_pct.round(2); dh['km_css_risk_24m_pct'] = dh.km_css_risk_24m_pct.round(2)
SUP['S13_histology_special_codes'] = ('Histology and special nodal-code descriptives (SEER)', dh.drop(columns='note').rename(columns={
    'group': 'Group', 'css_events': 'CSS deaths', 'css_pct': 'CSS deaths (%)', 'km_css_risk_24m_pct': '24-m KM CSS risk (%)', 'note': 'Note'}),
    'Known cause of death and survival time available. Special nodal codes 95-97: examined 95/96/97 or positive 95/97 (aspiration only or number unknown); LNR and LODDS not computable; excluded from the primary cohort. In the full cleaned file: 840 records (145 CSS deaths), DTC 675 (56).')
bs = rd(os.path.join(BR, 'bridge_strata.csv')); bsl = rd(os.path.join(BR, 'bridge_slopes.csv')); bas = rd(os.path.join(BR, 'bridge_age_sex.csv'))
SUP['S14_age_strata'] = ('Age strata: diagnostic malignancy and prognostic 24-month CSS cumulative incidence', bs[['stratum', 'xl_n', 'xl_malignant', 'xl_pct', 'xl_ci',
    'seer_dtc_n', 'seer_dtc_css_events', 'seer_dtc_cif24_pct', 'seer_dtc_cif24_ci']].round(2).rename(columns={
    'stratum': 'Age', 'xl_n': 'ThyroidXL n', 'xl_malignant': 'Malignant', 'xl_pct': 'Malignant (%)', 'xl_ci': 'Wilson 95% CI',
    'seer_dtc_n': 'SEER DTC n', 'seer_dtc_css_events': 'CSS deaths', 'seer_dtc_cif24_pct': '24-m CIF (%)', 'seer_dtc_cif24_ci': '95% CI'}),
    'Independent cohorts; within-cohort associations only.')
bsl['Estimate (95% CI)'] = [ci(e, l, h, 3) for e, l, h in zip(bsl.estimate, bsl.ci_low, bsl.ci_high)]
SUP['S15_age_slopes'] = ('Within-cohort age effects per 10 years (sex-adjusted)', bsl[['analysis', 'measure', 'Estimate (95% CI)', 'n', 'events']].rename(
    columns={'analysis': 'Analysis', 'measure': 'Measure', 'events': 'Events'}), '')
SUP['S16_age_sex'] = ('Age by sex: diagnostic malignancy and prognostic 24-month CSS cumulative incidence', bas.round(2), '')
rc = json.load(open(os.path.join(S2, 'ridge_cv.json')))
rcv = pd.DataFrame([{'Model': m, **{f'λ = {k}': round(v, 1) for k, v in d_['cv_partial_loglik'].items()}, 'Selected λ': d_['selected']} for m, d_ in rc.items()])
SUP['S17_ridge_cv'] = ('Ridge sensitivity: 5-fold cross-validated partial log-likelihood by λ', rcv, '')
dca = rd(os.path.join(S2, 'dca_24m_cv.csv')); dsel = dca[dca.threshold.round(4).isin([0.005, 0.01, 0.02, 0.03, 0.05, 0.10])].copy()
dsel['Net benefit per 1,000'] = (dsel.net_benefit * 1000).round(2); dsel['Treat all per 1,000'] = (dsel.nb_treat_all * 1000).round(2)
dsel['Threshold (%)'] = (dsel.threshold * 100).round(1); dsel['Positive (%)'] = dsel.pct_positive.round(1)
SUP['S18_dca'] = ('Decision-curve analysis at selected thresholds (exploratory)', dsel[['model', 'Threshold (%)', 'Positive (%)', 'Net benefit per 1,000', 'Treat all per 1,000']].rename(
    columns={'model': 'Model'}), '10-fold cross-validated 24-month risks.')
cd = rd(os.path.join(S2, 'cal_deciles_primary.csv'))
for c in ['mean_predicted', 'observed_KM', 'obs_low', 'obs_high']: cd[c] = (cd[c] * 100).round(3)
SUP['S19_calibration_deciles'] = ('24-month calibration by decile of predicted risk (apparent)', cd.drop(columns='variant').rename(columns={
    'model': 'Model', 'decile': 'Decile', 'events_24m': 'CSS deaths by 24 m', 'mean_predicted': 'Predicted (%)', 'observed_KM': 'Observed KM (%)',
    'obs_low': 'Lower 95% CI', 'obs_high': 'Upper 95% CI'}), '')

with pd.ExcelWriter(os.path.join(OUT, 'Supplementary_Tables.xlsx'), engine='openpyxl') as xw:
    idx = pd.DataFrame([{'Sheet': k, 'Title': v[0], 'Note': v[2]} for k, v in SUP.items()]); idx.to_excel(xw, sheet_name='Index', index=False)
    for k, (title_, df, note) in SUP.items():
        df.to_excel(xw, sheet_name=k[:31], index=False, startrow=2)
        ws = xw.sheets[k[:31]]; ws['A1'] = title_
        if note: ws.cell(row=len(df) + 5, column=1, value=note)
print('saved Supplementary_Tables.xlsx')
WIDE = {'S3', 'S5', 'S9', 'S10', 'S13', 'S17'}   # Supplementary tables scaled to text width
for i, (k, (title_, df, note)) in enumerate(SUP.items(), start=1):
    write_tex(df.astype(str).replace('nan', ''), f'Table{k.split("_")[0]}', title_, f'tab:{k.lower()}', note, size='scriptsize',
              resize=k.split('_')[0] in WIDE)
print('All tables written to', os.path.abspath(OUT))
