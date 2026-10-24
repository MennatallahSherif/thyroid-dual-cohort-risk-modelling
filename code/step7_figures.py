"""
STEP 7 — Journal-style figures from the frozen v1.1 results (24 Oct 2026). No model is fitted here.

Usage (from the code folder):
    python step7_figures.py [--data ..\\data\\processed_ref2024] [--results ..\\results] [--out ..\\figures]

Style: sans-serif 7-8 pt, double-column width 180 mm, Okabe-Ito colour-blind-safe palette, bold panel letters,
       vector PDF + 600-dpi PNG + 600-dpi TIFF for every figure.
Figure rules (reviewer comments and Stage-2 review):
  - every panel states its denominator (n, events) and whether estimates are apparent, cross-validated or optimism-corrected
  - time horizon stated wherever survival risk is shown (24 months)
  - deciles with no 24-month CSS death are not plotted on the log scale as if they had a value: shown as open markers on
    the lower axis edge and labelled
  - fixed terminology: ACR TI-RADS, AJCC N category, LNR, LODDS, CSS, short-term

Main      Fig1_flow, Fig2_diagnostic, Fig3_prognostic
Suppl.    FigS1_sensitivity_forest (incl. post hoc histology-stratified), FigS2_dca, FigS3_age_gradient
v2 (24 Oct 2026): visual redesign only (restrained journal style). Reads frozen result files; fits no model;
changes no number. Methods, CI methods, denominators and exploratory/post hoc labels are carried by the captions.
"""
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join('..', 'data', 'processed_ref2024'))
ap.add_argument('--results', default=os.path.join('..', 'results'))
ap.add_argument('--out', default=os.path.join('..', 'figures'))
a = ap.parse_args()
R, OUT = a.results, a.out
os.makedirs(OUT, exist_ok=True)
D2024, S2, BR = os.path.join(R, 'diagnostic_ref2024'), os.path.join(R, 'stage2'), os.path.join(R, 'bridge_ref2024')

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                     'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'lines.linewidth': 1.0, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 600})
MM = 1 / 25.4; W2 = 180 * MM
OI = {'black': '#000000', 'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73', 'blue': '#0072B2',
      'red': '#D55E00', 'pink': '#CC79A7', 'grey': '#7F7F7F'}
DCOL = {'TIRADS (empirical, fold-wise)': OI['black'], 'Logistic Regression': OI['blue'], 'Random Forest': OI['green'], 'XGBoost': OI['red']}
DLAB = {'TIRADS (empirical, fold-wise)': 'ACR TI-RADS', 'Logistic Regression': 'Logistic regression', 'Random Forest': 'Random forest', 'XGBoost': 'XGBoost'}
PCOL = {'M0 core': OI['grey'], 'M1 core + N': OI['blue'], 'M2 core + LNR': OI['orange'], 'M3 core + LODDS': OI['pink']}
PLAB = {'M0 core': 'M0 core', 'M1 core + N': 'M1 core + N category', 'M2 core + LNR': 'M2 core + LNR', 'M3 core + LODDS': 'M3 core + LODDS'}
CLAB = {'M2 core + LNR - M1 core + N': 'LNR vs N category', 'M3 core + LODDS - M1 core + N': 'LODDS vs N category',
        'M1 core + N - M0 core': 'N category vs core', 'M2 core + LNR - M0 core': 'LNR vs core', 'M3 core + LODDS - M0 core': 'LODDS vs core'}
COMPS = list(CLAB)

def need(path):
    if not os.path.exists(path): raise FileNotFoundError(f'Required input missing: {path}')
    return path
def save(fig, name):
    for ext in ['pdf', 'png', 'tiff']:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig); print('saved', name)
def letter(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='left')
def f(n): return f'{int(n):,}'

# ---- v2 visual design (24 Oct 2026): restrained journal style; captions carry methods, CI methods and denominators.
COMPS_MAIN = ['M1 core + N - M0 core', 'M2 core + LNR - M0 core', 'M3 core + LODDS - M0 core',
              'M2 core + LNR - M1 core + N', 'M3 core + LODDS - M1 core + N']
GREY = '0.45'
def title(ax, t, sub=None):
    ax.set_title(t, loc='left', fontsize=7.5, fontweight='bold', pad=10 if sub else 4)
    if sub: ax.text(0, 1.012, sub, transform=ax.transAxes, fontsize=6.3, color=GREY, va='bottom', ha='left')

# =====================================================================
# Figure 1 — minimal flow diagram (rectangular boxes; exclusions on the right)
flow = pd.read_csv(need(os.path.join(a.data, 'flow_counts_v2.csv')))
def fc(cohort, step):
    r = flow[(flow.cohort == cohort) & (flow.step.str.strip() == step)]
    if r.empty: raise KeyError(f'flow step not found: {cohort} / {step}')
    return r.iloc[0]
def rbox(ax, x, y, text, fc='white', bold=False, minw=0.0, fs=6.3, align='center'):
    t = ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal',
                linespacing=1.3, zorder=3, multialignment=align)
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer()).transformed(ax.transData.inverted())
    w, h = max(minw, bb.width + 0.05), bb.height + 0.035
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, lw=0.5, ec='black', fc=fc, zorder=2, clip_on=False))
    return x - w / 2, x + w / 2, y - h / 2, y + h / 2
def vline(ax, x, y1, y2, head=True):
    if head: ax.annotate('', xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle='-|>', lw=0.5, color='black', mutation_scale=5))
    else: ax.plot([x, x], [y1, y2], lw=0.5, color='black')
def hline(ax, x1, x2, y): ax.annotate('', xy=(x2, y), xytext=(x1, y), arrowprops=dict(arrowstyle='-|>', lw=0.5, color='black', mutation_scale=5))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(W2, 105 * MM), gridspec_kw={'wspace': 0.08})
for ax in (axA, axB): ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
FILL = '0.92'; MAINX, SIDEX = 0.33, 0.80
# panel a
raw = fc('ThyroidXL', 'Raw patients'); exa = fc('ThyroidXL', 'Excluded: unrecoverable age')
sa = fc('ThyroidXL', 'Set A analytic cohort (primary outcome)'); sd1 = fc('ThyroidXL', 'S-D1 histopathology-confirmed')
b1 = rbox(axA, MAINX, 0.88, f'ThyroidXL patients\nn = {f(raw.n)}', minw=0.42)
e1 = rbox(axA, SIDEX, 0.70, f'Excluded (n = {f(exa.n)})\nage not recoverable', fs=6.0, minw=0.30)
b2 = rbox(axA, MAINX, 0.50, f'Primary diagnostic cohort\nn = {f(sa.n)}; malignant {f(sa.events)}', fc=FILL, bold=True, minw=0.42)
vline(axA, MAINX, b1[2], b2[3]); hline(axA, MAINX, e1[0], 0.70)
b3 = rbox(axA, MAINX, 0.17, f'Histopathology-confirmed\nsubset (S-D1)\nn = {f(sd1.n)}', fs=6.0, minw=0.34)
vline(axA, MAINX, b2[2], b3[3])
letter(axA, 'a', 0.0, 0.97); axA.text(0.06, 0.985, 'Diagnostic cohort', fontsize=7.5, fontweight='bold', va='center')
# panel b
r0 = fc('SEER', 'Raw extraction'); dtc = fc('SEER', 'Histology: DTC')
hist_ex = sum(fc('SEER', f'Histology: {h}').n for h in ['ATC', 'MTC', 'Mixed medullary', 'Other'])
q = fc('SEER', 'DTC nodes: quantitative'); spc = fc('SEER', 'DTC nodes: special code 95-97')
ne = fc('SEER', 'DTC nodes: none examined'); un = fc('SEER', 'DTC nodes: unknown/NA')
st = fc('SEER', 'DTC quantitative nodal, survival time available'); cu = fc('SEER', 'Excluded (primary): unknown cause of death')
kc = fc('SEER', 'DTC nodal, known cause of death'); cc = fc('SEER', 'DTC nodal complete case (primary analytic cohort)')
assert spc.n == 675 and spc.events == 56, 'special-code DTC counts differ from frozen values'
c1 = rbox(axB, MAINX, 0.90, f'SEER thyroid cancer\nn = {f(r0.n)}', minw=0.40)
c2 = rbox(axB, MAINX, 0.66, f'DTC\nn = {f(dtc.n)}', minw=0.40)
c3 = rbox(axB, MAINX, 0.42, f'Quantitative nodal data\nn = {f(q.n)}', minw=0.40)
c4 = rbox(axB, MAINX, 0.13, f'Primary prognostic cohort\nn = {f(cc.n)}; CSS deaths {f(cc.events)}', fc=FILL, bold=True, minw=0.40)
vline(axB, MAINX, c1[2], c2[3]); vline(axB, MAINX, c2[2], c3[3]); vline(axB, MAINX, c3[2], c4[3])
x1 = rbox(axB, SIDEX, 0.78, f'Not DTC\nn = {f(hist_ex)}', fs=6.0, minw=0.30)
x2 = rbox(axB, SIDEX, 0.54, f'No nodes examined {f(ne.n)}\nNodal codes 95–97 {f(spc.n)} ({f(spc.events)} CSS deaths)\nNodal data unavailable {f(un.n)}',
          fs=6.0, minw=0.30, align='left')
x3 = rbox(axB, SIDEX, 0.275, f'Survival time missing {f(q.n - st.n)}\nUnknown cause of death {f(cu.n)}\nMissing T, N or M {f(kc.n - cc.n)}',
          fs=6.0, minw=0.30, align='left')
hline(axB, MAINX, x1[0], 0.78); hline(axB, MAINX, x2[0], 0.54); hline(axB, MAINX, x3[0], 0.275)
letter(axB, 'b', 0.0, 0.97); axB.text(0.06, 0.985, 'Prognostic cohort', fontsize=7.5, fontweight='bold', va='center')
save(fig, 'Fig1_flow')

# =====================================================================
# Figure 2 — diagnostic discrimination and calibration
oof = pd.read_csv(need(os.path.join(D2024, 'diag_oof_primary.csv')))
perf = pd.read_csv(need(os.path.join(D2024, 'diag_performance.csv')))
xl = pd.read_csv(need(os.path.join(a.data, 'thyroidxl_clean_v2.csv')))
oof = oof.merge(xl[['patient_id', 'tirads']], on='patient_id', how='left')
P = perf[perf.analysis.str.startswith('P:')].set_index('model')
y = oof.y.values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 80 * MM), gridspec_kw={'wspace': 0.32})
for m in DCOL:
    fpr, tpr, _ = roc_curve(y, oof[m]); ti = m.startswith('TIRADS')
    ax1.plot(fpr, tpr, color=DCOL[m], lw=0.9, ls='--' if ti else '-', zorder=3 if ti else 2,
             label=f'{DLAB[m]} ({P.loc[m, "AUC"]:.3f})')
op = (oof.tirads >= 4).astype(int); sens = (op[y == 1] == 1).mean(); spec = (op[y == 0] == 0).mean()
ax1.plot(1 - spec, sens, 'o', ms=3.6, mfc='white', mec='black', mew=0.7, zorder=5)
ax1.annotate('TI-RADS ≥4', xy=(1 - spec, sens), xytext=(1 - spec + 0.05, sens - 0.09), fontsize=6.2,
             arrowprops=dict(arrowstyle='-', lw=0.4, color='black'))
ax1.plot([0, 1], [0, 1], color='0.82', lw=0.5)
ax1.set_xlabel('1 − specificity'); ax1.set_ylabel('Sensitivity'); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.005)
ax1.legend(loc='lower right', frameon=False, handlelength=1.8, title='AUC', title_fontsize=6.3)
title(ax1, 'Discrimination', f'5-fold out-of-fold predictions; n = {f(len(y))}'); letter(ax1, 'a', -0.16, 1.06)
for m in DCOL:
    p = oof[m].values; ti = m.startswith('TIRADS')
    grp = oof.tirads.values if ti else pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False).values  # TI-RADS by category
    xs, ys, lo, hi = [], [], [], []
    for g in np.unique(grp):
        mm = grp == g; kk, nn = int(y[mm].sum()), int(mm.sum()); l, h = proportion_confint(kk, nn, method='wilson')
        xs.append(p[mm].mean()); ys.append(kk / nn); lo.append(max(kk / nn - l, 0)); hi.append(max(h - kk / nn, 0))
    ax2.errorbar(xs, ys, yerr=[lo, hi], fmt='s' if ti else 'o', ms=2.6 if ti else 2.4, mfc='white' if ti else DCOL[m],
                 mec=DCOL[m], mew=0.6, color=DCOL[m], ls='--' if ti else '-', lw=0.5, elinewidth=0.45, capsize=0, label=DLAB[m])
ax2.plot([0, 1], [0, 1], color='0.82', lw=0.5, zorder=0)
ax2.set_xlabel('Predicted malignancy probability'); ax2.set_ylabel('Observed malignancy proportion')
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.legend(loc='upper left', frameon=False, handlelength=1.8)
title(ax2, 'Calibration', ' '); letter(ax2, 'b', -0.16, 1.06)
save(fig, 'Fig2_diagnostic')

# =====================================================================
# Figure 3 — incremental discrimination (primary only) and 24-month calibration (2 x 2)
pdl = pd.read_csv(need(os.path.join(S2, 's2_paired_delta.csv')))
dec = pd.read_csv(need(os.path.join(S2, 'cal_deciles_primary.csv')))
U = pdl[pdl.metric == "Uno's C (tau = 24 m)"]
UP = U[U.variant == 'primary (unstratified)'].set_index('comparison')
fig = plt.figure(figsize=(W2, 92 * MM))
gs = fig.add_gridspec(2, 4, width_ratios=[1.45, 0.16, 1, 1], wspace=0.14, hspace=0.42)
ax = fig.add_subplot(gs[:, 0])
for i, c in enumerate(COMPS_MAIN):
    r = UP.loc[c]
    ax.errorbar(r.corrected_delta, -i, xerr=[[r.corrected_delta - r.CI_low], [r.CI_high - r.corrected_delta]], fmt='o', ms=3.2,
                color='black', elinewidth=0.7, capsize=0)
ax.axvline(0, color='0.6', lw=0.5, ls='--', zorder=0)
ax.axhline(2.5 * -1, color='0.88', lw=0.5, zorder=0)
ax.set_yticks([-i for i in range(len(COMPS_MAIN))]); ax.set_yticklabels([CLAB[c] for c in COMPS_MAIN])
ax.set_ylim(-len(COMPS_MAIN) + 0.4, 0.6); ax.set_xlim(-0.025, 0.025)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.set_xlabel("Difference in Uno's C")
title(ax, 'Incremental discrimination', "Optimism-corrected Uno's C at 24 months"); letter(ax, 'a', -0.55, 1.03)
FLOOR = 0.008; TK = [0.01, 0.1, 1, 10]
for j, m in enumerate(PCOL):
    axc = fig.add_subplot(gs[j // 2, 2 + j % 2])
    g = dec[dec.model == m].sort_values('decile')
    xs = g.mean_predicted.values * 100; ys = g.observed_KM.values * 100
    lo = np.clip(g.obs_low.values * 100, FLOOR, None); hi = g.obs_high.values * 100; pos = ys > 0
    axc.plot([FLOOR, 30], [FLOOR, 30], color='0.82', lw=0.5, zorder=0)
    axc.errorbar(xs[pos], ys[pos], yerr=[ys[pos] - np.minimum(lo[pos], ys[pos]), hi[pos] - ys[pos]], fmt='o', ms=2.4,
                 color=PCOL[m], elinewidth=0.45, capsize=0)
    axc.plot(xs[~pos], np.full((~pos).sum(), FLOOR), 'o', ms=2.6, mfc='white', mec=PCOL[m], mew=0.6, clip_on=False, zorder=4)
    axc.set_xscale('log'); axc.set_yscale('log'); axc.set_xlim(FLOOR, 30); axc.set_ylim(FLOOR, 30)
    axc.set_xticks(TK); axc.set_yticks(TK); lab = ['0.01', '0.1', '1', '10']
    axc.set_xticklabels(lab if j >= 2 else []); axc.set_yticklabels(lab if j % 2 == 0 else [])
    axc.minorticks_off(); axc.set_title(PLAB[m], fontsize=6.6, pad=2)
    if j == 0:
        letter(axc, 'b', -0.62, 1.12); axc.text(-0.48, 1.135, '24-month calibration', transform=axc.transAxes, fontsize=7.5, fontweight='bold', va='bottom')
        axc.set_ylabel('Observed 24-month CSS risk (%)', fontsize=7); axc.yaxis.set_label_coords(-0.42, -0.12)
fig.text(0.765, 0.005, 'Predicted 24-month risk (%)', ha='center', fontsize=7)
save(fig, 'Fig3_prognostic')

# =====================================================================
# Figure S1 — sensitivity forest (primary, post hoc histology-stratified, prespecified sensitivities)
sens = pd.read_csv(need(os.path.join(S2, 'sens_results.csv')))
SP = sens[sens.kind == 'paired']
UT = U[U.variant == 'post hoc: stratified by histology'].set_index('comparison')
ORDER = [('S-COD-A', 'Unknown COD as CSS'), ('S-COD-B', 'Unknown COD censored'), ('S-EX90', 'Examined = 90 excluded'),
         ('S-TREAT', 'Surgery + RAI added'), ('S-RCS', 'Age as spline'), ('S-D5', '≥5 nodes examined'),
         ('S-ALLHIST', 'All histologies'), ('S-RIDGE', 'Ridge')]
labels = ['Primary', 'Histology-stratified'] + [l for _, l in ORDER]
fig, axes = plt.subplots(1, 5, figsize=(W2, 72 * MM), sharey=True, gridspec_kw={'wspace': 0.12})
for ax, c in zip(axes, COMPS_MAIN):
    pts = [(UP.loc[c, 'corrected_delta'], UP.loc[c, 'CI_low'], UP.loc[c, 'CI_high'], 'D', 'black', 'black'),
           (UT.loc[c, 'corrected_delta'], UT.loc[c, 'CI_low'], UT.loc[c, 'CI_high'], 'D', 'black', 'white')]
    for key, _ in ORDER:
        s = SP[SP.analysis.str.startswith(key) & (SP.model == c)].iloc[0]
        pts.append((s.delta, s.CI_low, s.CI_high, 'o', GREY, GREY))
    for i, (d, lo, hi, mk, col, fcol) in enumerate(pts):
        yy = -i - (0.6 if i >= 2 else 0)
        ax.errorbar(d, yy, xerr=[[d - lo], [hi - d]], fmt=mk, ms=2.6, color=col, mfc=fcol, mec=col, mew=0.6, elinewidth=0.6, capsize=0)
    ax.axvline(0, color='0.6', lw=0.5, ls='--', zorder=0)
    ax.set_xlim(-0.03, 0.03); ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-0.02, 0, 0.02]))
    ax.set_title(CLAB[c], fontsize=6.6, pad=3); ax.tick_params(axis='y', length=0)
    if ax is not axes[0]: ax.spines['left'].set_visible(False)
axes[0].set_yticks([0, -1] + [-i - 0.6 for i in range(2, len(labels))]); axes[0].set_yticklabels(labels)
axes[0].spines['left'].set_visible(False)
fig.supxlabel("Difference in Uno's C (24 months)", fontsize=7, y=0.0)
save(fig, 'FigS1_sensitivity_forest')

# =====================================================================
# Figure S2 — decision curves
dca = pd.read_csv(need(os.path.join(S2, 'dca_24m_cv.csv')))
fig, ax = plt.subplots(figsize=(100 * MM, 68 * MM))
for m in PCOL:
    g = dca[dca.model == m]; ax.plot(g.threshold * 100, g.net_benefit * 1000, color=PCOL[m], lw=0.9, label=PLAB[m])
g = dca[dca.model == 'M0 core']
ax.plot(g.threshold * 100, g.nb_treat_all * 1000, color='0.55', ls='--', lw=0.8, label='Treat all')
ax.axhline(0, color='black', lw=0.5, label='Treat none')
ax.set_ylim(-1, 6); ax.set_xlim(0, 10)
ax.set_xlabel('Threshold 24-month CSS risk (%)'); ax.set_ylabel('Net benefit per 1,000 patients')
ax.legend(frameon=False, loc='upper right', handlelength=1.8)
title(ax, 'Decision-curve analysis at 24 months')
save(fig, 'FigS2_dca')

# =====================================================================
# Figure S3 — independent within-cohort age gradients
bs = pd.read_csv(need(os.path.join(BR, 'bridge_strata.csv'))); sl = pd.read_csv(need(os.path.join(BR, 'bridge_slopes.csv'))).set_index('analysis')
STR = [t.replace('>=', '≥').replace('-', '–') for t in bs.stratum]; xs = np.arange(len(bs))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 62 * MM), gridspec_kw={'wspace': 0.38})
lo = bs.xl_ci.str.split('-').str[0].astype(float); hi = bs.xl_ci.str.split('-').str[1].astype(float)
ax1.errorbar(xs, bs.xl_pct, yerr=[bs.xl_pct - lo, hi - bs.xl_pct], fmt='o', ms=3, color='black', elinewidth=0.6, capsize=0)
ax1.set_xticks(xs); ax1.set_xticklabels(STR); ax1.set_ylim(0, 50); ax1.set_xlim(-0.5, len(xs) - 0.5)
ax1.set_xlabel('Age (years)'); ax1.set_ylabel('Malignant nodules (%)')
o = sl.loc['ThyroidXL full cohort (primary)']
ax1.text(0.98, 0.95, f'OR per 10 y = {o.estimate:.2f}', transform=ax1.transAxes, ha='right', va='top', fontsize=6.3, color=GREY)
title(ax1, 'Diagnostic cohort'); letter(ax1, 'a', -0.2, 1.04)
lo = bs.seer_dtc_cif24_ci.str.split('-').str[0].astype(float); hi = bs.seer_dtc_cif24_ci.str.split('-').str[1].astype(float)
ax2.errorbar(xs, bs.seer_dtc_cif24_pct, yerr=[bs.seer_dtc_cif24_pct - lo, hi - bs.seer_dtc_cif24_pct], fmt='o', ms=3, color='black',
             elinewidth=0.6, capsize=0)
ax2.set_xticks(xs); ax2.set_xticklabels(STR); ax2.set_ylim(0, 6.5); ax2.set_xlim(-0.5, len(xs) - 0.5)
ax2.set_xlabel('Age (years)'); ax2.set_ylabel('24-month CSS cumulative incidence (%)')
o = sl.loc['SEER DTC (primary)']
ax2.text(0.04, 0.95, f'HR per 10 y = {o.estimate:.2f}', transform=ax2.transAxes, ha='left', va='top', fontsize=6.3, color=GREY)
title(ax2, 'Prognostic cohort'); letter(ax2, 'b', -0.2, 1.04)
save(fig, 'FigS3_age_gradient')
print('All figures written to', os.path.abspath(OUT))
