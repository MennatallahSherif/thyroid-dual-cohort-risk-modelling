"""
Assemble stage-2 tables (Plan v1.1) from step2v2_prognostic_stage2.py outputs.
Usage: python step2v2_stage2_report.py <results_dir>
Outputs: s2_performance.csv (apparent, optimism, corrected), s2_paired_delta.csv, s2_calibration.csv, s2_calibration_plot.png
"""
import os, sys, json
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

R = sys.argv[1]
app = json.load(open(os.path.join(R, 'main_apparent.json')))
bt = pd.read_csv(os.path.join(R, 'main_boot.csv'))
B = bt.b.nunique()
MET = {'uno24': "Uno's C (tau = 24 m)", 'harrell': "Harrell's C", 'td12': 'tdAUC 12 m', 'td24': 'tdAUC 24 m',
       'slope': 'Calibration slope (24 m)', 'OE': 'O:E (24 m)'}
perf = []
for (v, m), g in bt.groupby(['variant', 'model']):
    for q, lab in MET.items():
        opt = (g[f'boot_{q}'] - g[f'orig_{q}']).mean()
        a = app[v][m][q]
        perf.append(dict(variant=v, model=m, metric=lab, apparent=a, optimism=opt, corrected=a - opt, B=len(g)))
perf = pd.DataFrame(perf)
perf.to_csv(os.path.join(R, 's2_performance.csv'), index=False)

PAIRS = [('M2 core + LNR', 'M1 core + N', 'primary'), ('M3 core + LODDS', 'M1 core + N', 'primary'),
         ('M1 core + N', 'M0 core', 'secondary'), ('M2 core + LNR', 'M0 core', 'secondary'), ('M3 core + LODDS', 'M0 core', 'secondary')]
rows = []
for v, gv in bt.groupby('variant'):
    for q in ['uno24', 'td24', 'harrell']:
        W = gv.pivot(index='b', columns='model', values=f'boot_{q}'); O = gv.pivot(index='b', columns='model', values=f'orig_{q}')
        for a, b_, role in PAIRS:
            d_app = app[v][a][q] - app[v][b_][q]
            d_boot = W[a] - W[b_]; d_orig = O[a] - O[b_]
            opt = (d_boot - d_orig).mean(); corr = d_app - opt
            shifted = d_boot - opt
            p = max(2 * min((shifted <= 0).mean(), (shifted >= 0).mean()), 1 / len(shifted))
            rows.append(dict(variant=v, metric=MET[q], comparison=f'{a} - {b_}', role=role, apparent_delta=d_app, optimism=opt,
                             corrected_delta=corr, CI_low=np.percentile(shifted, 2.5), CI_high=np.percentile(shifted, 97.5), p_boot=p))
pdl = pd.DataFrame(rows)
pdl['p_holm_primary'] = np.nan
for (v, met), g in pdl[pdl.role == 'primary'].groupby(['variant', 'metric']):
    pdl.loc[g.index, 'p_holm_primary'] = multipletests(g.p_boot, method='holm')[1]
def interp(r):
    if r.CI_low > 0: return 'higher'
    if r.CI_high < 0: return 'lower'
    return 'comparable / numerically ' + ('higher' if r.corrected_delta > 0 else 'lower')
pdl['interpretation'] = pdl.apply(interp, axis=1)
pdl.to_csv(os.path.join(R, 's2_paired_delta.csv'), index=False)

cal = perf[perf.metric.isin(['Calibration slope (24 m)', 'O:E (24 m)'])].pivot_table(index=['variant', 'model'], columns='metric', values=['apparent', 'corrected'])
cal.columns = [f'{a} {b}' for a, b in cal.columns]; cal = cal.reset_index()
cal['DCA eligible (corrected slope 0.8-1.2)'] = cal['corrected Calibration slope (24 m)'].between(0.8, 1.2)
cal.to_csv(os.path.join(R, 's2_calibration.csv'), index=False)

# calibration plot (apparent deciles; log axes because risk concentrates in the top decile)
dec = pd.read_csv(os.path.join(R, 'cal_deciles_primary.csv'))
fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
for ax, (m, g) in zip(axes, dec.groupby('model', sort=False)):
    x = g.mean_predicted * 100; y = g.observed_KM * 100
    lo = (g.observed_KM - g.obs_low).clip(lower=0) * 100; hi = (g.obs_high - g.observed_KM) * 100
    ax.errorbar(x, y.clip(lower=0.005), yerr=[np.minimum(lo, y - 0.005).clip(lower=0), hi], fmt='o', ms=4, capsize=2)
    ax.plot([0.005, 20], [0.005, 20], 'k--', lw=0.8); ax.set_xscale('log'); ax.set_yscale('log')
    c = cal[(cal.model == m) & (cal.variant == 'primary (unstratified)')].iloc[0]
    ax.set_title(f"{m}\nslope {c['apparent Calibration slope (24 m)']:.2f} (corrected {c['corrected Calibration slope (24 m)']:.2f}); "
                 f"O:E {c['apparent O:E (24 m)']:.2f}", fontsize=9)
    ax.set_xlabel('Mean predicted 24-month CSS risk (%)')
axes[0].set_ylabel('Observed 24-month CSS risk, KM (%)')
plt.tight_layout(); plt.savefig(os.path.join(R, 's2_calibration_plot.png'), dpi=200); plt.close()
pd.set_option('display.width', 250)
print(perf[perf.metric.isin(["Uno's C (tau = 24 m)", 'Calibration slope (24 m)'])].round(4).to_string(index=False))
print(pdl[pdl.metric == "Uno's C (tau = 24 m)"].round(4).to_string(index=False))
