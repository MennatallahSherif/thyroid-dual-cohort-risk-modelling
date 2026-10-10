"""
Calibration and decision-curve analysis at a fixed 24-month horizon (Option B).
Replaces Sections E (calibration) and F (DCA) of step2_lnr_lodds.py, which (i) used a superseded dual-age model
in the submitted figure, (ii) labelled a 24-month analysis as 59-month, and (iii) scored 24-month predicted risk
against events at any time during follow-up.

Cohort : SEER nodal cohort (valid LNR/LODDS, known survival; n = 33,457)
Models : ridge-penalised Cox (lifelines penalizer = 0.1), same specification as Table 2:
         N stage (7 df) + age_mid ; LNR + age_mid ; LODDS + age_mid.  No age_55_plus.
Predicted risk  = 1 - S(24 months | x) from the fitted model (apparent, i.e. fitted and evaluated on the same patients).
Calibration     : deciles of predicted risk; observed risk = 1 - Kaplan-Meier(24 months) in each decile (95% CI from the
                  Kaplan-Meier log-log interval); calibration-in-the-large; Cox calibration slope (unpenalised Cox on the linear predictor).
DCA             : censoring-aware net benefit at 24 months (Vickers et al., BMC Med Inform Decis Mak 2008;8:53):
                  NB(pt) = [1 - S_pos(24)] * P(pos) - S_pos(24) * P(pos) * pt/(1-pt),
                  where pos = predicted 24-month risk >= pt, S_pos(24) = Kaplan-Meier survival at 24 months among pos,
                  P(pos) = proportion of patients with predicted risk >= pt. Treat-all uses the overall Kaplan-Meier risk.
                  Non-cancer deaths are censored (cause-specific endpoint), as in all CSS models.
"""
import numpy as np, pandas as pd, warnings, os
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter

T = 24; MIN_POS = 100
FIG = '../outputs/figures/'; TAB = '../outputs/tables/'
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
s = pd.read_csv('../data/processed/seer_clean.csv')
imp = s.NPLN.notna() & s.D.notna() & (s.NPLN > s.D); s.loc[imp, ['LNR','LODDS','NPLN','D']] = np.nan
node = s[s.LNR.notna() & s.LODDS.notna() & s.survival_months.notna() & s.css_event.notna() & s.age_mid.notna()].copy()
dm = pd.get_dummies(node.n_stage, prefix='n', drop_first=True).astype(float); node = pd.concat([node, dm], axis=1); dcols = list(dm.columns)
print('nodal cohort n =', len(node), '| CSS events (any time) =', int(node.css_event.sum()))
MODELS = {'N stage (7 df) + age': ['age_mid'] + dcols, 'LNR + age': ['LNR','age_mid'], 'LODDS + age': ['LODDS','age_mid']}

def km_risk(d, t=T):
    k = KaplanMeierFitter().fit(d.survival_months, d.css_event)
    return 1 - float(k.predict(t)), k
fitted = {}
for name, preds in MODELS.items():
    d = node[preds + ['survival_months','css_event']]
    m = CoxPHFitter(penalizer=0.1).fit(d, 'survival_months', 'css_event')
    pr = 1 - m.predict_survival_function(d, times=[T]).iloc[0].values
    lp = m.predict_log_partial_hazard(d).values
    fitted[name] = (m, pr, lp)

overall_obs, _ = km_risk(node)
print('overall Kaplan-Meier CSS risk at 24 months = %.4f' % overall_obs)

# ---------------- calibration
rows, summ = [], []
for name, (m, pr, lp) in fitted.items():
    dec = pd.qcut(pr, 10, duplicates='drop')
    for i, g in enumerate(sorted(set(dec))):
        mk = np.array(dec == g); d = node[mk]
        obs, k = km_risk(d)
        ci = k.confidence_interval_survival_function_.loc[:T].iloc[-1].values   # lower, upper survival at 24 months
        rows.append(dict(Model=name, Decile=i+1, n=int(mk.sum()), CSS_events_any_time=int(d.css_event.sum()),
                         Mean_predicted_24m=round(float(pr[mk].mean()), 4), Observed_KM_24m=round(obs, 4),
                         Obs_CI_low=round(1-float(ci[1]), 4), Obs_CI_high=round(1-float(ci[0]), 4)))
    cal = node[['survival_months','css_event']].copy(); cal['lp'] = lp - lp.mean()
    sl = CoxPHFitter().fit(cal, 'survival_months', 'css_event')
    summ.append(dict(Model=name, n=len(node), Mean_predicted_24m=round(float(pr.mean()), 4), Observed_KM_24m=round(overall_obs, 4),
                     Max_predicted_24m=round(float(pr.max()), 4),
                     Calibration_slope=round(float(sl.params_['lp']), 3),
                     Slope_95CI='%.3f-%.3f' % tuple(sl.confidence_intervals_.loc['lp'].values)))
cal_df = pd.DataFrame(rows); sum_df = pd.DataFrame(summ)
cal_df.to_csv(TAB + 'Table_calibration_24m_deciles.csv', index=False); sum_df.to_csv(TAB + 'Table_calibration_24m_summary.csv', index=False)
print(sum_df.to_string(index=False)); print(cal_df[cal_df.Model != 'N stage (7 df) + age'].to_string(index=False))

COL = {'LNR + age': '#2563EB', 'LODDS + age': '#0D9488', 'N stage (7 df) + age': '#EA580C'}
def panel(ax, name, title):
    c = cal_df[cal_df.Model == name]
    ax.errorbar(c.Mean_predicted_24m, c.Observed_KM_24m, yerr=[c.Observed_KM_24m - c.Obs_CI_low, c.Obs_CI_high - c.Observed_KM_24m],
                fmt='o', color=COL[name], capsize=2, label='Deciles of predicted risk (95% CI)')
    lim = max(c.Obs_CI_high.max(), c.Mean_predicted_24m.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='Perfect calibration')
    ax.set_xlabel('Predicted 24-month cancer-specific risk'); ax.set_ylabel('Observed 24-month risk (Kaplan-Meier)')
    ax.set_title(title); ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=.3)
    for sp in ('top','right'): ax.spines[sp].set_visible(False)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
panel(axes[0], 'LNR + age', 'LNR + age (ridge-penalised Cox)'); panel(axes[1], 'LODDS + age', 'LODDS + age (ridge-penalised Cox)')
fig.suptitle('Calibration at 24 months, SEER nodal cohort (n = %s; apparent)' % format(len(node), ','), y=1.02)
plt.tight_layout(); plt.savefig(FIG + 'fig_calibration_lnr_lodds_24m.png', dpi=200, bbox_inches='tight'); plt.savefig(FIG + 'fig_calibration_lnr_lodds.png', dpi=200, bbox_inches='tight'); plt.close()  # legacy name overwritten
for name, fn, t in [('LNR + age', 'FigS3_calibration_lnr_24m.png', 'LNR + age'), ('LODDS + age', 'FigS4_calibration_lodds_24m.png', 'LODDS + age')]:
    f, a = plt.subplots(figsize=(6, 5)); panel(a, name, 'Calibration at 24 months: ' + t); plt.tight_layout(); plt.savefig(FIG + fn, dpi=200, bbox_inches='tight'); plt.close()

# ---------------- censoring-aware DCA at 24 months
thr = np.round(np.arange(0.002, 0.0601, 0.002), 4)
overall_obs_nb = lambda pt: overall_obs - (1 - overall_obs) * pt / (1 - pt)
out = []
for name, (m, pr, lp) in fitted.items():
    for pt in thr:
        pos = pr >= pt; n_pos = int(pos.sum())
        if n_pos < MIN_POS: nb = np.nan; risk_pos = np.nan
        else:
            risk_pos, _ = km_risk(node[pos]); P = pos.mean()
            nb = risk_pos * P - (1 - risk_pos) * P * pt / (1 - pt)
        out.append(dict(Model=name, Threshold=pt, N_positive=n_pos, Pct_positive=round(100 * n_pos / len(node), 2),
                        KM_risk_among_positive=(np.nan if np.isnan(nb) else round(risk_pos, 4)), Net_benefit=(np.nan if np.isnan(nb) else round(nb, 5)),
                        NB_treat_all=round(overall_obs_nb(pt), 5)))
dca = pd.DataFrame(out); dca.to_csv(TAB + 'Table_DCA_24m.csv', index=False)
fig, ax = plt.subplots(figsize=(9, 6))
for name in MODELS:
    dd = dca[dca.Model == name]; ax.plot(dd.Threshold, dd.Net_benefit, color=COL[name], lw=2, label=name)
ax.plot(thr, [overall_obs_nb(p) for p in thr], 'k-.', lw=1.5, alpha=.6, label='Treat all'); ax.axhline(0, color='gray', ls=':', label='Treat none')
ax.set_xlabel('Threshold probability (24-month cancer-specific risk)'); ax.set_ylabel('Net benefit (Kaplan-Meier, censoring-aware)')
ax.set_title('Decision curve analysis at 24 months (apparent, n = %s)' % format(len(node), ',')); ax.set_xlim(0, 0.06); ax.set_ylim(-0.02, 0.02)
ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=.3)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
plt.tight_layout(); plt.savefig(FIG + 'fig_dca_lnr_lodds_24m.png', dpi=200, bbox_inches='tight'); plt.savefig(FIG + 'fig_dca_lnr_lodds.png', dpi=200, bbox_inches='tight'); plt.close()  # legacy name overwritten
pd.set_option('display.width', 200)
print(dca[dca.Threshold.isin([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])].to_string(index=False))
print('max predicted 24m risk:', {n: round(float(f[1].max()), 4) for n, f in fitted.items()})
# reference only: the superseded approach (24-month risk scored against events at any time)
for name, (m, pr, lp) in fitted.items():
    y = node.css_event.values; nb = []
    for pt in [0.01, 0.02, 0.03]:
        tp = ((pr >= pt) & (y == 1)).sum() / len(y); fp = ((pr >= pt) & (y == 0)).sum() / len(y); nb.append(round(tp - fp * pt / (1 - pt), 5))
    print('SUPERSEDED any-time NB at 1/2/3%%  %-22s %s' % (name, nb))
