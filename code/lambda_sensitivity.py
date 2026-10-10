"""Table S14: sensitivity of the apparent C-index to the ridge penalty (lifelines penalizer = 0, 0.01, 0.1, 1.0) in the nodal cohort.
Output: outputs/tables/TableS14_ridge_lambda_sensitivity.csv"""
import numpy as np, pandas as pd, warnings, os
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
s = pd.read_csv('../data/processed/seer_clean.csv')
imp = s.NPLN.notna() & s.D.notna() & (s.NPLN > s.D); s.loc[imp, ['LNR', 'LODDS', 'NPLN', 'D']] = np.nan
s['n_ordinal'] = s.n_stage.map({'N0': 0, 'N0a': 0, 'N0b': 0, 'N1a': 1, 'N1b': 2})
nd = s[s.LNR.notna() & s.LODDS.notna() & s.survival_months.notna()].copy()
dm = pd.get_dummies(nd.n_stage, prefix='n', drop_first=True).astype(float); nd = pd.concat([nd, dm], axis=1)
def cidx(df, preds, pen):
    d = df[preds + ['survival_months', 'css_event']].dropna()
    try: c = CoxPHFitter(penalizer=pen).fit(d, 'survival_months', 'css_event')
    except Exception: c = CoxPHFitter(penalizer=1e-4).fit(d, 'survival_months', 'css_event')   # singular unpenalised fit (sparse N0/N0b)
    return concordance_index(d.survival_months, -c.predict_partial_hazard(d), d.css_event)
rows = []
for lab, pr in [('Ordinal N + age (n = 32,316)', ['n_ordinal', 'age_mid']), ('N stage dummy-coded (7 df) + age', ['age_mid'] + list(dm.columns)),
                ('LNR + age', ['LNR', 'age_mid']), ('LODDS + age', ['LODDS', 'age_mid']), ('LNR + LODDS + age', ['LNR', 'LODDS', 'age_mid'])]:
    r = {pen: cidx(nd, pr, pen) for pen in [0.0, 0.01, 0.1, 1.0]}
    rows.append({'Model': lab, 'C-index lambda=0 (unpenalised)': round(r[0.0], 4), 'lambda=0.01': round(r[0.01], 4), 'lambda=0.1 (primary)': round(r[0.1], 4),
                 'lambda=1.0': round(r[1.0], 4), 'Range across 0.01-1.0': round(max(r[0.01], r[0.1], r[1.0]) - min(r[0.01], r[0.1], r[1.0]), 4)})
out = pd.DataFrame(rows); out.to_csv('../outputs/tables/TableS14_ridge_lambda_sensitivity.csv', index=False); print(out.to_string(index=False))
