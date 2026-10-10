"""Ridge-penalised (lifelines penalizer=0.1, as used in step2/step3 outputs) vs unpenalised Cox HRs.
Reference/sensitivity output only. Reads the post-fix seer_clean.csv."""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
P = sys.argv[1] if len(sys.argv)>1 else '../data/processed/seer_clean.csv'
s = pd.read_csv(P)
imp = s.NPLN.notna() & s.D.notna() & (s.NPLN > s.D); s.loc[imp, ['LNR','LODDS','NPLN','D']] = np.nan
ordm = {'N0':0,'N0a':0,'N0b':0,'N1a':1,'N1b':2}; s['n_ordinal'] = s.n_stage.map(ordm)
node = s[s.LNR.notna() & s.LODDS.notna() & s.survival_months.notna()].copy()
nd = pd.get_dummies(node.n_stage, prefix='n', drop_first=True).astype(float); node = pd.concat([node, nd], axis=1); ncols = list(nd.columns)
age = s[s.survival_months.notna() & s.age_mid.notna()].copy()
rows = []
def fit(df, preds, pen):
    d = df[preds+['survival_months','css_event']].dropna()
    try:
        c = CoxPHFitter(penalizer=pen).fit(d, 'survival_months', 'css_event')
    except Exception:
        # sparse reference category (e.g. N0: 3 events) can make the unpenalised fit singular -> tiny ridge, flagged in output
        print('FALLBACK penalizer=1e-4 for', preds, flush=True)
        c = CoxPHFitter(penalizer=1e-4).fit(d, 'survival_months', 'css_event')
    ci = concordance_index(d.survival_months, -c.predict_partial_hazard(d), d.css_event)
    return c, ci, len(d), int(d.css_event.sum())
def add(table, label, df, preds, per=None):
    r, cr, n, ev = fit(df, preds, 0.1); u, cu, _, _ = fit(df, preds, 0.0)
    for p in preds:
        rows.append(dict(Table=table, Model=label, Predictor=p, n=n, CSS_events=ev,
          HR_ridge=round(float(np.exp(r.params_[p])),4),
          HR_unpenalised=round(float(np.exp(u.params_[p])),4),
          HR_unpen_CI=f"{np.exp(u.confidence_intervals_.loc[p].iloc[0]):.4g}-{np.exp(u.confidence_intervals_.loc[p].iloc[1]):.4g}",
          Cindex_ridge=round(cr,4), Cindex_unpenalised=round(cu,4)))
# Table 2 family (nodal models, CSS)
add('Table 2','N stage full (7 df) + age', node, ['age_mid']+ncols)
add('Table 2','Ordinal N-stage (1 df) + age', node, ['n_ordinal','age_mid'])
add('Table 2','LNR + age', node, ['LNR','age_mid'])
add('Table 2','LODDS + age', node, ['LODDS','age_mid'])
add('Table 2','LNR + LODDS + age', node, ['LNR','LODDS','age_mid'])
add('Table 2','LNR + LODDS + age (PTC only)', node[node.histology_subtype=='PTC'], ['LNR','LODDS','age_mid'])
for g in ['N1a','N1b']:
    add('Table 2',f'LNR + LODDS + age ({g})', node[node.n_stage==g], ['LNR','LODDS','age_mid'])
# Table 3 family (age-threshold models, CSS) - specs verified to reproduce Table 3 C-index/HR exactly
add('Table 3','Binary age (55yr cutoff)', age, ['age_55_plus'])
add('Table 3','Continuous age (linear)', age, ['age_mid'])
for h in ['PTC','FTC','MTC']:
    add('Table 3',f'{h} binary age', age[age.histology_subtype==h], ['age_55_plus'])
    add('Table 3',f'{h} continuous age', age[age.histology_subtype==h], ['age_mid'])
# Common-sample comparison (patients with a rankable ordinal N category) - C-index only
com = node[node.n_ordinal.notna()].copy()
nz = [c for c in ncols if com[c].sum() > 0]
common = []
for lab, preds in [('Ordinal N-stage (1 df) + age', ['n_ordinal','age_mid']), ('N stage dummies + age', ['age_mid']+nz),
                   ('LNR + age', ['LNR','age_mid']), ('LODDS + age', ['LODDS','age_mid']), ('LNR + LODDS + age', ['LNR','LODDS','age_mid'])]:
    _, cr, n, ev = fit(com, preds, 0.1); _, cu, _, _ = fit(com, preds, 0.0)
    common.append(dict(Model=lab, n=n, CSS_events=ev, Cindex_ridge=round(cr,4), Cindex_unpenalised=round(cu,4)))
pd.DataFrame(common).to_csv(P.replace('seer_clean.csv','')+'../../outputs/tables/Table_common_sample_cindex.csv' if False else '../outputs/tables/Table_common_sample_cindex.csv', index=False)
print(pd.DataFrame(common).to_string())
out = pd.DataFrame(rows); out.to_csv('../outputs/tables/Table_HR_ridge_vs_unpenalised.csv', index=False)
pd.set_option('display.width',250); print(out.to_string())
