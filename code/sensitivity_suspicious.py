"""Sensitivity analysis: preoperative models with the 94 'suspicious' ThyroidXL cases (conclusion = 4) EXCLUDED.
In the primary analysis these cases are retained and coded non-malignant (as in the original pipeline).
Re-uses the feature sets, cross-validation and bootstrap code of step4 (executed up to Section C).
Output: outputs/tables/TableS15_sensitivity_excl_suspicious.csv"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
exec(open('step4_preop_model.py', encoding='utf-8').read().split("# SECTION C ")[0])
rows = []
def evaluate(df, feats, label, mask=None):
    X = df[feats].values; y = df.malignant.values; t = df.tirads.values
    oof = out_of_fold_predictions(X, y, t)
    sel = np.ones(len(df), bool) if mask is None else mask(df)
    for name in ['TIRADS alone (baseline)'] + MODEL_NAMES:
        auc = roc_auc_score(y[sel], oof[name][sel]); lo, hi = bootstrap_auc_ci(y[sel], oof[name][sel])
        rows.append({'Analysis': label, 'n': int(sel.sum()), 'Malignant': int(y[sel].sum()), 'Prevalence (%)': round(y[sel].mean() * 100, 1),
                     'Model': name, 'AUC': round(auc, 3), '95% CI': f'{lo:.3f}-{hi:.3f}'})
# df_A / df_B lose the original index; recover the 'conclusion' code of the retained rows from the cleaned file
_cA = xl.loc[xl[FEATURES_A + ['malignant']].notna().all(axis=1), 'conclusion'].values
_cB = xl.loc[xl[FEATURES_B + ['malignant']].notna().all(axis=1), 'conclusion'].values
assert len(_cA) == len(df_A) and len(_cB) == len(df_B)
A = df_A[_cA != 4].reset_index(drop=True); B = df_B[_cB != 4].reset_index(drop=True)
evaluate(A, FEATURES_A, 'Set A (without FNAC), suspicious cases excluded')
evaluate(B, FEATURES_B, 'Set B (with FNAC), suspicious cases excluded')
evaluate(B, FEATURES_B, 'Bethesda III-V subgroup, suspicious cases excluded', mask=lambda d: d.fnac.isin([3, 4, 5]).values)
out = pd.DataFrame(rows); out.to_csv(os.path.join(TABS, 'TableS15_sensitivity_excl_suspicious.csv'), index=False)
pd.set_option('display.width', 200); print(out.to_string(index=False))
