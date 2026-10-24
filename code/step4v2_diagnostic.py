"""
STEP 4 v2 — Preoperative diagnostic modelling (Analysis Plan v1.1, Section 2).
Usage  : python step4v2_diagnostic.py <processed_dir> <results_dir>
Input  : thyroidxl_clean_v2.csv (from step1v2_cleaning.py)
Outputs: diag_performance.csv, diag_paired_delta.csv, diag_calibration.csv, diag_thresholds.csv,
         diag_setB_descriptive.csv, diag_bethesda3.csv, diag_oof_primary.csv, diag_log.txt

Analyses
  P   Primary   : Set A (9 features), outcome = histopathology-over-FNAC hierarchy (1,231 malignant)
  S1  Sensitivity: released labels (1,229 malignant)
  S2  Sensitivity: v1.0.1 feature set (adds age_55_plus)
  SD1 Supportive : histopathology-confirmed subset (796; outcome from histopathology text), CV re-run inside the subset
  Set B          : Supplementary; full cohort (upper bound, incorporation bias) and S-D1 (descriptive only: 88 benign < 100)
  Bethesda III   : exploratory, Set A OOF predictions evaluated inside Bethesda III; no between-model claims
Validation: stratified 5-fold CV (seed 42), scaler and TIRADS empirical rates fitted in training folds only,
            fixed prespecified hyperparameters, identical folds for all models and TIRADS.
Inference : 2,000 patient-level bootstrap resamples of OOF predictions, identical across models;
            paired ΔAUC percentile CI; DeLong paired test (confirmatory); Holm across the 3 co-primary comparisons.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import xgboost as xgb
warnings.filterwarnings('ignore')

IN = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else '.'
os.makedirs(OUT, exist_ok=True)
SEED, N_BOOT = 42, 2000
LOG = []
def log(m=''):
    print(m); LOG.append(str(m))

xl = pd.read_csv(os.path.join(IN, 'thyroidxl_clean_v2.csv'))
A = xl[xl.in_setA == 1].reset_index(drop=True)
A['age_55_plus'] = (A.age >= 55).astype(int)
FEAT = ['tirads', 'age', 'gender', 'width', 'height', 'nodule_area', 'aspect_ratio', 'pos_isthmus', 'pos_left']
FEAT_OLD = FEAT[:2] + ['gender', 'width', 'height', 'nodule_area', 'aspect_ratio', 'age_55_plus', 'pos_isthmus', 'pos_left']
FEAT_B = FEAT + ['fnac']
MODELS = ['Logistic Regression', 'Random Forest', 'XGBoost']
REF = 'TIRADS (empirical, fold-wise)'

def make_models():
    return {'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=20,
                                                    class_weight='balanced', random_state=SEED, n_jobs=1),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, subsample=0.8,
                                         eval_metric='logloss', random_state=SEED, verbosity=0)}

def oof_predictions(X, y, tirads):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {m: np.zeros(len(y)) for m in MODELS + [REF]}
    for tr, va in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        for name, mdl in make_models().items():
            mdl.fit(sc.transform(X[tr]), y[tr])
            oof[name][va] = mdl.predict_proba(sc.transform(X[va]))[:, 1]
        rates = pd.Series(y[tr]).groupby(tirads[tr]).mean()
        oof[REF][va] = pd.Series(tirads[va]).map(rates).fillna(y[tr].mean()).values
    return oof

# ---- DeLong (fast, Sun & Xu 2014)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T; return out
def delong_paired(y, p1, p2):
    pos, neg = y == 1, y == 0; m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = _midrank(p[pos]), _midrank(p[neg]), _midrank(np.r_[p[pos], p[neg]])
        aucs.append(tz[:m].sum() / m / n - (m + 1) / 2 / n)
        v10.append((tz[:m] - tx) / n); v01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(v10)) / m + np.cov(np.vstack(v01)) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0] - aucs[1], 2 * stats.norm.sf(abs(z))

def calib(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = np.log(p / (1 - p))
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
    return citl, slope, brier_score_loss(y, p), p.mean(), y.mean()

def boot_idx(y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed); out = []
    while len(out) < n_boot:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i): out.append(i)
    return out

PERF, DELTA, CAL = [], [], []
def evaluate(label, y, oof, inferential=True, role=''):
    idx = boot_idx(y) if inferential else []
    auc = {k: roc_auc_score(y, v) for k, v in oof.items()}
    bs = {k: np.array([roc_auc_score(y[i], oof[k][i]) for i in idx]) for k in oof} if inferential else {}
    for k in [REF] + MODELS:
        r = dict(analysis=label, role=role, model=k, n=len(y), malignant=int(y.sum()), benign=int((y == 0).sum()), AUC=auc[k])
        if inferential:
            r['AUC_CI'] = f'{np.percentile(bs[k], 2.5):.3f}-{np.percentile(bs[k], 97.5):.3f}'
        PERF.append(r)
        if inferential:
            citl, slope, brier, mp, obs = calib(y, oof[k])
            CAL.append(dict(analysis=label, model=k, calibration_intercept=citl, calibration_slope=slope, brier=brier, mean_predicted=mp, observed=obs))
    if inferential:
        rows = []
        for k in MODELS:
            d = bs[k] - bs[REF]; dl, p = delong_paired(y, oof[k], oof[REF])
            rows.append(dict(analysis=label, comparison=f'{k} - TIRADS', delta_AUC=auc[k] - auc[REF],
                             CI_low=np.percentile(d, 2.5), CI_high=np.percentile(d, 97.5),
                             boot_prop_le0=(d <= 0).mean(), delong_p=p))
        holm = multipletests([r['delong_p'] for r in rows], method='holm')[1]
        for r, h in zip(rows, holm): r['delong_p_holm'] = h
        DELTA.extend(rows)

def run(label, df, feats, ycol, role, inferential=True):
    d = df[df[feats + [ycol]].notna().all(axis=1)].reset_index(drop=True)
    y = d[ycol].astype(int).values
    oof = oof_predictions(d[feats].values.astype(float), y, d.tirads.values)
    evaluate(label, y, oof, inferential, role)
    log(f'{label}: n={len(y)}, malignant={y.sum()}  ' + '  '.join(f'{k.split()[0]}={roc_auc_score(y, v):.4f}' for k, v in oof.items()))
    return d, y, oof

# ---------------- analyses
dP, yP, oofP = run('P: Set A, hierarchy outcome', A, FEAT, 'malignant', 'primary')
run('S1: Set A, released labels', A, FEAT, 'malignant_released', 'sensitivity')
run('S2: Set A, v1.0.1 features (+age_55_plus)', A, FEAT_OLD, 'malignant', 'sensitivity')
SD1 = A[A.sd1 == 1].reset_index(drop=True)
run('SD1: histopathology-confirmed, Set A', SD1, FEAT, 'sd1_malignant', 'supportive')
run('Set B: full cohort (upper bound, incorporation bias)', A, FEAT_B, 'malignant', 'supplementary')
SB = SD1[SD1.in_setB == 1].reset_index(drop=True)
nb = int((SB.sd1_malignant == 0).sum())
run('Set B within S-D1 (descriptive only)', SD1, FEAT_B, 'sd1_malignant', 'descriptive', inferential=(nb >= 100 and SB.sd1_malignant.sum() >= 100))

# Bethesda III exploratory (primary OOF predictions, evaluated in subgroup)
b3 = (dP.fnac == 3).values
beth = []
for k in [REF] + MODELS:
    y3, p3 = yP[b3], oofP[k][b3]
    beth.append(dict(model=k, n=int(b3.sum()), malignant=int(y3.sum()), AUC=roc_auc_score(y3, p3)))
pd.DataFrame(beth).to_csv(os.path.join(OUT, 'diag_bethesda3.csv'), index=False)

# ---------------- thresholds (primary)
def tm(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    def w(a, b): lo, hi = proportion_confint(a, a + b, method='wilson'); return f'{a/(a+b):.3f} ({lo:.3f}-{hi:.3f})' if a + b else 'NA'
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, Sensitivity=w(tp, fn), Specificity=w(tn, fp), PPV=w(tp, fp), NPV=w(tn, fn))
TH = [dict(rule='TIRADS >= 4', **tm(yP, (dP.tirads.values >= 4).astype(int)))]
target = (((dP.tirads.values >= 4) & (yP == 1)).sum()) / yP.sum()
for k in MODELS:
    p = oofP[k]; cands = np.unique(p)[::-1]
    thr = next(t for t in cands if ((p >= t) & (yP == 1)).sum() / yP.sum() >= target)
    TH.append(dict(rule=f'{k} >= {thr:.4f} (sensitivity matched to TIRADS >= 4; threshold chosen on the same OOF data)', **tm(yP, (p >= thr).astype(int))))
pd.DataFrame(TH).to_csv(os.path.join(OUT, 'diag_thresholds.csv'), index=False)

# ---------------- save
pd.DataFrame(PERF).to_csv(os.path.join(OUT, 'diag_performance.csv'), index=False)
pd.DataFrame(DELTA).to_csv(os.path.join(OUT, 'diag_paired_delta.csv'), index=False)
pd.DataFrame(CAL).to_csv(os.path.join(OUT, 'diag_calibration.csv'), index=False)
o = pd.DataFrame({'patient_id': dP.patient_id, 'y': yP, **{k: v for k, v in oofP.items()}}); o.to_csv(os.path.join(OUT, 'diag_oof_primary.csv'), index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 70)
for f in ['diag_performance', 'diag_paired_delta', 'diag_calibration', 'diag_thresholds', 'diag_bethesda3']:
    log(f'\n== {f}\n' + pd.read_csv(os.path.join(OUT, f + '.csv')).round(4).to_string(index=False))
open(os.path.join(OUT, 'diag_log.txt'), 'w', encoding='utf-8').write('\n'.join(LOG))
