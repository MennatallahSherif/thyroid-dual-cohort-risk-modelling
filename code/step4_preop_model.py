"""
=============================================================
STEP 4 — Preoperative Prediction Model  (Packet IV)
Thyroid Cancer Two-Cohort Study
=============================================================
Addresses review Packet IV:
  - Logistic regression, Random Forest, XGBoost
  - Features: TIRADS, age, gender, FNAC, nodule size,
              position, aspect ratio
  - AUROC, calibration curves, DCA
  - Feature importance
  - Two model sets:
      Model Set A — without FNAC (all 4,088 patients)
      Model Set B — with FNAC    (3,942 complete cases)
  - Pre-specified Bethesda III-V subgroup analysis

CORRECTIONS (to match Methods Sec. 4.2.5):
  1. TIRADS reference baseline uses FOLD-WISE EMPIRICAL malignancy
     probabilities per TIRADS category (NOT tirads/5). The tirads/5
     transform is rank-equivalent (same AUC) but produces meaningless
     Brier scores (~0.34 instead of ~0.09).
  2. XGBoost: 100 trees, learning rate 0.05, subsample 0.8.
  3. Random Forest: 100 trees, depth 6, min leaf 20, balanced weights.
  4. 95% CIs: 1,000 bootstrap resamples of OUT-OF-FOLD predictions
     (the previous refit-and-score-on-same-sample bootstrap was
     optimistically biased).
  5. NEW: pre-specified Bethesda III-V subgroup with bootstrap CIs.

Uses:  data/processed/thyroidxl_clean.csv
Outputs:
  outputs/figures/fig_roc_comparison.png
  outputs/figures/fig_calibration_preop.png
  outputs/figures/fig_dca_preop.png
  outputs/figures/fig_feature_importance.png
  outputs/figures/fig_tirads_roc.png
  outputs/tables/table_model_performance.csv
  outputs/tables/table_feature_importance.csv
  outputs/tables/table_tirads_thresholds.csv
  outputs/tables/TableS6_bethesda_subgroup.csv
  logs/step4_log.txt
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, os
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Library imports ────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (roc_auc_score, roc_curve, brier_score_loss)
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: scikit-learn not installed. Run: pip install scikit-learn")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

# ── Paths ──────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, 'data', 'processed')
FIGS  = os.path.join(BASE, 'outputs', 'figures')
TABS  = os.path.join(BASE, 'outputs', 'tables')
LOGS  = os.path.join(BASE, 'logs')

for d in [FIGS, TABS, LOGS]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOGS, 'step4_log.txt')

SEED = 42

# ── Logger ─────────────────────────────────────────────────
log_lines = []
def log(msg=''):
    print(msg)
    log_lines.append(str(msg))

def log_section(title):
    line = '=' * 60
    log(f'\n{line}')
    log(f'  {title}')
    log(line)

# ── Plot style ─────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        150,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.labelsize':    11,
    'axes.titlesize':    12,
    'legend.fontsize':   9,
    'figure.facecolor':  'white',
})

COLORS = {
    'blue':   '#2563EB',
    'teal':   '#0D9488',
    'orange': '#EA580C',
    'red':    '#DC2626',
    'purple': '#7C3AED',
    'gray':   '#6B7280',
    'green':  '#16A34A',
}

# =============================================================
log_section('STEP 4 — Preoperative Prediction Model (Packet IV)')
log(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# ── Load data ──────────────────────────────────────────────
log('\n[1] Loading thyroidxl_clean.csv...')
xl = pd.read_csv(os.path.join(PROC, 'thyroidxl_clean.csv'))
xl['age'] = pd.to_numeric(xl['age'], errors='coerce')
log(f'  Loaded: {xl.shape[0]:,} rows')
log(f'  Malignant: {xl["malignant"].sum():,} ({xl["malignant"].mean()*100:.1f}%)')
log(f'  Benign:    {(xl["malignant"]==0).sum():,} ({(xl["malignant"]==0).mean()*100:.1f}%)')

if not HAS_SKLEARN:
    log('\nCRITICAL: scikit-learn not installed. Cannot run Step 4.')
    log('Run: pip install scikit-learn xgboost')
    import sys; sys.exit(1)

# =============================================================
# SECTION A — Feature engineering
# =============================================================
log_section('A — Feature Engineering')

log('\n[A1] Encoding position...')
xl['pos_isthmus'] = (xl['position'] == 'Isthmus').astype(int)
xl['pos_left']    = (xl['position'] == 'Left').astype(int)
# Reference: Right (most common)

log('\n[A2] Defining feature sets...')

# Set A: without FNAC (maximizes sample size)
FEATURES_A = [
    'tirads',       # strongest single predictor
    'age',          # continuous
    'gender',       # 1=Male, 2=Female
    'width',        # nodule size
    'height',       # nodule size
    'nodule_area',  # width x height
    'aspect_ratio', # width / height
    'age_55_plus',  # binary age flag
    'pos_isthmus',  # isthmus position (higher malignancy)
    'pos_left',     # left lobe
]

# Set B: with FNAC (Bethesda system)
# NOTE: 42.4% of malignant labels are FNAC-confirmed, so the Bethesda
# predictor is not fully independent of the outcome. Set B is therefore
# an UPPER-BOUND scenario (Methods Sec. 4.2.5); primary inference rests
# on Set A and on the Bethesda III-V subgroup.
FEATURES_B = FEATURES_A + ['fnac']

df_A = xl[FEATURES_A + ['malignant']].dropna().reset_index(drop=True)
df_B = xl[FEATURES_B + ['malignant']].dropna().reset_index(drop=True)

log(f'  Set A (without FNAC): {len(df_A):,} patients')
log(f'  Set B (with FNAC):    {len(df_B):,} patients')
log(f'  Prevalence A: {df_A["malignant"].mean()*100:.1f}%  '
    f'B: {df_B["malignant"].mean()*100:.1f}%')

# =============================================================
# SECTION B — Model definitions and validation helpers
# =============================================================
log_section('B — Model and Validation Setup')

def make_models():
    """Fresh model instances (Methods Sec. 4.2.5 specifications)."""
    m = {
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=SEED
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=20,
            class_weight='balanced', random_state=SEED, n_jobs=1
        ),
    }
    if HAS_XGB:
        m['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, subsample=0.8,
            eval_metric='logloss', random_state=SEED, verbosity=0
        )
    return m

MODEL_NAMES = list(make_models().keys())


def out_of_fold_predictions(X, y, tirads_raw, n_splits=5):
    """
    Five-fold stratified CV. Returns out-of-fold predicted probabilities for
    each model, plus the TIRADS reference baseline.

    TIRADS baseline: each TIRADS category is mapped to its EMPIRICAL malignancy
    rate computed WITHIN THE TRAINING FOLD ONLY, then applied to the held-out
    fold. This makes TIRADS a properly calibrated probability reference (Brier
    is meaningful) while leaving AUC unchanged, since the mapping is monotone.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    oof['TIRADS alone (baseline)'] = np.zeros(len(y))

    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        X_tr = scaler.transform(X[train_idx])
        X_va = scaler.transform(X[val_idx])

        for name, model in make_models().items():
            model.fit(X_tr, y[train_idx])
            oof[name][val_idx] = model.predict_proba(X_va)[:, 1]

        # Fold-wise empirical TIRADS -> P(malignant)
        train_rates = pd.Series(y[train_idx]).groupby(tirads_raw[train_idx]).mean()
        fallback    = y[train_idx].mean()
        oof['TIRADS alone (baseline)'][val_idx] = (
            pd.Series(tirads_raw[val_idx]).map(train_rates).fillna(fallback).values
        )

    return oof


def bootstrap_auc_ci(y, probs, n_boot=1000, seed=SEED):
    """95% CI by resampling OUT-OF-FOLD predictions (no refitting)."""
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], probs[idx]))
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)

# =============================================================
# SECTION C — Model training and evaluation
# =============================================================
log_section('C — Model Training and Evaluation')

results  = []
cv_probs = {}   # (set_name, model) -> (probs, y)
store    = {}   # set_name -> (df, y, oof)

for set_name, df, features in [
    ('Without FNAC', df_A, FEATURES_A),
    ('With FNAC',    df_B, FEATURES_B),
]:
    X = df[features].values
    y = df['malignant'].values
    tirads_raw = df['tirads'].values

    log(f'\n--- Model set: {set_name} (n={len(y):,}) ---')
    oof = out_of_fold_predictions(X, y, tirads_raw)
    store[set_name] = (df, y, oof)

    base_auc = roc_auc_score(y, oof['TIRADS alone (baseline)'])

    for model_name in ['TIRADS alone (baseline)'] + MODEL_NAMES:
        probs = oof[model_name]
        auc   = roc_auc_score(y, probs)
        brier = brier_score_loss(y, probs)
        lo, hi = bootstrap_auc_ci(y, probs)

        log(f'\n  [{model_name}]')
        log(f'    CV AUC:      {auc:.4f}  (95% CI {lo:.4f}-{hi:.4f})')
        log(f'    Brier score: {brier:.4f}')

        cv_probs[f'{set_name}|{model_name}'] = (probs, y)

        results.append({
            'Feature set':      set_name,
            'Model':            model_name,
            'CV AUC':           round(auc, 4),
            'Bootstrap 95% CI': f'{lo:.3f}-{hi:.3f}',
            'Brier score':      round(brier, 4),
            'Delta AUC vs TIRADS': '—' if model_name.startswith('TIRADS')
                                   else f'{auc - base_auc:+.3f}',
            'N':                len(y),
        })

perf_df = pd.DataFrame(results)
perf_path = os.path.join(TABS, 'table_model_performance.csv')
perf_df.to_csv(perf_path, index=False, encoding='utf-8')
log('\n  Model performance summary (Table 5):')
log(perf_df.to_string(index=False))
log(f'\n  Saved -> {perf_path}')

# =============================================================
# SECTION C2 — Pre-specified Bethesda III-V subgroup  [NEW]
# =============================================================
log_section('C2 — Bethesda III-V Subgroup (pre-specified)')

df_b, y_b, oof_b = store['With FNAC']
sub_mask = df_b['fnac'].isin([3, 4, 5]).values
y_sub = y_b[sub_mask]

log(f'\n  Subgroup n:        {sub_mask.sum()}')
log(f'  Malignant:         {int(y_sub.sum())} ({y_sub.mean()*100:.1f}%)')
log(f'  Bethesda III:      {(df_b["fnac"]==3).sum()}')
log(f'  Bethesda IV:       {(df_b["fnac"]==4).sum()}')
log(f'  Bethesda V:        {(df_b["fnac"]==5).sum()}')
log('\n  NOTE: this band is dominated by Bethesda V and is therefore high-prevalence;'
    '\n  it is the indeterminate/suspicious band, not a balanced sample.')

sub_rows = []
for model_name in ['TIRADS alone (baseline)'] + MODEL_NAMES:
    probs_sub = oof_b[model_name][sub_mask]
    auc = roc_auc_score(y_sub, probs_sub)
    lo, hi = bootstrap_auc_ci(y_sub, probs_sub)
    log(f'  {model_name:26s} AUC = {auc:.3f}  [95% CI {lo:.3f}-{hi:.3f}]')
    sub_rows.append({
        'Model': model_name,
        'AUC': round(auc, 3),
        'Bootstrap 95% CI': f'{lo:.3f}-{hi:.3f}',
        'n': int(sub_mask.sum()),
        'Malignant': int(y_sub.sum()),
        'Prevalence (%)': round(y_sub.mean()*100, 1),
    })

sub_df = pd.DataFrame(sub_rows)
sub_path = os.path.join(TABS, 'TableS6_bethesda_subgroup.csv')
sub_df.to_csv(sub_path, index=False, encoding='utf-8')
log(f'\n  Saved -> {sub_path}')

# =============================================================
# SECTION D — ROC comparison figure
# =============================================================
log_section('D — ROC Comparison Figure')

log('\n[D1] ROC curves: all models vs TIRADS baseline...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

set_labels = ['Without FNAC', 'With FNAC']
model_styles = {
    'Logistic Regression':     (COLORS['blue'],   '-',  2.0),
    'Random Forest':           (COLORS['teal'],   '-',  2.0),
    'XGBoost':                 (COLORS['orange'], '-',  2.0),
    'TIRADS alone (baseline)': (COLORS['gray'],   '--', 1.5),
}

for ax, set_name in zip(axes, set_labels):
    for model_name, (color, ls, lw) in model_styles.items():
        key = f'{set_name}|{model_name}'
        if key not in cv_probs:
            continue
        probs, y = cv_probs[key]
        fpr, tpr, _ = roc_curve(y, probs)
        auc = roc_auc_score(y, probs)
        ax.plot(fpr, tpr, color=color, linestyle=ls, lw=lw,
                label=f'{model_name} (AUC={auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
    ax.set_xlabel('1 - Specificity (False positive rate)')
    ax.set_ylabel('Sensitivity (True positive rate)')
    ax.set_title(f'ROC Curves — {set_name}')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

plt.suptitle('Preoperative Malignancy Prediction — ROC Curves (5-fold CV)',
             fontsize=12, y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_roc_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION E — Calibration figure
# =============================================================
log_section('E — Calibration Plot')

log('\n[E1] Calibration curves: predicted vs observed malignancy...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cal_models = ['Logistic Regression', 'Random Forest',
              'TIRADS alone (baseline)'] + (['XGBoost'] if HAS_XGB else [])
cal_colors = [COLORS['blue'], COLORS['teal'], COLORS['gray'], COLORS['orange']]

for ax, set_name in zip(axes, set_labels):
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Perfect calibration')

    for model_name, color in zip(cal_models, cal_colors):
        key = f'{set_name}|{model_name}'
        if key not in cv_probs:
            continue
        probs, y = cv_probs[key]
        try:
            fraction_pos, mean_pred = calibration_curve(
                y, probs, n_bins=8, strategy='uniform'
            )
            ax.plot(mean_pred, fraction_pos, 'o-', color=color, lw=1.5,
                    markersize=5, label=model_name)
        except Exception as e:
            log(f'    Calibration error for {model_name}: {e}')

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction malignant (observed)')
    ax.set_title(f'Calibration — {set_name}')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.suptitle('Calibration: Preoperative Malignancy Prediction Models',
             fontsize=12, y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_calibration_preop.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION F — Decision Curve Analysis
# =============================================================
log_section('F — Decision Curve Analysis (DCA)')

log('\n[F1] DCA: all models vs TIRADS baseline...')

def net_benefit(pred_prob, outcome, threshold):
    n  = len(pred_prob)
    tp = ((pred_prob >= threshold) & (outcome == 1)).sum()
    fp = ((pred_prob >= threshold) & (outcome == 0)).sum()
    return (tp / n) - (fp / n) * (threshold / (1 - threshold))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
thresholds = np.linspace(0.01, 0.90, 150)

for ax, set_name in zip(axes, set_labels):
    _, y_ref = cv_probs[f'{set_name}|TIRADS alone (baseline)']
    treat_all = [net_benefit(np.ones(len(y_ref)), y_ref, t) for t in thresholds]
    ax.plot(thresholds, treat_all, 'k-.', lw=1.5, alpha=0.5, label='Treat all')
    ax.axhline(0, color='gray', lw=1, linestyle=':', label='Treat none')

    for model_name, (color, ls, lw) in model_styles.items():
        key = f'{set_name}|{model_name}'
        if key not in cv_probs:
            continue
        probs, y = cv_probs[key]
        nbs = [net_benefit(probs, y, t) for t in thresholds]
        ax.plot(thresholds, nbs, color=color, linestyle=ls, lw=lw, label=model_name)

    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'DCA — {set_name}')
    ax.set_xlim(0.0, 0.90)
    ax.set_ylim(-0.05, 0.35)
    ax.legend(loc='upper right', fontsize=8)

    for thresh, label in [(0.2, 'T4 threshold'), (0.8, 'T5 threshold')]:
        ax.axvline(thresh, color='gray', lw=0.8, linestyle='--', alpha=0.4)
        ax.text(thresh + 0.01, 0.30, label, fontsize=7, color='gray')

plt.suptitle('Decision Curve Analysis — Preoperative Malignancy Models',
             fontsize=12, y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_dca_preop.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION G — Feature importance
# =============================================================
log_section('G — Feature Importance')

log('\n[G1] Feature importance: Logistic Regression + Random Forest...')

feat_results    = []
rf_importances  = {'Without FNAC': {}, 'With FNAC': {}}

for set_name, df, features in [
    ('Without FNAC', df_A, FEATURES_A),
    ('With FNAC',    df_B, FEATURES_B),
]:
    X = df[features].values
    y = df['malignant'].values
    X_sc = StandardScaler().fit_transform(X)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED).fit(X_sc, y)
    for feat, coef in zip(features, lr.coef_[0]):
        feat_results.append({
            'Feature set':    set_name,
            'Feature':        feat,
            'LR coefficient': round(coef, 4),
            'LR odds ratio':  round(np.exp(coef), 4),
        })

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=6, min_samples_leaf=20,
        class_weight='balanced', random_state=SEED, n_jobs=1
    ).fit(X_sc, y)
    for feat, imp in zip(features, rf.feature_importances_):
        rf_importances[set_name][feat] = imp

for row in feat_results:
    row['RF importance'] = round(
        rf_importances[row['Feature set']].get(row['Feature'], 0), 4
    )

feat_df = pd.DataFrame(feat_results)
feat_path = os.path.join(TABS, 'table_feature_importance.csv')
feat_df.to_csv(feat_path, index=False, encoding='utf-8')
log(f'  Saved -> {feat_path}')

log('\n[G2] Feature importance figure...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

feature_labels = {
    'tirads':       'TIRADS score',
    'age':          'Age (continuous)',
    'gender':       'Gender',
    'width':        'Nodule width',
    'height':       'Nodule height',
    'nodule_area':  'Nodule area',
    'aspect_ratio': 'Aspect ratio',
    'age_55_plus':  'Age >= 55',
    'pos_isthmus':  'Position: isthmus',
    'pos_left':     'Position: left',
    'fnac':         'FNAC (Bethesda)',
}

for ax, (set_name, features) in zip(axes, [
    ('Without FNAC', FEATURES_A),
    ('With FNAC',    FEATURES_B),
]):
    imp_dict = rf_importances[set_name]
    names_raw   = [feature_labels.get(f, f) for f in features]
    importances = [imp_dict.get(f, 0) for f in features]

    sorted_pairs = sorted(zip(importances, names_raw), reverse=True)
    imps, names = zip(*sorted_pairs)

    colors = [COLORS['orange'] if 'FNAC' in n else
              COLORS['blue']   if 'TIRADS' in n else
              COLORS['teal']   for n in names]

    bars = ax.barh(range(len(names)), imps, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Random Forest feature importance')
    ax.set_title(f'Feature Importance — {set_name}')
    ax.invert_yaxis()

    for bar, val in zip(bars, imps):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

plt.suptitle('Random Forest Feature Importance — Preoperative Malignancy Prediction',
             fontsize=11, y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_feature_importance.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION H — TIRADS standalone ROC + clinical thresholds
# =============================================================
log_section('H — TIRADS Standalone Analysis')

log('\n[H1] TIRADS ROC with clinical threshold annotations...')
fig, ax = plt.subplots(figsize=(8, 7))

y_A = df_A['malignant'].values
tirads_probs, _ = cv_probs['Without FNAC|TIRADS alone (baseline)']
fpr_t, tpr_t, _ = roc_curve(y_A, tirads_probs)
auc_t = roc_auc_score(y_A, tirads_probs)
ax.plot(fpr_t, tpr_t, color=COLORS['gray'], lw=2,
        label=f'TIRADS score (AUC={auc_t:.3f})')

best_key = 'Without FNAC|Logistic Regression'
if best_key in cv_probs:
    probs_lr, y_lr = cv_probs[best_key]
    fpr_lr, tpr_lr, _ = roc_curve(y_lr, probs_lr)
    auc_lr = roc_auc_score(y_lr, probs_lr)
    ax.plot(fpr_lr, tpr_lr, color=COLORS['blue'], lw=2,
            label=f'Logistic Regression (AUC={auc_lr:.3f})')

tirads_vals = sorted(df_A['tirads'].unique())
for t_val in tirads_vals:
    pred = (df_A['tirads'] >= t_val).astype(int)
    tp = ((pred == 1) & (y_A == 1)).sum()
    fp = ((pred == 1) & (y_A == 0)).sum()
    fn = ((pred == 0) & (y_A == 1)).sum()
    tn = ((pred == 0) & (y_A == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr_point = 1 - spec

    ax.scatter(fpr_point, sens, s=80, zorder=5, color=COLORS['red'], marker='D')
    ax.annotate(f'T>={t_val}\n(Sens={sens:.2f},\nSpec={spec:.2f})',
                xy=(fpr_point, sens),
                xytext=(fpr_point + 0.05, sens - 0.05),
                fontsize=7, color=COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=0.8))

ax.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
ax.set_xlabel('1 - Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title('TIRADS Score ROC with Clinical Cutpoint Analysis\n'
             '(diamonds = TIRADS >= threshold operating points)')
ax.legend(loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_tirads_roc.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

log('\n[H2] TIRADS sensitivity/specificity table:')
thresh_rows = []
for t_val in tirads_vals:
    pred = (df_A['tirads'] >= t_val).astype(int)
    tp = ((pred == 1) & (y_A == 1)).sum()
    fp = ((pred == 1) & (y_A == 0)).sum()
    fn = ((pred == 0) & (y_A == 1)).sum()
    tn = ((pred == 0) & (y_A == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0
    log(f'  TIRADS >= {t_val}: Sens={sens:.3f} Spec={spec:.3f} '
        f'PPV={ppv:.3f} NPV={npv:.3f}')
    thresh_rows.append({
        'Cutpoint':    f'TIRADS >= {t_val}',
        'Sensitivity': round(sens, 3),
        'Specificity': round(spec, 3),
        'PPV':         round(ppv, 3),
        'NPV':         round(npv, 3),
    })

thresh_df = pd.DataFrame(thresh_rows)
thresh_path = os.path.join(TABS, 'table_tirads_thresholds.csv')
thresh_df.to_csv(thresh_path, index=False, encoding='utf-8')
log(f'  Saved -> {thresh_path}')

# =============================================================
# SECTION I — Summary findings for paper
# =============================================================
log_section('I — Summary findings for paper')

for set_name in set_labels:
    sub = perf_df[perf_df['Feature set'] == set_name]
    base = sub[sub['Model'].str.startswith('TIRADS')].iloc[0]
    best = sub[~sub['Model'].str.startswith('TIRADS')].sort_values(
        'CV AUC', ascending=False).iloc[0]
    log(f'\n  {set_name}:')
    log(f'    TIRADS baseline: AUC={base["CV AUC"]:.3f}, Brier={base["Brier score"]:.3f}')
    log(f'    Best model:      {best["Model"]} AUC={best["CV AUC"]:.3f}, '
        f'Brier={best["Brier score"]:.3f} ({best["Delta AUC vs TIRADS"]})')

log('\n  Bethesda III-V subgroup:')
log(sub_df.to_string(index=False))

# =============================================================
# SAVE LOG
# =============================================================
log_section('STEP 4 COMPLETE')
log(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('\nOutputs:')
log(f'  Figures: {FIGS}')
log(f'  Tables:  {TABS}')

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

print(f'\nLog saved -> {LOG_FILE}')