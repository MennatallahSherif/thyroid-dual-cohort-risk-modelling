"""
=============================================================
STEP 2 — LNR & LODDS Survival Analysis  (Packets II & III)
Thyroid Cancer Two-Cohort Study
=============================================================
Uses: data/processed/seer_clean.csv
Outputs:
  outputs/tables/table_patient_characteristics.csv
  outputs/tables/table_lnr_lodds_performance.csv
  outputs/figures/fig_km_css_lnr.png
  outputs/figures/fig_km_css_lodds.png
  outputs/figures/fig_km_by_nstage.png
  outputs/figures/fig_calibration_lnr_lodds.png
  outputs/figures/fig_dca_lnr_lodds.png
  logs/step2_log.txt
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings, os
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Try importing survival/ML libraries ───────────────────
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import concordance_index
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("WARNING: lifelines not installed. Run: pip install lifelines")

try:
    from sklearn.calibration import calibration_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Paths ──────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC     = os.path.join(BASE, 'data', 'processed')
FIGS     = os.path.join(BASE, 'outputs', 'figures')
TABS     = os.path.join(BASE, 'outputs', 'tables')
LOGS     = os.path.join(BASE, 'logs')

for d in [FIGS, TABS, LOGS]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOGS, 'step2_log.txt')

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
}

# =============================================================
log_section('STEP 2 — LNR & LODDS Survival Analysis')
log(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# ── Load clean data ────────────────────────────────────────
log('\n[1] Loading seer_clean.csv...')
seer = pd.read_csv(os.path.join(PROC, 'seer_clean.csv'))
log(f'  Loaded: {seer.shape[0]:,} rows x {seer.shape[1]} cols')

# =============================================================
# SECTION A — Data preparation for survival analysis
# =============================================================
log_section('A — Preparing LNR/LODDS analysis subset')

# ── A1. Fix NPLN sentinel codes ────────────────────────────
# SEER codes: NPLN=95 means "all examined nodes positive" (D=95 or D=NPLN)
# NPLN=97 means "97 or more" — unreliable when D < 97
# Rule: keep NPLN=95 only when D=95 (LNR=1.0, valid edge case)
#        exclude NPLN=97 when D != 97 (impossible ratio)
log('\n[A1] Fixing NPLN sentinel codes...')
before = len(seer)

# Rows where NPLN > D are physically impossible — exclude
impossible = seer['NPLN'].notna() & seer['D'].notna() & (seer['NPLN'] > seer['D'])
log(f'  Impossible rows (NPLN > D): {impossible.sum()} -> excluded from node analysis')
seer.loc[impossible, ['LNR','LODDS','NPLN','D']] = np.nan

# ── A2. Build analysis subset ──────────────────────────────
log('\n[A2] Building LNR/LODDS valid subset...')

# Require: valid LNR + LODDS + survival months + css_event
node_df = seer[
    seer['LNR'].notna()   &
    seer['LODDS'].notna() &
    seer['survival_months'].notna() &
    seer['css_event'].notna()
].copy()

log(f'  Full SEER:        {len(seer):,}')
log(f'  LNR/LODDS subset: {len(node_df):,} ({len(node_df)/len(seer)*100:.1f}%)')
log(f'  CSS events:       {node_df["css_event"].sum():,} ({node_df["css_event"].mean()*100:.2f}%)')
log(f'  OS events:        {node_df["os_event"].sum():,} ({node_df["os_event"].mean()*100:.2f}%)')

# ── A3. LNR & LODDS risk groups ──────────────────────────
log('\n[A3] Creating LNR and LODDS risk groups...')

# LNR: 52% of values = 0 (no positive nodes)
# Use clinical meaningful bins (Lee et al. threshold ~0.18, review Packet II)
node_df['LNR_clinical'] = pd.cut(
    node_df['LNR'],
    bins=[-0.001, 0.0, 0.2, 1.0],
    labels=['LNR=0 (N0)', 'LNR 0-0.2', 'LNR >0.2']
)
node_df['LNR_tertile'] = node_df['LNR_clinical']  # alias for KM compatibility

# LODDS tertiles (continuous distribution works well)
node_df['LODDS_tertile'] = pd.qcut(
    node_df['LODDS'], q=3,
    labels=['Low LODDS', 'Mid LODDS', 'High LODDS'],
    duplicates='drop'
)

log('  LNR clinical groups:')
for cat, cnt in node_df['LNR_clinical'].value_counts().sort_index().items():
    log(f'    {cat}: {cnt:,}')
log('  LODDS tertile groups:')
for cat, cnt in node_df['LODDS_tertile'].value_counts().sort_index().items():
    log(f'    {cat}: {cnt:,}')

# ── A4. Histology subsets ──────────────────────────────────
log('\n[A4] Histology subsets for stratified analysis...')
ptc_df = node_df[node_df['histology_subtype'] == 'PTC']
mtc_df = node_df[node_df['histology_subtype'] == 'MTC']
log(f'  PTC subset: {len(ptc_df):,}')
log(f'  MTC subset: {len(mtc_df):,}')

# =============================================================
# SECTION B — Patient characteristics table
# =============================================================
log_section('B — Patient Characteristics Table')

def characteristics_table(df, label):
    rows = []
    rows.append({'Variable': f'--- {label} (n={len(df):,}) ---', 'Value': ''})
    rows.append({'Variable': 'Sex: Female', 'Value': f'{(df["sex"]=="Female").sum():,} ({(df["sex"]=="Female").mean()*100:.1f}%)'})
    rows.append({'Variable': 'Sex: Male',   'Value': f'{(df["sex"]=="Male").sum():,} ({(df["sex"]=="Male").mean()*100:.1f}%)'})
    rows.append({'Variable': 'Age median (IQR)', 'Value': f'{df["age_mid"].median():.0f} ({df["age_mid"].quantile(0.25):.0f}-{df["age_mid"].quantile(0.75):.0f})'})
    rows.append({'Variable': 'Age >= 55',   'Value': f'{df["age_55_plus"].sum():,} ({df["age_55_plus"].mean()*100:.1f}%)'})
    rows.append({'Variable': 'Stage I',     'Value': f'{(df["stage_group"]=="1").sum():,} ({(df["stage_group"]=="1").mean()*100:.1f}%)'})
    rows.append({'Variable': 'Stage II',    'Value': f'{(df["stage_group"]=="2").sum():,} ({(df["stage_group"]=="2").mean()*100:.1f}%)'})
    rows.append({'Variable': 'Stage III',   'Value': f'{(df["stage_group"]=="3").sum():,} ({(df["stage_group"]=="3").mean()*100:.1f}%)'})
    rows.append({'Variable': 'Stage IV',    'Value': f'{df["stage_group"].isin(["4A","4B","4C","4"]).sum():,} ({df["stage_group"].isin(["4A","4B","4C","4"]).mean()*100:.1f}%)'})
    rows.append({'Variable': 'N0a',  'Value': f'{(df["n_stage"]=="N0a").sum():,} ({(df["n_stage"]=="N0a").mean()*100:.1f}%)'})
    rows.append({'Variable': 'N0b',  'Value': f'{(df["n_stage"]=="N0b").sum():,} ({(df["n_stage"]=="N0b").mean()*100:.1f}%)'})
    rows.append({'Variable': 'N1a',  'Value': f'{(df["n_stage"]=="N1a").sum():,} ({(df["n_stage"]=="N1a").mean()*100:.1f}%)'})
    rows.append({'Variable': 'N1b',  'Value': f'{(df["n_stage"]=="N1b").sum():,} ({(df["n_stage"]=="N1b").mean()*100:.1f}%)'})
    rows.append({'Variable': 'Nodes examined median (IQR)', 'Value': f'{df["D"].median():.0f} ({df["D"].quantile(0.25):.0f}-{df["D"].quantile(0.75):.0f})'})
    rows.append({'Variable': 'LNR median (IQR)',   'Value': f'{df["LNR"].median():.3f} ({df["LNR"].quantile(0.25):.3f}-{df["LNR"].quantile(0.75):.3f})'})
    rows.append({'Variable': 'LODDS median (IQR)', 'Value': f'{df["LODDS"].median():.3f} ({df["LODDS"].quantile(0.25):.3f}-{df["LODDS"].quantile(0.75):.3f})'})
    rows.append({'Variable': 'CSS events', 'Value': f'{df["css_event"].sum():,} ({df["css_event"].mean()*100:.2f}%)'})
    rows.append({'Variable': 'Total thyroidectomy', 'Value': f'{(df["surgery_type"]=="Total thyroidectomy").sum():,} ({(df["surgery_type"]=="Total thyroidectomy").mean()*100:.1f}%)'})
    rows.append({'Variable': 'RAI received', 'Value': f'{df["rai"].sum():,} ({df["rai"].mean()*100:.1f}%)'})
    return pd.DataFrame(rows)

tbl = characteristics_table(node_df, 'LNR/LODDS Analysis Subset')
tbl_path = os.path.join(TABS, 'table_patient_characteristics.csv')
tbl.to_csv(tbl_path, index=False, encoding='utf-8')
log(f'  Saved -> {tbl_path}')

# =============================================================
# SECTION C — Kaplan-Meier curves
# =============================================================
log_section('C — Kaplan-Meier Curves')

if not HAS_LIFELINES:
    log('  SKIPPED — lifelines not installed')
else:
    # ── C1. KM: CSS by LNR clinical groups ────────────────
    log('\n[C1] KM: CSS by LNR clinical groups...')
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = [COLORS['teal'], COLORS['blue'], COLORS['red']]
    kmf = KaplanMeierFitter()

    lnr_groups = node_df['LNR_clinical'].dropna().unique()
    lnr_groups = sorted(lnr_groups, key=lambda x: str(x))

    for i, grp in enumerate(lnr_groups):
        mask = node_df['LNR_clinical'] == grp
        sub  = node_df[mask]
        if len(sub) < 10:
            continue
        kmf.fit(sub['survival_months'], sub['css_event'], label=f'{grp} (n={len(sub):,})')
        kmf.plot_survival_function(ax=ax, color=palette[i % len(palette)], ci_show=True)

    ax.set_xlabel('Months from diagnosis')
    ax.set_ylabel('Cancer-specific survival probability')
    ax.set_title('Cancer-Specific Survival by Lymph Node Ratio (LNR)')
    ax.set_ylim(0.7, 1.02)
    ax.legend(loc='lower left')

    # Add at-risk table hint
    ax.annotate('Shaded areas = 95% CI', xy=(0.02, 0.03),
                xycoords='axes fraction', fontsize=8, color='gray')

    plt.tight_layout()
    fig_path = os.path.join(FIGS, 'fig_km_css_lnr.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  Saved -> {fig_path}')

    # Log-rank test
    lnr0 = node_df[node_df['LNR_clinical'] == 'LNR=0 (N0)']
    lnr_high = node_df[node_df['LNR_clinical'] == 'LNR >0.2']
    if len(lnr0) > 0 and len(lnr_high) > 0:
        lr = logrank_test(
            lnr0['survival_months'], lnr_high['survival_months'],
            lnr0['css_event'],       lnr_high['css_event']
        )
        log(f'  Log-rank test (LNR=0 vs LNR>0.2): p = {lr.p_value:.4f}')

    # ── C2. KM: CSS by LODDS tertiles ─────────────────────
    log('\n[C2] KM: CSS by LODDS tertiles...')
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf2 = KaplanMeierFitter()

    for i, grp in enumerate(['Low LODDS', 'Mid LODDS', 'High LODDS']):
        mask = node_df['LODDS_tertile'] == grp
        sub  = node_df[mask]
        if len(sub) < 10:
            continue
        kmf2.fit(sub['survival_months'], sub['css_event'], label=f'{grp} (n={len(sub):,})')
        kmf2.plot_survival_function(ax=ax, color=palette[i], ci_show=True)

    ax.set_xlabel('Months from diagnosis')
    ax.set_ylabel('Cancer-specific survival probability')
    ax.set_title('Cancer-Specific Survival by Log Odds of Positive Nodes (LODDS)')
    ax.set_ylim(0.7, 1.02)
    ax.legend(loc='lower left')
    ax.annotate('Shaded areas = 95% CI', xy=(0.02, 0.03),
                xycoords='axes fraction', fontsize=8, color='gray')

    plt.tight_layout()
    fig_path = os.path.join(FIGS, 'fig_km_css_lodds.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  Saved -> {fig_path}')

    # ── C3. KM: CSS by N stage (N0a/N0b/N1a/N1b) ─────────
    log('\n[C3] KM: CSS by N stage...')
    fig, ax = plt.subplots(figsize=(8, 5))
    n_colors = [COLORS['teal'], COLORS['blue'], COLORS['orange'], COLORS['red']]
    kmf3 = KaplanMeierFitter()

    for i, nstage in enumerate(['N0a', 'N0b', 'N1a', 'N1b']):
        mask = node_df['n_stage'] == nstage
        sub  = node_df[mask]
        if len(sub) < 10:
            continue
        kmf3.fit(sub['survival_months'], sub['css_event'],
                 label=f'{nstage} (n={len(sub):,})')
        kmf3.plot_survival_function(ax=ax, color=n_colors[i], ci_show=False)

    ax.set_xlabel('Months from diagnosis')
    ax.set_ylabel('Cancer-specific survival probability')
    ax.set_title('Cancer-Specific Survival by AJCC N Stage')
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc='lower left')

    plt.tight_layout()
    fig_path = os.path.join(FIGS, 'fig_km_by_nstage.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION D — Cox Proportional Hazards Models
# =============================================================
log_section('D — Cox PH Models: LNR vs LODDS vs N stage')

if not HAS_LIFELINES:
    log('  SKIPPED — lifelines not installed')
else:
    results = []

    def run_cox(df, predictors, label, duration='survival_months', event='css_event'):
        """Fit Cox model, return C-index and summary."""
        cox_df = df[predictors + [duration, event]].dropna()
        if len(cox_df) < 50:
            log(f'  SKIPPED {label} — too few rows ({len(cox_df)})')
            return None, None
        cph = CoxPHFitter(penalizer=0.1)
        try:
            cph.fit(cox_df, duration_col=duration, event_col=event)
            ci = concordance_index(cox_df[duration], -cph.predict_partial_hazard(cox_df), cox_df[event])
            return cph, ci
        except Exception as e:
            log(f'  ERROR fitting {label}: {e}')
            return None, None

    # Covariates shared across models.
    # NOTE: binary age (age_55_plus) is NOT entered alongside continuous age in the
    # nodal models — it is reserved for the dedicated age-threshold analysis (step3),
    # avoiding the dual-specification redundancy (Methods Sec. 4.2.3).
    base_covars = ['age_mid']

    # ── D1. Model 1: N stage only (baseline) ──────────────
    log('\n[D1] Model 1 — N stage (baseline AJCC)...')
    n_stage_dummies = pd.get_dummies(node_df['n_stage'], prefix='n', drop_first=True)
    node_df_enc = pd.concat([node_df, n_stage_dummies], axis=1)
    n_dummy_cols = [c for c in node_df_enc.columns if c.startswith('n_N')]

    cph_nstage, ci_nstage = run_cox(
        node_df_enc,
        base_covars + n_dummy_cols,
        'N stage baseline'
    )
    if cph_nstage:
        log(f'  C-index (N stage): {ci_nstage:.4f}')
        results.append({'Model': 'N stage (AJCC baseline)', 'C-index': round(ci_nstage, 4),
                        'N': len(node_df_enc[base_covars + n_dummy_cols + ['survival_months','css_event']].dropna())})
        log(cph_nstage.summary[['coef','exp(coef)','p']].to_string())

    # ── D2. Model 2: LNR + age ────────────────────────────
    log('\n[D2] Model 2 — LNR + age...')
    cph_lnr, ci_lnr = run_cox(
        node_df,
        ['LNR', 'age_mid'],
        'LNR model'
    )
    if cph_lnr:
        log(f'  C-index (LNR): {ci_lnr:.4f}')
        results.append({'Model': 'LNR + age', 'C-index': round(ci_lnr, 4),
                        'N': len(node_df[['LNR','age_mid','survival_months','css_event']].dropna())})
        log(cph_lnr.summary[['coef','exp(coef)','p']].to_string())

    # ── D3. Model 3: LODDS + age ──────────────────────────
    log('\n[D3] Model 3 — LODDS + age...')
    cph_lodds, ci_lodds = run_cox(
        node_df,
        ['LODDS', 'age_mid'],
        'LODDS model'
    )
    if cph_lodds:
        log(f'  C-index (LODDS): {ci_lodds:.4f}')
        results.append({'Model': 'LODDS + age', 'C-index': round(ci_lodds, 4),
                        'N': len(node_df[['LODDS','age_mid','survival_months','css_event']].dropna())})
        log(cph_lodds.summary[['coef','exp(coef)','p']].to_string())

    # ── D4. Model 4: LNR + LODDS + age (combined) ─────────
    log('\n[D4] Model 4 — LNR + LODDS + age (combined)...')
    cph_both, ci_both = run_cox(
        node_df,
        ['LNR', 'LODDS', 'age_mid'],
        'LNR+LODDS combined'
    )
    if cph_both:
        log(f'  C-index (LNR+LODDS): {ci_both:.4f}')
        results.append({'Model': 'LNR + LODDS + age', 'C-index': round(ci_both, 4),
                        'N': len(node_df[['LNR','LODDS','age_mid','survival_months','css_event']].dropna())})

    # ── D4b. Ordinal N-stage (1 df) — PRIMARY df-controlled benchmark ──
    # Ordinal rank of specified N categories only (N0/N1a/N1b); non-specific
    # (N0 NOS, N1 NOS), NX, and unknown codes cannot be ordinally ranked and
    # are excluded, so this model runs on a smaller n than LNR/LODDS.
    log('\n[D4b] Ordinal N-stage (1 df) benchmark...')
    ord_map = {'N0': 0, 'N0a': 0, 'N0b': 0, 'N1a': 1, 'N1b': 2}
    node_df['n_ordinal'] = node_df['n_stage'].map(ord_map)  # unmapped -> NaN (dropped)

    cph_ord, ci_ord = run_cox(node_df, ['n_ordinal', 'age_mid'], 'Ordinal N-stage')
    if cph_ord:
        n_ord = len(node_df[['n_ordinal','age_mid','survival_months','css_event']].dropna())
        log(f'  C-index (ordinal N-stage): {ci_ord:.4f}  n={n_ord}')
        results.append({'Model': 'Ordinal N-stage (1 df)', 'C-index': round(ci_ord, 4), 'N': n_ord})

    # ── D-boot. Bootstrap 95% CIs for the main models (300 iters, seed 42) ──
    log('\n[D-boot] Bootstrap 95% CIs (300 iterations, seed 42)...')

    def boot_cindex(df, preds, n_boot=300, seed=42,
                    duration='survival_months', event='css_event'):
        rng = np.random.default_rng(seed)
        d = df[preds + [duration, event]].dropna().reset_index(drop=True)
        cs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(d), len(d))
            s = d.iloc[idx]
            try:
                c = CoxPHFitter(penalizer=0.1).fit(s, duration_col=duration, event_col=event)
                cs.append(concordance_index(s[duration], -c.predict_partial_hazard(s), s[event]))
            except Exception:
                pass
        lo, hi = np.percentile(cs, [2.5, 97.5])
        return lo, hi

    boot_specs = [
        ('Ordinal N-stage (1 df)', node_df,     ['n_ordinal', 'age_mid']),
        ('N stage full (7 df)',    node_df_enc, ['age_mid'] + n_dummy_cols),
        ('LNR + age',              node_df,     ['LNR', 'age_mid']),
        ('LODDS + age',            node_df,     ['LODDS', 'age_mid']),
    ]
    boot_ci = {}
    for name, df_b, preds_b in boot_specs:
        lo, hi = boot_cindex(df_b, preds_b)
        boot_ci[name] = (lo, hi)
        log(f'  {name:24s} 95% CI [{lo:.3f}, {hi:.3f}]')

    # ── D5. Subgroup: PTC only ─────────────────────────────
    log('\n[D5] Subgroup: PTC only...')
    cph_ptc, ci_ptc = run_cox(ptc_df, ['LNR','LODDS','age_mid'], 'PTC subgroup')
    if cph_ptc:
        log(f'  C-index (PTC, LNR+LODDS): {ci_ptc:.4f}')
        results.append({'Model': 'LNR+LODDS+age (PTC only)', 'C-index': round(ci_ptc, 4),
                        'N': len(ptc_df[['LNR','LODDS','age_mid','survival_months','css_event']].dropna())})

    # ── D6. Subgroup: N1a vs N1b ──────────────────────────
    log('\n[D6] Subgroup: N1a vs N1b...')
    for ngrp in ['N1a', 'N1b']:
        sub = node_df[node_df['n_stage'] == ngrp]
        cph_n, ci_n = run_cox(sub, ['LNR','LODDS','age_mid'], f'{ngrp} subgroup')
        if cph_n:
            log(f'  C-index ({ngrp}): {ci_n:.4f}')
            results.append({'Model': f'LNR+LODDS+age ({ngrp})', 'C-index': round(ci_n, 4),
                            'N': len(sub[['LNR','LODDS','age_mid','survival_months','css_event']].dropna())})

    # Save performance table
    if results:
        perf_df = pd.DataFrame(results)
        perf_path = os.path.join(TABS, 'table_lnr_lodds_performance.csv')
        perf_df.to_csv(perf_path, index=False, encoding='utf-8')
        log(f'\n  Model performance summary:')
        log(perf_df.to_string(index=False))
        log(f'\n  Saved -> {perf_path}')

# =============================================================
# SECTION E — Calibration plot
# =============================================================
log_section('E — Calibration Plot')

if not HAS_LIFELINES:
    log('  SKIPPED — lifelines not installed')
else:
    log('\n[E1] Calibration: LNR vs LODDS predicted vs observed CSS...')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (model, label, color) in zip(
        axes,
        [(cph_lnr, 'LNR model', COLORS['blue']),
         (cph_lodds, 'LODDS model', COLORS['teal'])]
    ):
        if model is None:
            ax.set_title(f'{label} — not fitted')
            continue

        try:
            pred_cols = [c for c in model.params_.index]
            sub = node_df[pred_cols + ['survival_months', 'css_event']].dropna()

            # Predicted risk at 24 months
            surv_fn = model.predict_survival_function(sub, times=[24])
            pred_surv = surv_fn.iloc[0].values
            pred_risk = 1 - pred_surv

            # Observed: Kaplan-Meier estimate at 24m per decile of predicted risk
            deciles = pd.qcut(pred_risk, q=10, duplicates='drop')
            obs_risks, pred_risks = [], []

            kmf_cal = KaplanMeierFitter()
            for grp in sorted(set(deciles)):
                mask = np.array(deciles == grp)
                if mask.sum() < 5:
                    continue
                kmf_cal.fit(sub.loc[mask, 'survival_months'], sub.loc[mask, 'css_event'])
                km_at_24 = kmf_cal.predict(24)
                obs_risk = 1 - km_at_24
                obs_risks.append(float(obs_risk))
                pred_risks.append(float(pred_risk[mask].mean()))

            ax.scatter(pred_risks, obs_risks, color=color, s=60, zorder=5, label='Deciles')
            # Perfect calibration line
            lim = max(max(pred_risks), max(obs_risks)) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', lw=1, label='Perfect calibration')
            ax.set_xlabel('Predicted 59-month risk')
            ax.set_ylabel('Observed 59-month risk (KM)')
            ax.set_title(f'Calibration Plot — {label}')
            ax.legend()

        except Exception as e:
            log(f'  Calibration error for {label}: {e}')
            ax.set_title(f'{label} — calibration error')

    plt.suptitle('Calibration: Predicted vs Observed 59-month Cancer-Specific Risk',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(FIGS, 'fig_calibration_lnr_lodds.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION F — Decision Curve Analysis (DCA)
# =============================================================
log_section('F — Decision Curve Analysis')

log('\n[F1] DCA: LNR vs LODDS vs N stage at 5% and 10% thresholds...')

if not HAS_LIFELINES:
    log('  SKIPPED — lifelines not installed')
else:
    try:
        fig, ax = plt.subplots(figsize=(9, 6))

        def net_benefit(pred_risk, outcome, threshold):
            """Compute net benefit at a given probability threshold."""
            n = len(pred_risk)
            tp = ((pred_risk >= threshold) & (outcome == 1)).sum()
            fp = ((pred_risk >= threshold) & (outcome == 0)).sum()
            nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            return float(nb)

        thresholds = np.linspace(0.001, 0.15, 100)

        models_for_dca = {}
        if cph_lnr:
            sub_lnr = node_df[['LNR','age_mid','survival_months','css_event']].dropna()
            surv_lnr = cph_lnr.predict_survival_function(sub_lnr, times=[24])
            models_for_dca['LNR model'] = (1 - surv_lnr.iloc[0].values, sub_lnr['css_event'].values)

        if cph_lodds:
            sub_lodds = node_df[['LODDS','age_mid','survival_months','css_event']].dropna()
            surv_lodds = cph_lodds.predict_survival_function(sub_lodds, times=[24])
            models_for_dca['LODDS model'] = (1 - surv_lodds.iloc[0].values, sub_lodds['css_event'].values)

        dca_colors = [COLORS['blue'], COLORS['teal'], COLORS['orange'], COLORS['gray'], COLORS['red']]

        for (model_name, (pred, obs)), color in zip(models_for_dca.items(), dca_colors):
            nbs = [net_benefit(pred, obs, t) for t in thresholds]
            ax.plot(thresholds, nbs, label=model_name, color=color, lw=2)

        # Treat all strategy
        all_nbs = [net_benefit(np.ones(len(node_df)), node_df['css_event'].values, t)
                   for t in thresholds]
        ax.plot(thresholds, all_nbs, 'k-.', lw=1.5, label='Treat all', alpha=0.6)

        # Treat none = 0
        ax.axhline(0, color='gray', lw=1, linestyle=':', label='Treat none')

        ax.set_xlabel('Threshold probability')
        ax.set_ylabel('Net benefit')
        ax.set_title('Decision Curve Analysis — LNR vs LODDS (24-month CSS)')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 0.15)

        # Annotate clinical thresholds
        for thresh, label in [(0.05, '5%'), (0.10, '10%')]:
            ax.axvline(thresh, color='gray', lw=0.8, linestyle='--', alpha=0.5)
            ax.text(thresh + 0.001, ax.get_ylim()[0] * 0.95, label,
                    fontsize=8, color='gray')

        plt.tight_layout()
        fig_path = os.path.join(FIGS, 'fig_dca_lnr_lodds.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        log(f'  Saved -> {fig_path}')

    except Exception as e:
        log(f'  DCA error: {e}')

# =============================================================
# SECTION G — LNR vs LODDS head-to-head comparison figure
# =============================================================
log_section('G — LNR vs LODDS Head-to-Head Figure')

log('\n[G1] Scatter: LNR vs LODDS colored by N stage...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: scatter LNR vs LODDS
ax = axes[0]
n_colors_map = {'N0a': COLORS['teal'], 'N0b': COLORS['blue'],
                'N1a': COLORS['orange'], 'N1b': COLORS['red'],
                'NX':  COLORS['gray']}
for nstage, color in n_colors_map.items():
    mask = node_df['n_stage'] == nstage
    sub  = node_df[mask]
    if len(sub) == 0:
        continue
    ax.scatter(sub['LNR'], sub['LODDS'], alpha=0.15, s=6,
               color=color, label=f'{nstage} (n={len(sub):,})', rasterized=True)

ax.set_xlabel('LNR (lymph node ratio)')
ax.set_ylabel('LODDS (log odds positive nodes)')
ax.set_title('LNR vs LODDS by N Stage\n(illustrates saturation at LNR extremes)')
ax.legend(markerscale=3, fontsize=8)
ax.axvline(0, color='gray', lw=0.5, linestyle=':')
ax.axvline(1, color='gray', lw=0.5, linestyle=':')

# Right: distribution of LODDS by LNR=0 vs LNR=1 (saturation zones)
ax2 = axes[1]
lnr0_mask  = node_df['LNR'] == 0
lnr1_mask  = node_df['LNR'] == 1
lnr_mid    = (node_df['LNR'] > 0) & (node_df['LNR'] < 1)

ax2.hist(node_df.loc[lnr0_mask,  'LODDS'].dropna(), bins=40,
         alpha=0.6, color=COLORS['teal'],  label=f'LNR=0 (n={lnr0_mask.sum():,})', density=True)
ax2.hist(node_df.loc[lnr_mid,    'LODDS'].dropna(), bins=40,
         alpha=0.6, color=COLORS['blue'],  label=f'LNR 0-1 (n={lnr_mid.sum():,})', density=True)
ax2.hist(node_df.loc[lnr1_mask,  'LODDS'].dropna(), bins=40,
         alpha=0.6, color=COLORS['red'],   label=f'LNR=1 (n={lnr1_mask.sum():,})', density=True)

ax2.set_xlabel('LODDS value')
ax2.set_ylabel('Density')
ax2.set_title('LODDS Distribution at LNR Extremes\n(LODDS preserves ranking where LNR saturates)')
ax2.legend(fontsize=8)

plt.suptitle('LNR vs LODDS: comparative view (scatter legend lists the N categories N0a, N0b, N1a, N1b and NX only)',
             fontsize=11, y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_lnr_vs_lodds_comparison.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SAVE LOG
# =============================================================
log_section('STEP 2 COMPLETE')
log(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'\nOutputs:')
log(f'  Figures: {FIGS}')
log(f'  Tables:  {TABS}')

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

print(f'\nLog saved -> {LOG_FILE}')
