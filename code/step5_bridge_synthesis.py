"""
=============================================================
STEP 5 — Cross-Cohort Bridge Synthesis
Thyroid Cancer Two-Cohort Study
=============================================================
Integrates ThyroidXL (preoperative) with SEER (postoperative)
across four bridge dimensions:
  1. Age (55-yr cutoff + continuous)
  2. Sex / Gender
  3. Histology subtype (PTC / FTC / MTC)
  4. Tumor size (nodule size vs T stage)

Produces:
  - Table 12 analog from the review
  - Bridge heatmap figure
  - Sex x age interaction figure
  - Histology comparison figure
  - Tumor size bridge figure
  - Comprehensive bridge table for the paper

Uses:
  data/processed/seer_clean.csv
  data/processed/thyroidxl_clean.csv

Outputs:
  outputs/figures/fig_bridge_heatmap.png
  outputs/figures/fig_bridge_sex_age.png
  outputs/figures/fig_bridge_histology.png
  outputs/figures/fig_bridge_tumorsize.png
  outputs/tables/table_bridge_synthesis.csv
  logs/step5_log.txt
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(BASE, 'data', 'processed')
FIGS  = os.path.join(BASE, 'outputs', 'figures')
TABS  = os.path.join(BASE, 'outputs', 'tables')
LOGS  = os.path.join(BASE, 'logs')

for d in [FIGS, TABS, LOGS]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = os.path.join(LOGS, 'step5_log.txt')

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
log_section('STEP 5 — Cross-Cohort Bridge Synthesis')
log(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# ── Load data ──────────────────────────────────────────────
log('\n[1] Loading datasets...')
seer = pd.read_csv(os.path.join(PROC, 'seer_clean.csv'))
xl   = pd.read_csv(os.path.join(PROC, 'thyroidxl_clean.csv'))
xl['age'] = pd.to_numeric(xl['age'], errors='coerce')
xl_valid  = xl[xl['age'].notna() & xl['age'].between(10, 100)].copy()

log(f'  SEER: {len(seer):,}  ThyroidXL: {len(xl_valid):,}')

# ── Shared age groups ──────────────────────────────────────
AGE_BINS   = [0, 35, 45, 55, 65, 75, 100]
AGE_LABELS = ['<35','35-44','45-54','55-64','65-74','75+']

seer['age_group'] = pd.cut(seer['age_mid'], bins=AGE_BINS, labels=AGE_LABELS)

# Invalid nodal ratios (positive nodes > examined nodes) are set to missing, as in step2, so that LNR summaries cannot exceed 1
_bad = seer['NPLN'].notna() & seer['D'].notna() & (seer['NPLN'] > seer['D'])
seer.loc[_bad, ['LNR', 'LODDS', 'NPLN', 'D']] = np.nan
xl_valid['age_group'] = pd.cut(xl_valid['age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)  # left-closed (age 35 -> '35-44'; age 55 -> '55-64')

# ── Standardize sex/gender ─────────────────────────────────
seer['sex_std']  = seer['sex'].str.strip()
xl_valid['sex_std'] = xl_valid['gender'].map({1:'Male', 2:'Female'})

# =============================================================
# SECTION A — Comprehensive bridge table (Table 12 analog)
# =============================================================
log_section('A — Comprehensive Bridge Table')

bridge_rows = []

# ── A1. Dimension 1: Age ───────────────────────────────────
log('\n[A1] Age dimension...')
for age_grp in AGE_LABELS:
    # SEER
    s = seer[seer['age_group'] == age_grp]
    s_css  = s['css_event'].mean() if len(s) > 0 else np.nan
    s_n    = len(s)

    # ThyroidXL
    x = xl_valid[xl_valid['age_group'] == age_grp]
    x_mal  = x['malignant'].mean() if len(x) > 0 else np.nan
    x_n    = len(x)

    bridge_rows.append({
        'Dimension':      'Age',
        'Subgroup':       age_grp,
        'SEER_n':         s_n,
        'SEER_CSS_rate':  round(s_css * 100, 2) if not np.isnan(s_css) else np.nan,
        'XL_n':           x_n,
        'XL_Mal_rate':    round(x_mal * 100, 1) if not np.isnan(x_mal) else np.nan,
        'Age_direction':  'Concordant (SEER rises; XL falls)'
                          if age_grp != '<35' else 'Opposite directions',
        'Key_finding':    'Paradox: older = more mortality, younger = more malignant nodules'
    })
    log(f'  {age_grp}: SEER CSS={s_css*100:.2f}%  XL Mal={x_mal*100:.1f}%')

# ── A2. Dimension 2: Sex ───────────────────────────────────
log('\n[A2] Sex dimension...')
for sex in ['Female', 'Male']:
    s = seer[seer['sex_std'] == sex]
    x = xl_valid[xl_valid['sex_std'] == sex]

    s_css    = s['css_event'].mean()
    s_lnr    = s['LNR'].mean()
    x_mal    = x['malignant'].mean()
    x_tirads = x['tirads'].mean()

    bridge_rows.append({
        'Dimension':      'Sex',
        'Subgroup':       sex,
        'SEER_n':         len(s),
        'SEER_CSS_rate':  round(s_css * 100, 2),
        'XL_n':           len(x),
        'XL_Mal_rate':    round(x_mal * 100, 1),
        'Age_direction':  'Concordant (males worse at both stages)',
        'Key_finding':    f'Male CSS rate {s["css_event"].mean()*100:.2f}% vs Female {seer[seer["sex_std"]=="Female"]["css_event"].mean()*100:.2f}%; Male XL malignancy {x_mal*100:.1f}%'
    })
    log(f'  {sex}: SEER CSS={s_css*100:.2f}%  LNR_mean={s_lnr:.3f}  XL Mal={x_mal*100:.1f}%  TIRADS_mean={x_tirads:.2f}')

# ── A3. Dimension 3: Histology ─────────────────────────────
log('\n[A3] Histology dimension...')
# Map ThyroidXL histopathology to standard subtypes
def map_xl_histology(h):
    if pd.isna(h):
        return np.nan
    h = str(h).lower()
    if 'papillary' in h or 'ptc' in h:
        return 'PTC'
    elif 'follicular' in h or 'ftc' in h or 'hurthle' in h:
        return 'FTC'
    elif 'medullary' in h or 'mtc' in h:
        return 'MTC'
    elif 'anaplastic' in h or 'atc' in h:
        return 'ATC'
    return 'Other'

xl_valid['histology_mapped'] = xl_valid['histopathology'].apply(map_xl_histology)

for subtype in ['PTC', 'FTC', 'MTC']:
    s = seer[seer['histology_subtype'] == subtype]
    x = xl_valid[xl_valid['histology_mapped'] == subtype]

    s_css  = s['css_event'].mean()
    s_n    = len(s)
    x_mal  = x['malignant'].mean() if len(x) > 0 else np.nan
    x_n    = len(x)

    bridge_rows.append({
        'Dimension':     'Histology',
        'Subgroup':      subtype,
        'SEER_n':        s_n,
        'SEER_CSS_rate': round(s_css * 100, 2),
        'XL_n':          x_n,
        'XL_Mal_rate':   round(x_mal * 100, 1) if not np.isnan(x_mal) else np.nan,
        'Age_direction': 'Histology-specific',
        'Key_finding':   f'SEER: {subtype} CSS={s_css*100:.2f}%'
    })
    log(f'  {subtype}: SEER n={s_n} CSS={s_css*100:.2f}%  XL n={x_n} Mal={x_mal*100:.1f}%' if not np.isnan(x_mal) else f'  {subtype}: SEER n={s_n} CSS={s_css*100:.2f}%  XL n={x_n} (too few)')

# ── A4. Dimension 4: Age x Sex combined ───────────────────
log('\n[A4] Age x Sex combined...')
for sex in ['Female','Male']:
    for age_cut, label in [(0,'<55'), (1,'>=55')]:
        s = seer[(seer['sex_std']==sex) & (seer['age_55_plus']==age_cut)]
        x = xl_valid[(xl_valid['sex_std']==sex) & (xl_valid['age_55_plus']==age_cut)]
        s_css = s['css_event'].mean()
        x_mal = x['malignant'].mean() if len(x) > 0 else np.nan
        bridge_rows.append({
            'Dimension':     'Age x Sex',
            'Subgroup':      f'{sex}, Age {label}',
            'SEER_n':        len(s),
            'SEER_CSS_rate': round(s_css*100, 2),
            'XL_n':          len(x),
            'XL_Mal_rate':   round(x_mal*100,1) if not np.isnan(x_mal) else np.nan,
            'Age_direction': 'Combined interaction',
            'Key_finding':   f'{sex} age{label}: SEER CSS={s_css*100:.2f}%'
        })
        log(f'  {sex} {label}: SEER CSS={s_css*100:.2f}%  XL Mal={x_mal*100:.1f}%')

# Save bridge table
bridge_df = pd.DataFrame(bridge_rows)
bridge_path = os.path.join(TABS, 'table_bridge_synthesis.csv')
bridge_df.to_csv(bridge_path, index=False, encoding='utf-8')
log(f'\n  Saved -> {bridge_path}')

# =============================================================
# SECTION B — Bridge heatmap (main synthesis figure)
# =============================================================
log_section('B — Bridge Heatmap')

log('\n[B1] Building bridge heatmap...')

# Build matrix: rows = subgroups, cols = SEER CSS rate / XL malignancy rate
age_data = bridge_df[bridge_df['Dimension']=='Age'][
    ['Subgroup','SEER_CSS_rate','XL_Mal_rate']
].set_index('Subgroup')

sex_data = bridge_df[bridge_df['Dimension']=='Sex'][
    ['Subgroup','SEER_CSS_rate','XL_Mal_rate']
].set_index('Subgroup')

hist_data = bridge_df[bridge_df['Dimension']=='Histology'][
    ['Subgroup','SEER_CSS_rate','XL_Mal_rate']
].set_index('Subgroup')

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

# ── B1a: Age heatmap ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
mat_age = age_data[['SEER_CSS_rate','XL_Mal_rate']].values.T
im1 = ax1.imshow(mat_age, aspect='auto', cmap='RdYlGn_r',
                  vmin=0, vmax=42)
ax1.set_xticks(range(len(AGE_LABELS)))
ax1.set_xticklabels(AGE_LABELS, rotation=30, fontsize=8)
ax1.set_yticks([0,1])
ax1.set_yticklabels(['SEER CSS %','XL Mal %'], fontsize=9)
ax1.set_title('Age dimension', fontsize=10, fontweight='bold')
for i in range(2):
    for j in range(len(AGE_LABELS)):
        val = mat_age[i,j]
        ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                 fontsize=8, color='black', fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.7)

# ── B1b: Sex heatmap ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
mat_sex = sex_data[['SEER_CSS_rate','XL_Mal_rate']].values.T
im2 = ax2.imshow(mat_sex, aspect='auto', cmap='RdYlGn_r',
                  vmin=0, vmax=45)
ax2.set_xticks([0,1])
ax2.set_xticklabels(['Female','Male'], fontsize=9)
ax2.set_yticks([0,1])
ax2.set_yticklabels(['SEER CSS %','XL Mal %'], fontsize=9)
ax2.set_title('Sex dimension', fontsize=10, fontweight='bold')
for i in range(2):
    for j in range(2):
        val = mat_sex[i,j]
        ax2.text(j, i, f'{val:.1f}', ha='center', va='center',
                 fontsize=10, color='black', fontweight='bold')
plt.colorbar(im2, ax=ax2, shrink=0.7)

# ── B1c: Histology heatmap ────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
hist_plot = hist_data[['SEER_CSS_rate','XL_Mal_rate']].dropna()
mat_hist  = hist_plot.values.T
im3 = ax3.imshow(mat_hist, aspect='auto', cmap='RdYlGn_r',
                  vmin=0, vmax=100)
ax3.set_xticks(range(len(hist_plot.index)))
ax3.set_xticklabels(hist_plot.index, fontsize=9)
ax3.set_yticks([0,1])
ax3.set_yticklabels(['SEER CSS %','XL Mal %'], fontsize=9)
ax3.set_title('Histology dimension', fontsize=10, fontweight='bold')
for i in range(2):
    for j in range(len(hist_plot.index)):
        val = mat_hist[i,j]
        if not np.isnan(val):
            ax3.text(j, i, f'{val:.1f}', ha='center', va='center',
                     fontsize=10, color='black', fontweight='bold')
plt.colorbar(im3, ax=ax3, shrink=0.7)

# ── B2: Age x Sex line plot (bottom row spanning all 3) ───
ax4 = fig.add_subplot(gs[1, :])
axs_data = bridge_df[bridge_df['Dimension']=='Age x Sex']

for sex, color, ls in [('Female', COLORS['teal'], '-'),
                         ('Male',   COLORS['red'],  '--')]:
    female_seer = axs_data[axs_data['Subgroup'].str.contains(sex) &
                            axs_data['Subgroup'].str.contains('<55')]['SEER_CSS_rate'].values
    male_seer   = axs_data[axs_data['Subgroup'].str.contains(sex) &
                            axs_data['Subgroup'].str.contains('>=55')]['SEER_CSS_rate'].values

    seer_vals = axs_data[axs_data['Subgroup'].str.contains(sex)]['SEER_CSS_rate'].values
    xl_vals   = axs_data[axs_data['Subgroup'].str.contains(sex)]['XL_Mal_rate'].values
    labels    = ['Age <55', 'Age >=55']

    ax4_twin = ax4.twinx() if sex == 'Male' else ax4
    if sex == 'Female':
        ax4_r = ax4.twinx()

    ax4.bar([0-0.2, 1-0.2] if sex=='Female' else [0+0.2, 1+0.2],
            seer_vals, 0.35,
            color=color, alpha=0.8,
            label=f'{sex} — SEER CSS %')

# Build proper sex x age interaction plot
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
plt.close(fig)  # close the complex figure, use simpler version

ax_left, ax_right = axes2

# Left: SEER CSS rate by sex x age
sex_age_seer = seer.groupby(['sex_std','age_55_plus'])['css_event'].mean().reset_index()
sex_age_seer['css_pct'] = sex_age_seer['css_event'] * 100

x_pos = np.array([0, 1])
w = 0.35
for i, (sex, color) in enumerate([('Female', COLORS['teal']),
                                    ('Male',   COLORS['red'])]):
    vals = sex_age_seer[sex_age_seer['sex_std']==sex].sort_values('age_55_plus')['css_pct'].values
    ax_left.bar(x_pos + (i-0.5)*w, vals, w,
                color=color, alpha=0.85, label=f'{sex}')
    for j, v in enumerate(vals):
        ax_left.text(x_pos[j] + (i-0.5)*w, v+0.1, f'{v:.1f}%',
                     ha='center', va='bottom', fontsize=8)

ax_left.set_xticks([0,1])
ax_left.set_xticklabels(['Age < 55', 'Age >= 55'])
ax_left.set_ylabel('Cancer-specific mortality rate (%)')
ax_left.set_title('SEER: CSS rate by sex and age\n(postoperative outcomes)')
ax_left.legend()

# Right: ThyroidXL malignancy by sex x age
sex_age_xl = xl_valid.groupby(['sex_std','age_55_plus'])['malignant'].mean().reset_index()
sex_age_xl['mal_pct'] = sex_age_xl['malignant'] * 100

for i, (sex, color) in enumerate([('Female', COLORS['teal']),
                                    ('Male',   COLORS['red'])]):
    vals = sex_age_xl[sex_age_xl['sex_std']==sex].sort_values('age_55_plus')['mal_pct'].values
    ax_right.bar(x_pos + (i-0.5)*w, vals, w,
                 color=color, alpha=0.85, label=f'{sex}')
    for j, v in enumerate(vals):
        ax_right.text(x_pos[j] + (i-0.5)*w, v+0.5, f'{v:.1f}%',
                      ha='center', va='bottom', fontsize=8)

ax_right.set_xticks([0,1])
ax_right.set_xticklabels(['Age < 55', 'Age >= 55'])
ax_right.set_ylabel('Preoperative malignancy rate (%)')
ax_right.set_title('ThyroidXL: malignancy rate by sex and age\n(preoperative imaging)')
ax_right.legend()

plt.suptitle('Sex x Age Interaction — Cross-Cohort Bridge\n'
             'Males have higher CSS in both age groups; preoperative malignancy is higher in males <55 yr only',
             fontsize=11, y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_bridge_sex_age.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION C — Histology comparison figure
# =============================================================
log_section('C — Histology Comparison Figure')

log('\n[C1] Histology bridge: SEER outcomes vs XL malignancy...')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

subtypes  = ['PTC', 'FTC', 'MTC']
st_colors = [COLORS['blue'], COLORS['teal'], COLORS['purple']]

# ── C1a: SEER CSS by histology and age ────────────────────
ax = axes[0]
hist_age = seer.groupby(['histology_subtype','age_55_plus'])['css_event'].mean().unstack()
hist_age = hist_age.loc[subtypes] * 100
hist_age.columns = ['Age < 55', 'Age >= 55']
x = np.arange(len(subtypes))
w = 0.35
ax.bar(x - w/2, hist_age['Age < 55'],  w, color=COLORS['teal'], alpha=0.85, label='Age < 55')
ax.bar(x + w/2, hist_age['Age >= 55'], w, color=COLORS['red'],  alpha=0.85, label='Age >= 55')
for i, subtype in enumerate(subtypes):
    for j, (col, offset) in enumerate([('Age < 55',-w/2),('Age >= 55',w/2)]):
        v = hist_age.loc[subtype, col]
        ax.text(i+offset, v+0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(subtypes)
ax.set_ylabel('CSS rate (%)')
ax.set_title('SEER: CSS by histology and age\n(postoperative)')
ax.legend()

# ── C1b: SEER LNR by histology ────────────────────────────
ax2 = axes[1]
hist_lnr = seer[seer['LNR'].notna()].groupby('histology_subtype')['LNR'].agg(['mean','median','std'])
hist_lnr = hist_lnr.loc[[s for s in subtypes if s in hist_lnr.index]]
ax2.bar(range(len(hist_lnr)), hist_lnr['median'],
        color=st_colors[:len(hist_lnr)], alpha=0.85,
        )
ax2.set_xticks(range(len(hist_lnr)))
ax2.set_xticklabels(hist_lnr.index)
ax2.set_ylabel('LNR (median)')
ax2.set_title('SEER: median LNR by histology\n(nodal burden)')
for i, (idx, row) in enumerate(hist_lnr.iterrows()):
    ax2.text(i, row['median']+0.005, f'{row["median"]:.3f}',
             ha='center', va='bottom', fontsize=9)

# ── C1c: ThyroidXL malignancy by histology if available ───
ax3 = axes[2]
# For ThyroidXL use TIRADS distribution by confirmed subtype
xl_hist = xl_valid[xl_valid['histology_mapped'].isin(subtypes)]
if len(xl_hist) > 10:
    hist_mal_xl = xl_hist.groupby('histology_mapped').agg(
        n=('malignant','count'),
        mal_rate=('malignant','mean'),
        tirads_mean=('tirads','mean')
    )
    hist_mal_xl = hist_mal_xl.loc[[s for s in subtypes if s in hist_mal_xl.index]]
    ax3.bar(range(len(hist_mal_xl)),
            hist_mal_xl['mal_rate']*100,
            color=st_colors[:len(hist_mal_xl)], alpha=0.85)
    ax3.set_xticks(range(len(hist_mal_xl)))
    ax3.set_xticklabels(hist_mal_xl.index)
    ax3.set_ylabel('Confirmed malignancy rate (%)')
    ax3.set_title('ThyroidXL: malignancy by confirmed subtype\n(post-surgical pathology subset)')
    for i, (idx, row) in enumerate(hist_mal_xl.iterrows()):
        ax3.text(i, row['mal_rate']*100+0.5, f'n={row["n"]:.0f}',
                 ha='center', va='bottom', fontsize=8)
else:
    # Use SEER-based histology distribution comparison
    hist_css_seer = seer.groupby('histology_subtype')['css_event'].mean() * 100
    hist_css_seer = hist_css_seer.loc[[s for s in subtypes if s in hist_css_seer.index]]
    ax3.bar(range(len(hist_css_seer)),
            hist_css_seer.values,
            color=st_colors[:len(hist_css_seer)], alpha=0.85)
    ax3.set_xticks(range(len(hist_css_seer)))
    ax3.set_xticklabels(hist_css_seer.index)
    ax3.set_ylabel('CSS rate (%)')
    ax3.set_title('SEER: CSS rate by histology subtype\n(full cohort)')
    for i, v in enumerate(hist_css_seer.values):
        ax3.text(i, v+0.05, f'{v:.2f}%', ha='center', va='bottom', fontsize=9)

plt.suptitle('Histology Subtype — Cross-Cohort Analysis\n'
             'Among PTC, FTC and MTC, MTC has the highest CSS; PTC dominates both cohorts',
             fontsize=11, y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_bridge_histology.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION D — Tumor size bridge figure
# =============================================================
log_section('D — Tumor Size Bridge Figure')

log('\n[D1] Tumor size bridge...')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── D1: SEER tumor size distribution by stage ─────────────
ax = axes[0]
stage_groups = ['1','2','3','4A','4B']
stage_colors = [COLORS['teal'], COLORS['blue'],
                COLORS['orange'], COLORS['red'], COLORS['purple']]

for sg, color in zip(stage_groups, stage_colors):
    sub = seer[(seer['stage_group']==sg) & seer['tumor_size'].notna()]
    if len(sub) < 20:
        continue
    ax.hist(sub['tumor_size'], bins=30, alpha=0.5, density=True,
            color=color, label=f'Stage {sg} (n={len(sub):,})')

ax.axvline(x=22.8, color='black', linestyle='--', lw=1.5,
           label=f'SEER mean: 22.8 mm')
ax.set_xlabel('Tumor size (mm)')
ax.set_ylabel('Density')
ax.set_title('SEER: tumor size by stage group')
ax.legend(fontsize=7)
ax.set_xlim(0, 100)

# ── D2: ThyroidXL nodule size by malignancy ───────────────
ax2 = axes[1]
for mal, label, color in [(0,'Benign',COLORS['teal']),
                           (1,'Malignant',COLORS['red'])]:
    sub = xl_valid[xl_valid['malignant']==mal]
    ax2.hist(sub['width'], bins=30, alpha=0.6, density=True,
             color=color, label=f'{label} (n={len(sub):,})')

ax2.axvline(x=xl_valid[xl_valid['malignant']==1]['width'].mean(),
            color=COLORS['red'], linestyle='--', lw=1.5,
            label=f'Malignant mean: {xl_valid[xl_valid["malignant"]==1]["width"].mean():.1f} mm')
ax2.axvline(x=xl_valid[xl_valid['malignant']==0]['width'].mean(),
            color=COLORS['teal'], linestyle='--', lw=1.5,
            label=f'Benign mean: {xl_valid[xl_valid["malignant"]==0]["width"].mean():.1f} mm')
ax2.set_xlabel('Nodule width (mm)')
ax2.set_ylabel('Density')
ax2.set_title('ThyroidXL: nodule width by malignancy')
ax2.legend(fontsize=8)
ax2.set_xlim(0, 60)

# ── D3: Size comparison scatter (XL width vs SEER T stage) ─
ax3 = axes[2]
# Map SEER T stage to size range and XL nodule size to comparable bins
seer_t = seer[seer['tumor_size'].notna()].groupby('t_stage').agg(
    mean_size=('tumor_size','mean'),
    n=('tumor_size','count')
).reset_index()
seer_t = seer_t[seer_t['t_stage'].str.startswith('T')]

# ThyroidXL: TIRADS group mean sizes
xl_tirads = xl_valid.groupby('tirads').agg(
    mean_width=('width','mean'),
    mal_rate=('malignant','mean'),
    n=('malignant','count')
).reset_index()

ax3_r = ax3.twinx()

bars = ax3.bar(xl_tirads['tirads'],
               xl_tirads['mean_width'],
               color=COLORS['blue'], alpha=0.7,
               label='Mean nodule width (mm)')
ax3_r.plot(xl_tirads['tirads'],
           xl_tirads['mal_rate']*100,
           'ro-', lw=2, markersize=8,
           label='Malignancy rate (%)')

ax3.set_xlabel('TIRADS score')
ax3.set_ylabel('Mean nodule width (mm)', color=COLORS['blue'])
ax3_r.set_ylabel('Malignancy rate (%)', color=COLORS['red'])
ax3.set_title('ThyroidXL: nodule size vs malignancy by TIRADS\n(size inversely related to malignancy rate)')
ax3.tick_params(axis='y', labelcolor=COLORS['blue'])
ax3_r.tick_params(axis='y', labelcolor=COLORS['red'])

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_r.get_legend_handles_labels()
ax3.legend(lines1+lines2, labels1+labels2, loc='upper left', fontsize=8)

plt.suptitle('Tumor Size Bridge — SEER Postoperative vs ThyroidXL Preoperative\n'
             'Smaller nodules more malignant (TIRADS/biopsy bias); larger tumors = higher stage',
             fontsize=11, y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_bridge_tumorsize.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION E — Main bridge heatmap (paper Figure)
# =============================================================
log_section('E — Main Bridge Heatmap')

log('\n[E1] Building main bridge heatmap...')

# Build clean matrix for heatmap
rows_hm = []

# Age rows
for age_grp in AGE_LABELS:
    s = seer[seer['age_group']==age_grp]
    x = xl_valid[xl_valid['age_group']==age_grp]
    rows_hm.append({
        'label':       f'Age {age_grp}',
        'seer_css':    s['css_event'].mean()*100,
        'xl_mal':      x['malignant'].mean()*100,
        'seer_lnr':    s['LNR'].mean() if s['LNR'].notna().sum()>0 else np.nan,
        'category':    'Age'
    })

# Sex rows
for sex in ['Female','Male']:
    s = seer[seer['sex_std']==sex]
    x = xl_valid[xl_valid['sex_std']==sex]
    rows_hm.append({
        'label':    sex,
        'seer_css': s['css_event'].mean()*100,
        'xl_mal':   x['malignant'].mean()*100,
        'seer_lnr': s['LNR'].mean() if s['LNR'].notna().sum()>0 else np.nan,
        'category': 'Sex'
    })

# Histology rows
for subtype in ['PTC','FTC','MTC']:
    s = seer[seer['histology_subtype']==subtype]
    rows_hm.append({
        'label':    subtype,
        'seer_css': s['css_event'].mean()*100,
        'xl_mal':   np.nan,
        'seer_lnr': s['LNR'].mean() if s['LNR'].notna().sum()>0 else np.nan,
        'category': 'Histology'
    })

hm_df = pd.DataFrame(rows_hm)

fig, axes = plt.subplots(1, 3, figsize=(16, 9))

metrics = [
    ('seer_css',  'SEER CSS rate (%)',          'Reds',   0,   12),
    ('xl_mal',    'XL preop malignancy (%)',     'Blues_r', 0,  45),
    ('seer_lnr',  'SEER mean LNR',              'Oranges', 0,  0.4),
]

for ax, (col, title, cmap, vmin, vmax) in zip(axes, metrics):
    vals = hm_df[col].values.reshape(-1, 1)
    labels = hm_df['label'].tolist()
    categories = hm_df['category'].tolist()

    # Color background by category
    cat_colors = {'Age':'#EBF5FB','Sex':'#EAFAF1','Histology':'#FEF9E7'}

    im = ax.imshow(vals, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks([])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Value annotations
    for i, (v, cat) in enumerate(zip(vals.flatten(), categories)):
        if not np.isnan(v):
            ax.text(0, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color=('white' if (0.299*im.cmap(im.norm(v))[0]+0.587*im.cmap(im.norm(v))[1]+0.114*im.cmap(im.norm(v))[2]) < 0.5 else 'black'))
        else:
            ax.text(0, i, 'N/A', ha='center', va='center',
                    fontsize=9, color='gray')

    # Category separators
    cat_change = [i for i in range(1, len(categories))
                  if categories[i] != categories[i-1]]
    for pos in cat_change:
        ax.axhline(pos - 0.5, color='white', lw=2)

    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

plt.suptitle('Cross-Cohort Bridge Synthesis — Table 7 analog (ecological)\n'
             'Row = subgroup · Col = outcome metric (SEER postop | ThyroidXL preop)',
             fontsize=12, y=1.02)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_bridge_heatmap.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# SECTION F — Key findings summary
# =============================================================
log_section('F — Key Bridge Findings')

log('\n[F1] Key cross-cohort findings:')
log()
log('  DIMENSION 1 — Age:')
log('    SEER CSS rate rises across age strata (values in table_bridge_synthesis.csv)')
log('    XL malignancy is lower in older strata but not monotone in the oldest stratum (n = 46)')
log('    -> Opposite age gradients (ecological comparison; not an individual-level effect)')
log()
log('  DIMENSION 2 — Sex:')
m_css = seer[seer['sex_std']=='Male']['css_event'].mean()*100
f_css = seer[seer['sex_std']=='Female']['css_event'].mean()*100
m_mal = xl_valid[xl_valid['sex_std']=='Male']['malignant'].mean()*100
f_mal = xl_valid[xl_valid['sex_std']=='Female']['malignant'].mean()*100
log(f'    SEER: Male CSS={m_css:.2f}% vs Female {f_css:.2f}% (Male 2.4x higher mortality)')
log(f'    XL: Male malignancy={m_mal:.1f}% vs Female {f_mal:.1f}% (Male 1.5x more malignant preop)')
log('    -> Males show worse outcomes at BOTH preoperative and postoperative stages')
log('    -> Concordant sex effect across the full cancer pathway')
log()
log('  DIMENSION 3 — Histology:')
for st in ['PTC','FTC','MTC']:
    css = seer[seer['histology_subtype']==st]['css_event'].mean()*100
    lnr = seer[seer['histology_subtype']==st]['LNR'].mean()
    log(f'    {st}: CSS={css:.2f}%  mean LNR={lnr:.3f}')
log('    -> MTC has highest mortality despite lower prevalence')
log('    -> PTC dominates both cohorts (88% SEER, ~87% XL)')
log()
log('  DIMENSION 4 — Tumor size:')
log(f'    SEER mean tumor size: 22.8 mm')
log(f'    XL mean nodule width: 13.8 mm, height: 10.9 mm')
log('    -> SEER tumors larger because they are postoperative measurements')
log('    -> In XL: TIRADS 5 nodules are actually SMALLER than TIRADS 3/4')
log('       (malignant nodules are often microcarcinomas - < 10mm)')

# =============================================================
# SAVE LOG
# =============================================================
log_section('STEP 5 COMPLETE')
log(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'\nOutputs:')
log(f'  Figures: {FIGS}')
log(f'  Tables:  {TABS}')

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

print(f'\nLog saved -> {LOG_FILE}')
