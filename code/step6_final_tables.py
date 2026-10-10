"""
=============================================================
STEP 6 — Final Paper Tables, PRISMA Diagram & Publication Polish
Thyroid Cancer Two-Cohort Study
=============================================================
Produces all submission-ready outputs:
  Table 1  — Patient characteristics (both cohorts)
  Table 2  — LNR vs LODDS model performance
  Table 3  — Age threshold models comparison
  Table 4  — Preoperative prediction model performance
  Table 5  — Cross-cohort bridge synthesis (Table 12 analog)
  Figure   — PRISMA flow diagram
  Figure   — Combined figure panel (summary)

Uses all previously generated tables and data.

Outputs:
  outputs/tables/Table1_patient_characteristics.csv
  outputs/tables/Table2_lnr_lodds_performance.csv
  outputs/tables/Table3_age_threshold_models.csv
  outputs/tables/Table4_preop_model_performance.csv
  outputs/tables/Table5_bridge_synthesis.csv
  outputs/figures/fig_prisma_flow.png
  outputs/figures/fig_summary_panel.png
  logs/step6_log.txt
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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

LOG_FILE = os.path.join(LOGS, 'step6_log.txt')

log_lines = []
def log(msg=''):
    print(msg)
    log_lines.append(str(msg))

def log_section(title):
    line = '=' * 60
    log(f'\n{line}')
    log(f'  {title}')
    log(line)

plt.rcParams.update({
    'figure.dpi':        150,
    'font.family':       'DejaVu Sans',
    'font.size':         10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
})

COLORS = {
    'blue':   '#2563EB', 'teal':   '#0D9488',
    'orange': '#EA580C', 'red':    '#DC2626',
    'purple': '#7C3AED', 'gray':   '#6B7280',
    'green':  '#16A34A',
}

# =============================================================
log_section('STEP 6 — Final Paper Tables & Figures')
log(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# ── Load data ──────────────────────────────────────────────
log('\n[1] Loading all datasets and prior tables...')
seer = pd.read_csv(os.path.join(PROC, 'seer_clean.csv'))
xl   = pd.read_csv(os.path.join(PROC, 'thyroidxl_clean.csv'))
xl['age'] = pd.to_numeric(xl['age'], errors='coerce')
xl_valid  = xl[xl['age'].notna() & xl['age'].between(10, 100)].copy()
xl_valid['sex_std'] = xl_valid['gender'].map({1:'Male', 2:'Female'})

# =============================================================
# TABLE 1 — Patient characteristics
# =============================================================
log_section('TABLE 1 — Patient Characteristics')

rows = []

def add_row(var, seer_val, xl_val, notes=''):
    rows.append({'Variable': var, 'SEER (n=61,615)': seer_val,
                 'ThyroidXL (n=4,088)': xl_val, 'Notes': notes})

def pct(n, total): return f'{n:,} ({n/total*100:.1f}%)'

N_S = len(seer)
N_X = len(xl_valid)

add_row('N (total)',          f'{N_S:,}',                    f'{N_X:,}',             'Complete cases with age')
add_row('--- Demographics ---','','','')
add_row('Female',             pct((seer['sex']=='Female').sum(), N_S),
                              pct((xl_valid['gender']==2).sum(), N_X),            '')
add_row('Male',               pct((seer['sex']=='Male').sum(), N_S),
                              pct((xl_valid['gender']==1).sum(), N_X),            '')
add_row('Age median (IQR)',   f'52 (37–62)',                 f'48 (38–58)',           'Years')
add_row('Age < 55',           pct((seer['age_55_plus']==0).sum(), N_S),
                              pct((xl_valid['age_55_plus']==0).sum(), N_X),       '')
add_row('Age >= 55',          pct((seer['age_55_plus']==1).sum(), N_S),
                              pct((xl_valid['age_55_plus']==1).sum(), N_X),       '')
add_row('--- Histology ---',  '','','')
add_row('PTC',                pct((seer['histology_subtype']=='PTC').sum(), N_S),
                              '~87% (pathology-confirmed subset)', 'ICD-O-3 coded')
add_row('FTC',                pct((seer['histology_subtype']=='FTC').sum(), N_S),
                              '—',                               'ICD-O-3 coded')
add_row('MTC',                pct((seer['histology_subtype']=='MTC').sum(), N_S),
                              '—',                               'ICD-O-3 coded')
add_row('Other/ATC',          pct((seer['histology_subtype'].isin(['ATC','Other'])).sum(), N_S),
                              '—',                               '')
add_row('--- SEER staging ---','','','')
add_row('Stage I',            pct((seer['stage_group']=='1').sum(), N_S),   '—',   'EOD 2018')
add_row('Stage II',           pct((seer['stage_group']=='2').sum(), N_S),   '—',   '')
add_row('Stage III',          pct((seer['stage_group']=='3').sum(), N_S),   '—',   '')
add_row('Stage IV (any)',      pct(seer['stage_group'].isin(['4','4A','4B','4C']).sum(), N_S), '—', '')
add_row('N0a',                pct((seer['n_stage']=='N0a').sum(), N_S),     '—',   '')
add_row('N1a',                pct((seer['n_stage']=='N1a').sum(), N_S),     '—',   '')
add_row('N1b',                pct((seer['n_stage']=='N1b').sum(), N_S),     '—',   '')
add_row('--- Tumor size ---',  '','','')
add_row('Tumor size median (mm)', f'16.0 (valid n=58,512)', f'Width 11.0 / Height 9.0', 'mm')
add_row('--- Nodal metrics (SEER) ---','','','')
add_row('Valid node data',    pct(seer['LNR'].notna().sum(), N_S),          '—',   'Nodes examined < 98')
add_row('LNR median (IQR)',   '0.000 (0.000–0.400)',                        '—',   'NPLN/D')
add_row('LODDS median (IQR)', '-1.099 (-2.197–0.405)',                      '—',   'ln((NPLN+0.5)/(D-NPLN+0.5))')
add_row('--- ThyroidXL imaging ---','','','')
add_row('TIRADS 1–2',         '—', pct((xl_valid['tirads']<=2).sum(), N_X),  '')
add_row('TIRADS 3',           '—', pct((xl_valid['tirads']==3).sum(), N_X),  '')
add_row('TIRADS 4',           '—', pct((xl_valid['tirads']==4).sum(), N_X),  '')
add_row('TIRADS 5',           '—', pct((xl_valid['tirads']==5).sum(), N_X),  '')
add_row('Malignant (confirmed)','—', pct(xl_valid['malignant'].sum(), N_X),   'Histopathology/FNAC')
add_row('--- Outcomes ---','','','')
add_row('CSS events',         pct(seer['css_event'].sum(), N_S),            '—',   '2.28% cancer-specific death')
add_row('OS events',          pct(seer['os_event'].sum(), N_S),             '—',   '5.09% all-cause death')
add_row('Survival months median', '26',                                     '—',   'Max 59 months')
add_row('Total thyroidectomy', pct((seer['surgery_type']=='Total thyroidectomy').sum(), N_S), '—', '')
add_row('RAI received',        pct(seer['rai'].sum(), N_S),                 '—',   '29.0%')

tbl1 = pd.DataFrame(rows)
p1   = os.path.join(TABS, 'Table1_patient_characteristics.csv')
tbl1.to_csv(p1, index=False, encoding='utf-8')
log(f'  Saved -> {p1}')

# =============================================================
# TABLE 2 — LNR vs LODDS model performance
# =============================================================
log_section('TABLE 2 — LNR vs LODDS Model Performance')

# Load from prior output
lnr_perf = pd.read_csv(os.path.join(TABS, 'table_lnr_lodds_performance.csv'))

# Enrich with HR values from log
hr_data = {
    'N stage (AJCC baseline)': {'HR': '—', 'p': '—', 'Predictor': 'N stage categories'},
    'LNR + age':               {'HR': '1.469 per unit (ridge-penalised)', 'p': '<0.001', 'Predictor': 'LNR + age'},
    'LODDS + age':             {'HR': '1.115 per unit (ridge-penalised)', 'p': '<0.001', 'Predictor': 'LODDS + age'},
    'LNR + LODDS + age':       {'HR': 'Combined',       'p': '<0.001', 'Predictor': 'LNR + LODDS + age'},
    'LNR+LODDS+age (PTC only)':{'HR': '—',              'p': '—',      'Predictor': 'PTC subgroup'},
    'LNR+LODDS+age (N1a)':     {'HR': '—',              'p': '—',      'Predictor': 'N1a subgroup'},
    'LNR+LODDS+age (N1b)':     {'HR': '—',              'p': '—',      'Predictor': 'N1b subgroup'},
}

lnr_perf['HR (key predictor)'] = lnr_perf['Model'].map(
    lambda m: hr_data.get(m, {}).get('HR', '—')
)
lnr_perf['p-value'] = lnr_perf['Model'].map(
    lambda m: hr_data.get(m, {}).get('p', '—')
)
lnr_perf['Endpoint'] = 'CSS (cancer-specific survival)'
lnr_perf['Validation'] = 'Apparent C-index (resubstitution); bootstrap 95% CI, 300 iterations'

p2 = os.path.join(TABS, 'Table2_lnr_lodds_performance.csv')
lnr_perf.to_csv(p2, index=False, encoding='utf-8')
log(f'  Saved -> {p2}')
log(lnr_perf[['Model','C-index','HR (key predictor)','p-value']].to_string(index=False))

# =============================================================
# TABLE 3 — Age threshold models
# =============================================================
log_section('TABLE 3 — Age Threshold Models')

age_df = pd.read_csv(os.path.join(TABS, 'table_age_threshold_models.csv'))

# Add interpretation column
def interpret_age_model(row):
    if 'Binary' in row['Model'] and 'stage' not in row['Model'].lower():
        return 'AJCC-8 approach: single 55-yr cutoff'
    elif 'Continuous' in row['Model'] and 'stage' not in row['Model'].lower():
        return 'Linear continuous age — captures gradient'
    elif 'stage' in row['Model'].lower():
        return 'Age + EOD stage — clinical full model'
    elif 'binary' in row['Model'].lower():
        return 'Subtype-stratified binary cutoff'
    else:
        return 'Subtype-stratified continuous'

age_df['Interpretation'] = age_df.apply(interpret_age_model, axis=1)
age_df['Outcome'] = 'CSS'
# Delta = C-index gain of continuous over binary age within the SAME specification / subtype
# (the submitted version subtracted the all-patient binary C-index 0.7378 from every row, which is not a like-for-like comparison).
_pair = {'Continuous age (linear)': 'Binary age (55yr cutoff)', 'Continuous age + stage': 'Binary age + stage',
         'PTC — continuous age': 'PTC — binary age', 'FTC — continuous age': 'FTC — binary age', 'MTC — continuous age': 'MTC — binary age'}
_c = age_df.set_index('Model')['C-index']
age_df['Delta vs binary'] = age_df['Model'].map(lambda m: round(_c[m] - _c[_pair[m]], 4) if m in _pair else '—')

p3 = os.path.join(TABS, 'Table3_age_threshold_models.csv')
age_df.to_csv(p3, index=False, encoding='utf-8')
log(f'  Saved -> {p3}')
log(age_df[['Model','C-index','HR_age','p_age','Delta vs binary']].to_string(index=False))

# =============================================================
# TABLE 4 — Preoperative prediction model performance
# =============================================================
log_section('TABLE 4 — Preoperative Model Performance')

perf_df = pd.read_csv(os.path.join(TABS, 'table_model_performance.csv'))

# Add TIRADS baseline comparison
# Delta is taken against the calibrated TIRADS reference of the same feature set (as in step4 / Table 5);
# the submitted version subtracted a hard-coded 0.9141 (raw ordinal-score AUC), giving +0.018-0.020 instead of +0.022-0.024.
_ref = perf_df[perf_df['Model'] == 'TIRADS alone (baseline)'].set_index('Feature set')['CV AUC']
perf_df['Delta AUC vs TIRADS'] = perf_df.apply(
    lambda r: round(r['CV AUC'] - _ref[r['Feature set']], 4)
              if 'TIRADS' not in r['Model'] else '—', axis=1
)
perf_df['Validation method'] = '5-fold stratified CV; 1,000-resample bootstrap of out-of-fold predictions'
perf_df.loc[perf_df['Model']=='TIRADS alone (baseline)', 'Validation method'] = 'Ordinal score (no training)'

p4 = os.path.join(TABS, 'Table4_preop_model_performance.csv')
perf_df.to_csv(p4, index=False, encoding='utf-8')
log(f'  Saved -> {p4}')
log(perf_df[['Feature set','Model','CV AUC','Brier score','Delta AUC vs TIRADS']].to_string(index=False))

# =============================================================
# TABLE 5 — Bridge synthesis (formatted)
# =============================================================
log_section('TABLE 5 — Cross-Cohort Bridge Synthesis')

bridge_raw = pd.read_csv(os.path.join(TABS, 'table_bridge_synthesis.csv'))

# Build clean formatted version
tbl5_rows = []
for _, row in bridge_raw.iterrows():
    direction = ''
    if row['Dimension'] == 'Age':
        direction = 'Opposite (CSS rises, malignancy falls)'
    elif row['Dimension'] == 'Sex':
        direction = 'Concordant (males worse at both stages)'
    elif row['Dimension'] == 'Histology':
        direction = 'SEER-only (insufficient XL pathology n)'
    else:
        direction = 'Combined interaction'

    tbl5_rows.append({
        'Dimension':           row['Dimension'],
        'Subgroup':            row['Subgroup'],
        'SEER n':              f'{row["SEER_n"]:,}',
        'SEER CSS rate (%)':   row['SEER_CSS_rate'],
        'ThyroidXL n':         f'{int(row["XL_n"]):,}' if not pd.isna(row['XL_n']) else '—',
        'XL malignancy (%)':   row['XL_Mal_rate'] if not pd.isna(row['XL_Mal_rate']) else '—',
        'Effect direction':    direction,
    })

tbl5 = pd.DataFrame(tbl5_rows)
p5 = os.path.join(TABS, 'Table5_bridge_synthesis.csv')
tbl5.to_csv(p5, index=False, encoding='utf-8')
log(f'  Saved -> {p5}')
log(tbl5.to_string(index=False))

# =============================================================
# FIGURE — PRISMA flow diagram
# =============================================================
log_section('FIGURE — PRISMA Flow Diagram')

log('\n[P1] Building PRISMA flow diagram...')
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#EBF5FB', border='#2563EB',
             fontsize=9, bold=False):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle='round,pad=0.1',
                         facecolor=color, edgecolor=border, lw=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, color='#6B7280'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# ── Left column: SEER ─────────────────────────────────────
ax.text(3.5, 9.7, 'Cohort B — SEER Registry', ha='center',
        fontsize=11, fontweight='bold', color='#2563EB')
draw_box(ax, 0.5, 8.8, 6.0, 0.7,
         'SEER thyroid cancer records 2018–2022\nn = 61,615 patients',
         color='#EBF5FB', border='#2563EB', bold=True)
draw_arrow(ax, 3.5, 8.8, 3.5, 7.9)
draw_box(ax, 0.5, 7.2, 6.0, 0.6,
         'Excluded: laterality "not paired site"\nn = 0 (all retained for analysis)',
         color='#FEF3C7', border='#D97706', fontsize=8)
draw_box(ax, 0.5, 6.3, 6.0, 0.7,
         'Sentinel codes excluded (nodes 98/99)\nValid node data: n = 33,488 (54.4%)',
         color='#EBF5FB', border='#2563EB')
draw_arrow(ax, 3.5, 6.3, 3.5, 5.4)
draw_box(ax, 0.5, 4.7, 6.0, 0.6,
         'Excluded: NPLN > D (n = 29); unknown survival (n = 2)\nNodal cohort: n = 33,457',
         color='#FEF3C7', border='#D97706', fontsize=8)
draw_box(ax, 0.5, 3.8, 6.0, 0.7,
         'Nodal cohort (LNR + LODDS + survival)\nn = 33,457',
         color='#EBF5FB', border='#2563EB')
draw_arrow(ax, 3.5, 3.8, 3.5, 2.9)
draw_box(ax, 0.5, 2.2, 6.0, 0.6,
         'Unknown survival (141) -> NaN\nFinal analysis N = 61,474 (survival)',
         color='#FEF3C7', border='#D97706', fontsize=8)
draw_box(ax, 0.5, 1.3, 6.0, 0.7,
         'Full cohort n = 61,615 | survival cohort 61,474\nnodal cohort 33,457',
         color='#DCFCE7', border='#16A34A', bold=True)

# ── Right column: ThyroidXL ───────────────────────────────
ax.text(10.5, 9.7, 'Cohort A — ThyroidXL Dataset', ha='center',
        fontsize=11, fontweight='bold', color='#0D9488')
draw_box(ax, 7.5, 8.8, 6.0, 0.7,
         'ThyroidXL JSON dataset\nn = 4,093 patients (11,545 US images)',
         color='#F0FDF4', border='#0D9488', bold=True)
draw_arrow(ax, 10.5, 8.8, 10.5, 7.9)
draw_box(ax, 7.5, 7.2, 6.0, 0.6,
         'Excluded: age outliers (birth year entries)\nn = 39 corrected; n = 5 set to NaN',
         color='#FEF3C7', border='#D97706', fontsize=8)
draw_box(ax, 7.5, 6.3, 6.0, 0.7,
         'Valid age: n = 4,088\nMalignant: 1,229 (30.1%) | Non-malignant: 2,859 (incl. 94 suspicious)',
         color='#F0FDF4', border='#0D9488')
draw_arrow(ax, 10.5, 6.3, 10.5, 5.4)
draw_box(ax, 7.5, 4.7, 6.0, 0.6,
         'FNAC missing: n = 146 (3.6%) -> Model Set A\nComplete FNAC: n = 3,942 -> Model Set B',
         color='#FEF3C7', border='#D97706', fontsize=8)
draw_box(ax, 7.5, 3.8, 6.0, 0.7,
         'Preop model Set A (no FNAC): n = 4,088\nPreop model Set B (with FNAC): n = 3,942',
         color='#F0FDF4', border='#0D9488')
draw_arrow(ax, 10.5, 3.8, 10.5, 2.9)
draw_box(ax, 7.5, 2.2, 6.0, 0.6,
         'Suspicious (n = 94) retained, coded non-malignant\nHistopathology available: n = 798',
         color='#FEF3C7', border='#D97706', fontsize=8)
draw_box(ax, 7.5, 1.3, 6.0, 0.7,
         'Final ThyroidXL analysis cohort\nn = 4,088 (model A) | 3,942 (model B)',
         color='#DCFCE7', border='#16A34A', bold=True)

# ── Bridge arrow ──────────────────────────────────────────
draw_box(ax, 4.8, 0.2, 4.4, 0.8,
         'Ecological stratum-level comparison\nAge + Sex + Histology (cohorts not linked)',
         color='#F3E8FF', border='#7C3AED', bold=True, fontsize=9)
draw_arrow(ax, 3.5, 1.3, 5.5, 1.0)
draw_arrow(ax, 10.5, 1.3, 8.5, 1.0)

ax.set_title('PRISMA-Aligned Study Selection and Data Harmonization Flow\n'
             'Thyroid Cancer Two-Cohort Risk Stratification Study',
             fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_prisma_flow.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# FIGURE — Summary panel (key results at a glance)
# =============================================================
log_section('FIGURE — Summary Results Panel')

log('\n[S1] Building summary results panel...')
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

# ── Panel 1: LNR KM (load from saved) ─────────────────────
# Re-create compact versions of key figures

# Panel 1: C-index comparison bar chart
ax1 = fig.add_subplot(gs[0, 0])
models = ['Ordinal N\n(1 df)', 'LNR\n+age', 'LODDS\n+age', 'LNR+LODDS\n+age']
_ci = lnr_perf.set_index('Model')['C-index']   # read from step2 output (ridge-penalised, apparent)
cidx   = [float(_ci['Ordinal N-stage (1 df)']), float(_ci['LNR + age']), float(_ci['LODDS + age']), float(_ci['LNR + LODDS + age'])]
cidx   = [round(x + 1e-9, 6) for x in cidx]   # table values are stored to 4 decimals; avoid a binary-float tie when printing 3 decimals
colors_b = [COLORS['purple'], COLORS['blue'], COLORS['teal'], COLORS['orange']]
bars = ax1.bar(models, cidx, color=colors_b, alpha=0.85, width=0.6)
ax1.set_ylim(0.82, 0.90)
ax1.set_ylabel('C-index (CSS)')
ax1.set_title('Nodal burden models\nvs AJCC N stage', fontsize=9, fontweight='bold')
for bar, val in zip(bars, cidx):
    ax1.text(bar.get_x()+bar.get_width()/2, val+0.001,
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# Panel 2: Age effect gradient
ax2 = fig.add_subplot(gs[0, 1])
age_grps = ['<35','35-44','45-54','55-64','65-74','75+']
seer_css  = [0.09, 0.34, 0.74, 2.03, 4.32, 12.47]
xl_mal    = [41.2, 35.2, 27.4, 22.5, 19.6, 16.1]
x = np.arange(len(age_grps))
ax2_r = ax2.twinx()
ax2.bar(x, seer_css, color=COLORS['red'],  alpha=0.7, width=0.4,
        label='SEER CSS %', align='edge')
ax2_r.bar(x-0.4, xl_mal, color=COLORS['teal'], alpha=0.7, width=0.4,
          label='XL malignancy %', align='edge')
ax2.set_xticks(x)
ax2.set_xticklabels(age_grps, rotation=30, fontsize=7)
ax2.set_ylabel('SEER CSS %', color=COLORS['red'], fontsize=8)
ax2_r.set_ylabel('XL malignancy %', color=COLORS['teal'], fontsize=8)
ax2.set_title('Age paradox\n(opposite directions)', fontsize=9, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=COLORS['red'])
ax2_r.tick_params(axis='y', labelcolor=COLORS['teal'])

# Panel 3: ROC summary
ax3 = fig.add_subplot(gs[0, 2])
model_names = ['TIRADS\nalone', 'LR\n(no FNAC)', 'XGBoost\n(no FNAC)',
               'LR\n(+FNAC)', 'RF\n(+FNAC)']
_pa = perf_df.set_index(['Feature set', 'Model'])['CV AUC']   # read from step4 output
aucs = [float(_pa[('Without FNAC', 'TIRADS alone (baseline)')]), float(_pa[('Without FNAC', 'Logistic Regression')]), float(_pa[('Without FNAC', 'XGBoost')]), float(_pa[('With FNAC', 'Logistic Regression')]), float(_pa[('With FNAC', 'Random Forest')])]
colors_r = [COLORS['gray'], COLORS['blue'], COLORS['orange'],
            COLORS['teal'], COLORS['purple']]
bars3 = ax3.bar(model_names, aucs, color=colors_r, alpha=0.85, width=0.6)
ax3.set_ylim(0.88, 1.01)
ax3.set_ylabel('AUC (5-fold CV)')
ax3.set_title('Preop malignancy\nprediction models', fontsize=9, fontweight='bold')
for bar, val in zip(bars3, aucs):
    ax3.text(bar.get_x()+bar.get_width()/2, val+0.001,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7)
ax3.axhline(aucs[0], color=COLORS['gray'], linestyle='--', lw=1,
            alpha=0.6, label='TIRADS baseline')

# Panel 4: Sex concordance
ax4 = fig.add_subplot(gs[0, 3])
sexes   = ['Female\n(SEER)', 'Male\n(SEER)', 'Female\n(XL)', 'Male\n(XL)']
vals4   = [1.65, 4.00, 28.6, 42.0]
colors4 = [COLORS['teal'], COLORS['red'], COLORS['teal'], COLORS['red']]
bars4   = ax4.bar(sexes, vals4, color=colors4, alpha=0.85, width=0.6)
ax4.set_ylabel('Rate (%)')
ax4.set_title('Sex concordance\nacross both cohorts', fontsize=9, fontweight='bold')
for bar, val in zip(bars4, vals4):
    ax4.text(bar.get_x()+bar.get_width()/2, val+0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
ax4.set_ylim(0, 50)

# ── Bottom row: Feature importance + Bridge heatmap text ──
ax5 = fig.add_subplot(gs[1, 0:2])
features_nofnac = ['TIRADS', 'Aspect ratio', 'Width', 'Area',
                   'Height', 'Age', 'Age>=55', 'Isthmus', 'Gender', 'Left']
imps_nofnac = [0.785, 0.074, 0.041, 0.036, 0.025, 0.024, 0.006, 0.004, 0.004, 0.002]
colors5 = [COLORS['blue'] if 'TIRADS' in f else COLORS['teal'] for f in features_nofnac]
ax5.barh(range(len(features_nofnac)), imps_nofnac[::-1],
         color=colors5[::-1], alpha=0.85)
ax5.set_yticks(range(len(features_nofnac)))
ax5.set_yticklabels(features_nofnac[::-1], fontsize=9)
ax5.set_xlabel('RF importance')
ax5.set_title('Feature importance (without FNAC)\nTIRADS dominates preop prediction',
              fontsize=9, fontweight='bold')

# Panel 6: TIRADS threshold table visualization
ax6 = fig.add_subplot(gs[1, 2:4])
ax6.axis('off')
table_data = [
    ['TIRADS', 'Sensitivity', 'Specificity', 'PPV', 'NPV'],
    ['>=4', '98.5%', '57.6%', '50.0%', '98.9%'],
    ['>=5', '79.6%', '92.8%', '82.6%', '91.4%'],
    ['LR model (no FNAC)', '—', '—', f'AUC={aucs[1]:.3f}', f'vs TIRADS +{aucs[1]-aucs[0]:.3f}'],
    ['XGBoost (no FNAC)',  '—', '—', f'AUC={aucs[2]:.3f}', f'vs TIRADS +{aucs[2]-aucs[0]:.3f}'],
    ['LR model (+FNAC)',   '—', '—', f'AUC={aucs[3]:.3f}', f'vs TIRADS (Set B) +{aucs[3]-float(_pa[("With FNAC", "TIRADS alone (baseline)")]):.3f}'],
]
tbl_obj = ax6.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
tbl_obj.auto_set_font_size(False)
tbl_obj.set_fontsize(9)
for (r, c), cell in tbl_obj.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1E40AF')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#EFF6FF')
    cell.set_edgecolor('#D1D5DB')
ax6.set_title('TIRADS thresholds and model performance summary',
              fontsize=9, fontweight='bold', pad=10)

plt.suptitle('Summary of Key Results — Thyroid Cancer Two-Cohort Risk Stratification Study\n'
             'SEER 2018–2022 (n=61,615) + ThyroidXL (n=4,088)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
fig_path = os.path.join(FIGS, 'fig_summary_panel.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
log(f'  Saved -> {fig_path}')

# =============================================================
# FINAL CHECKLIST
# =============================================================
log_section('Final Output Checklist')

log('\n=== FIGURES (18 total) ===')
figs_list = [
    'fig_km_css_lnr.png',
    'fig_km_css_lodds.png',
    'fig_km_by_nstage.png',
    'fig_calibration_lnr_lodds.png',
    'fig_dca_lnr_lodds.png',
    'fig_lnr_vs_lodds_comparison.png',
    'fig_age_css_km.png',
    'fig_age_continuous_cox.png',
    'fig_age_malignancy_xl.png',
    'fig_age_histology_subgroup.png',
    'fig_age_crosscohort_bridge.png',
    'fig_roc_comparison.png',
    'fig_calibration_preop.png',
    'fig_dca_preop.png',
    'fig_feature_importance.png',
    'fig_tirads_roc.png',
    'fig_bridge_heatmap.png',
    'fig_bridge_sex_age.png',
    'fig_bridge_histology.png',
    'fig_bridge_tumorsize.png',
    'fig_prisma_flow.png',
    'fig_summary_panel.png',
]
for f in figs_list:
    path = os.path.join(FIGS, f)
    status = '[OK]' if os.path.exists(path) else '[MISSING]'
    log(f'  {status} {f}')

log('\n=== TABLES (9 total) ===')
tabs_list = [
    'Table1_patient_characteristics.csv',
    'Table2_lnr_lodds_performance.csv',
    'Table3_age_threshold_models.csv',
    'Table4_preop_model_performance.csv',
    'Table5_bridge_synthesis.csv',
    'table_age_crosscohort.csv',
    'table_feature_importance.csv',
    'table_tirads_thresholds.csv',
    'table_bridge_synthesis.csv',
]
for t in tabs_list:
    path = os.path.join(TABS, t)
    status = '[OK]' if os.path.exists(path) else '[MISSING]'
    log(f'  {status} {t}')

log('\n=== SCRIPTS (6 total) ===')
scripts = ['step1_cleaning.py','step2_lnr_lodds.py','step3_age_analysis.py',
           'step4_preop_model.py','step5_bridge_synthesis.py','step6_final_tables.py']
for s in scripts:
    path = os.path.join(BASE, 'code', s)
    status = '[OK]' if os.path.exists(path) else '[MISSING]'
    log(f'  {status} {s}')

# =============================================================
log_section('STEP 6 COMPLETE — PROJECT ANALYSIS DONE')
log(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log('\nNext steps for submission:')
log('  1. Review all 5 main tables for journal formatting')
log('  2. Select 5-6 main figures + supplementary figures')
log('  3. Write manuscript sections using key numbers from this analysis')
log('  4. Submit to: Journal of Intelligent Medicine / Frontiers in Endocrinology')

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

print(f'\nLog saved -> {LOG_FILE}')
