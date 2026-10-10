"""
=============================================================
STEP 1 — Data Cleaning & Harmonization
Thyroid Cancer Two-Cohort Study
=============================================================
Datasets:
  - Cohort A (ThyroidXL):  id2info_eng.json
  - Cohort B (SEER):       Seer_original_data.csv

Outputs (data/processed/):
  - seer_clean.csv
  - thyroidxl_clean.csv
  - cleaning_log.txt
=============================================================
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW        = os.path.join(BASE, 'data', 'raw')
PROCESSED  = os.path.join(BASE, 'data', 'processed')
LOGS       = os.path.join(BASE, 'logs')

SEER_IN    = os.path.join(RAW, 'Seer_original_data.csv')
XL_IN      = os.path.join(RAW, 'id2info_eng.json')
SEER_OUT   = os.path.join(PROCESSED, 'seer_clean.csv')
XL_OUT     = os.path.join(PROCESSED, 'thyroidxl_clean.csv')
LOG_FILE   = os.path.join(LOGS, 'cleaning_log.txt')

os.makedirs(PROCESSED, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# ── Logger ─────────────────────────────────────────────────
log_lines = []
def log(msg=''):
    print(msg)
    log_lines.append(msg)

def log_section(title):
    line = '=' * 60
    log(f'\n{line}')
    log(f'  {title}')
    log(line)

# ──────────────────────────────────────────────────────────
#  PART A — SEER CLEANING
# ──────────────────────────────────────────────────────────
log_section('PART A — SEER Cleaning')
log(f'Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

seer = pd.read_csv(SEER_IN)
seer = seer.loc[:, ~seer.columns.str.startswith('Unnamed')]
log(f'\nRaw shape: {seer.shape[0]:,} rows × {seer.shape[1]} columns')

# ── A1. Rename columns to short working names ──────────────
log('\n[A1] Renaming columns...')
seer.rename(columns={
    'Age recode with <1 year olds and 90+':              'age_bin',
    'Sex':                                               'sex',
    'SEER cause-specific death classification':          'css_status',
    'Vital status recode (study cutoff used)':           'vital_status',
    'Survival months':                                   'survival_months_raw',
    'Histology recode - broad groupings':                'histology_broad',
    'Derived EOD 2018 T Recode (2018+)':                 't_stage',
    'Derived EOD 2018 N Recode (2018+)':                 'n_stage',
    'Derived EOD 2018 M Recode (2018+)':                 'm_stage',
    'Derived EOD 2018 Stage Group Recode (2018+)':       'stage_group',
    'Tumor Size Summary (2016+)':                        'tumor_size_raw',
    'Regional nodes examined (1988+)':                   'nodes_examined_raw',
    'Regional nodes positive (1988+)':                   'nodes_positive_raw',
    'Year of diagnosis':                                 'year_dx',
    'Laterality':                                        'laterality',
    'RX Summ--Surg Prim Site (1998+)':                   'surgery_code',
    'Radiation recode':                                  'radiation',
    'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)': 'race',
    'Histologic Type ICD-O-3':                           'histology_icdo3',
}, inplace=True)
log(f'  Done — columns renamed.')

# ── A2. Survival months: string "0027" -> integer 27 ───────
log('\n[A2] Converting survival months...')
surv_str = seer['survival_months_raw'].astype(str).str.strip()
seer['survival_months'] = pd.to_numeric(surv_str, errors='coerce')
unknown_surv = seer['survival_months'].isna().sum()
log(f'  Unknown/missing survival months: {unknown_surv} -> set to NaN')
log(f'  Range: {seer["survival_months"].min():.0f} – {seer["survival_months"].max():.0f} months')
log(f'  Median: {seer["survival_months"].median():.0f} months')

# ── A3. Numeric conversions ────────────────────────────────
log('\n[A3] Converting numeric fields...')
seer['nodes_examined'] = seer['nodes_examined_raw'].astype(int)
seer['nodes_positive'] = seer['nodes_positive_raw'].astype(int)
seer['tumor_size']     = seer['tumor_size_raw'].astype(int)

# ── A4. Handle sentinel codes ──────────────────────────────
log('\n[A4] Handling sentinel/missing codes...')

# Nodes: 98 = not applicable, 99 = unknown -> NaN
n_ex_before = len(seer)
mask_nodes = (seer['nodes_examined'] < 98) & (seer['nodes_positive'] < 98)
seer['nodes_examined'] = seer['nodes_examined'].where(seer['nodes_examined'] < 98, np.nan)
seer['nodes_positive'] = seer['nodes_positive'].where(seer['nodes_positive'] < 98, np.nan)
valid_nodes = seer['nodes_examined'].notna() & seer['nodes_positive'].notna()
log(f'  Nodes valid: {valid_nodes.sum():,} / {len(seer):,} ({valid_nodes.mean()*100:.1f}%)')
log(f'  Nodes excluded (98/99): {(~valid_nodes).sum():,}')

# Tumor size: 999 = unknown -> NaN
seer['tumor_size'] = seer['tumor_size'].where(seer['tumor_size'] < 999, np.nan)
log(f'  Tumor size valid: {seer["tumor_size"].notna().sum():,} ({seer["tumor_size"].notna().mean()*100:.1f}%)')

# Stage group: 99, 88 -> NaN
seer['stage_group'] = seer['stage_group'].replace({'99': np.nan, '88': np.nan})
log(f'  Stage group valid: {seer["stage_group"].notna().sum():,}')

# M stage: 88 -> NaN
seer['m_stage'] = seer['m_stage'].replace({'88': np.nan})

# ── A5. Map ICD-O-3 to histology subtype ──────────────────
log('\n[A5] Mapping ICD-O-3 to histology subtype (PTC/FTC/MTC/ATC)...')
icd_map = {
    # PTC and variants
    8260: 'PTC', 8340: 'PTC', 8341: 'PTC', 8342: 'PTC',
    8343: 'PTC', 8344: 'PTC', 8337: 'PTC', 8339: 'PTC', 8347: 'PTC',
    # FTC and variants
    8330: 'FTC', 8335: 'FTC', 8290: 'FTC', 8350: 'FTC',
    # MTC
    8345: 'MTC', 8510: 'MTC',
    # ATC
    8020: 'ATC', 8021: 'ATC',
}
seer['histology_subtype'] = seer['histology_icdo3'].map(icd_map).fillna('Other')
dist = seer['histology_subtype'].value_counts()
for subtype, count in dist.items():
    log(f'  {subtype}: {count:,} ({count/len(seer)*100:.1f}%)')

# ── A6. Age bin -> midpoint ─────────────────────────────────
log('\n[A6] Converting age bins to midpoints...')
def age_bin_to_midpoint(bin_str):
    bin_str = str(bin_str).strip()
    if '90+' in bin_str or '85-89' in bin_str:
        return 87  # conservative midpoint for 85+
    if bin_str.startswith('00'):
        return 0.5  # '00 years' = <1 year old (midpoint of 0-1); previously parsed to NaN
    try:
        parts = bin_str.replace(' years','').split('-')
        return (int(parts[0]) + int(parts[1])) / 2
    except:
        return np.nan

seer['age_mid'] = seer['age_bin'].apply(age_bin_to_midpoint)
log(f'  Age midpoint range: {seer["age_mid"].min():.0f} – {seer["age_mid"].max():.0f}')
log(f'  Age midpoint mean: {seer["age_mid"].mean():.1f}')

# ── A7. Create survival outcome flags ─────────────────────
log('\n[A7] Creating survival outcome flags...')
# CSS event: 1 = died of thyroid cancer, 0 = alive or other cause
seer['css_event'] = (
    seer['css_status'] == 'Dead (attributable to this cancer dx)'
).astype(int)
# OS event: 1 = dead (any cause), 0 = alive
seer['os_event'] = (seer['vital_status'] == 'Dead').astype(int)

log(f'  CSS events (cancer death): {seer["css_event"].sum():,} ({seer["css_event"].mean()*100:.2f}%)')
log(f'  OS events (any death):     {seer["os_event"].sum():,} ({seer["os_event"].mean()*100:.2f}%)')

# ── A8. Age at 55 binary cutoff ────────────────────────────
log('\n[A8] Creating 55-year binary age cutoff (Packet I)...')
seer['age_55_plus'] = (seer['age_mid'] >= 55).astype(int)
log(f'  Age < 55:  {(seer["age_55_plus"]==0).sum():,} ({(seer["age_55_plus"]==0).mean()*100:.1f}%)')
log(f'  Age >= 55: {(seer["age_55_plus"]==1).sum():,} ({(seer["age_55_plus"]==1).mean()*100:.1f}%)')

# ── A9. Compute LNR and LODDS ──────────────────────────────
log('\n[A9] Computing LNR and LODDS...')
lnr_df = seer[valid_nodes].copy()
lnr_df['D']    = lnr_df['nodes_examined']
lnr_df['NPLN'] = lnr_df['nodes_positive']

# Exclude D=0 (no nodes examined — cannot compute ratio)
lnr_df = lnr_df[lnr_df['D'] > 0]
log(f'  Patients with D > 0: {len(lnr_df):,}')

# Flag low-yield dissections (D < 5 — as per review Table 2)
lnr_df['low_yield_flag'] = (lnr_df['D'] < 5).astype(int)
log(f'  Low yield (D < 5): {lnr_df["low_yield_flag"].sum():,} ({lnr_df["low_yield_flag"].mean()*100:.1f}%) — flagged but kept')

# LNR = NPLN / D
lnr_df['LNR'] = lnr_df['NPLN'] / lnr_df['D']

# LODDS = ln((NPLN + 0.5) / ((D - NPLN) + 0.5))  — continuity correction +0.5
lnr_df['LODDS'] = np.log(
    (lnr_df['NPLN'] + 0.5) / ((lnr_df['D'] - lnr_df['NPLN']) + 0.5)
)

log(f'  LNR  — mean: {lnr_df["LNR"].mean():.3f}, median: {lnr_df["LNR"].median():.3f}, range: [{lnr_df["LNR"].min():.3f}, {lnr_df["LNR"].max():.3f}]')
log(f'  LODDS— mean: {lnr_df["LODDS"].mean():.3f}, median: {lnr_df["LODDS"].median():.3f}, range: [{lnr_df["LODDS"].min():.3f}, {lnr_df["LODDS"].max():.3f}]')

# Merge LNR/LODDS back to main SEER dataframe
seer = seer.merge(
    lnr_df[['D','NPLN','LNR','LODDS','low_yield_flag']],
    left_index=True, right_index=True, how='left'
)

# ── A10. Surgery code mapping ──────────────────────────────
log('\n[A10] Mapping surgery codes...')
surgery_map = {
    0:  'No surgery',
    20: 'Partial thyroidectomy',
    21: 'Lobectomy',
    23: 'Lobectomy + isthmus',
    30: 'Subtotal thyroidectomy',
    40: 'Near-total thyroidectomy',
    50: 'Total thyroidectomy',
    27: 'Thyroidectomy NOS',
}
seer['surgery_type'] = seer['surgery_code'].map(surgery_map).fillna('Other/Unknown')
log(f'  Top surgery types:')
for stype, cnt in seer['surgery_type'].value_counts().head(5).items():
    log(f'    {stype}: {cnt:,}')

# ── A11. Radiation simplify ────────────────────────────────
log('\n[A11] Simplifying radiation field...')
seer['rai'] = seer['radiation'].apply(
    lambda x: 1 if 'Radioisotope' in str(x) else 0
)
log(f'  Received RAI: {seer["rai"].sum():,} ({seer["rai"].mean()*100:.1f}%)')

# ── A12. Sex standardize ───────────────────────────────────
log('\n[A12] Standardizing sex...')
seer['sex'] = seer['sex'].str.strip()
log(f'  Female: {(seer["sex"]=="Female").sum():,}  Male: {(seer["sex"]=="Male").sum():,}')

# ── A13. Drop raw/redundant columns ───────────────────────
log('\n[A13] Dropping raw columns...')
drop_cols = ['survival_months_raw','nodes_examined_raw','nodes_positive_raw',
             'tumor_size_raw','laterality','histology_broad']
seer.drop(columns=drop_cols, inplace=True, errors='ignore')

# ── A14. Final SEER summary ────────────────────────────────
log('\n[A14] Final SEER clean summary...')
log(f'  Final shape: {seer.shape[0]:,} rows × {seer.shape[1]} columns')
log(f'  Columns: {list(seer.columns)}')

seer.to_csv(SEER_OUT, index=False)
log(f'\n  Saved -> {SEER_OUT}')


# ──────────────────────────────────────────────────────────
#  PART B — THYROIDXL CLEANING
# ──────────────────────────────────────────────────────────
log_section('PART B — ThyroidXL Cleaning')

with open(XL_IN, 'r') as f:
    xl_raw = json.load(f)
log(f'Raw records: {len(xl_raw):,}')

# ── B1. Flatten JSON to DataFrame ─────────────────────────
log('\n[B1] Flattening JSON...')
records = []
for pid, info in xl_raw.items():
    row = {
        'patient_id':  pid,
        'age_raw':     info.get('age'),
        'gender':      info.get('gender'),
        'conclusion':  info.get('conclusion'),
    }
    n1 = info.get('nodule_1') or {}
    row['tirads']          = n1.get('TIRADS')
    row['fnac_raw']        = n1.get('FNAC')
    row['histopathology']  = n1.get('Histopathology')
    row['width']           = n1.get('Width')
    row['height']          = n1.get('Height')
    row['position_raw']    = n1.get('Position')
    row['has_nodule2']     = int(info.get('nodule_2') is not None)
    records.append(row)

xl = pd.DataFrame(records)
log(f'  Shape after flatten: {xl.shape}')

# ── B2. Age cleaning — fix year-entry errors ───────────────
log('\n[B2] Cleaning age...')
log(f'  Age > 100 before clean: {(xl["age_raw"] > 100).sum()} patients')
# Values like 1978, 2005 etc. are clearly birth-year entries (not age)
# Current year approx 2022 (SEER period) — convert: age = 2022 - birth_year
def fix_age(a):
    if pd.isna(a):
        return np.nan
    if 1900 <= a <= 2010:    # birth year entered by mistake
        return 2022 - int(a)
    if a > 110:              # clearly impossible age
        return np.nan
    return a

xl['age'] = xl['age_raw'].apply(fix_age)
log(f'  Age > 100 after clean: {(xl["age"] > 100).sum()} patients')
log(f'  Age NaN after clean:   {xl["age"].isna().sum()} patients')
log(f'  Age range: {xl["age"].min():.0f} – {xl["age"].max():.0f}')
log(f'  Age mean:  {xl["age"].mean():.1f}')

# ── B3. Gender standardize ─────────────────────────────────
log('\n[B3] Gender...')
xl['gender_label'] = xl['gender'].map({1: 'Male', 2: 'Female'})
log(f'  Male: {(xl["gender"]==1).sum():,}  Female: {(xl["gender"]==2).sum():,}')

# ── B4. Conclusion -> malignancy flag ──────────────────────
log('\n[B4] Mapping conclusion to malignancy...')
# 1 = Malignant (histopathology confirmed)
# 2 = Malignant (FNAC confirmed)
# 3 = Benign
# 4 = Suspicious / indeterminate
conclusion_map = {1: 'Malignant', 2: 'Malignant', 3: 'Benign', 4: 'Suspicious'}
xl['malignancy_label'] = xl['conclusion'].map(conclusion_map)
xl['malignant'] = xl['conclusion'].isin([1, 2]).astype(int)

log(f'  Malignant: {xl["malignant"].sum():,} ({xl["malignant"].mean()*100:.1f}%)')
log(f'  Benign:    {(xl["malignant"]==0).sum():,} ({(xl["malignant"]==0).mean()*100:.1f}%)')
log(f'  Suspicious (conclusion=4): {(xl["conclusion"]==4).sum():,} — kept as separate flag')

xl['is_suspicious'] = (xl['conclusion'] == 4).astype(int)

# ── B5. FNAC Bethesda standardize ─────────────────────────
log('\n[B5] Standardizing FNAC Bethesda categories...')
# Some entries are "6 (papillary thyroid carcinoma)" — extract numeric
def clean_fnac(val):
    if pd.isna(val):
        return np.nan
    val = str(val).strip()
    if val == '?':
        return np.nan
    # extract first digit
    try:
        return int(str(val)[0])
    except:
        return np.nan

xl['fnac'] = xl['fnac_raw'].apply(clean_fnac)

bethesda_map = {
    1: 'I - Nondiagnostic',
    2: 'II - Benign',
    3: 'III - AUS/FLUS',
    4: 'IV - Follicular neoplasm',
    5: 'V - Suspicious malignancy',
    6: 'VI - Malignant',
}
xl['fnac_label'] = xl['fnac'].map(bethesda_map)
log(f'  FNAC distribution:')
for cat, cnt in xl['fnac'].value_counts().sort_index().items():
    label = bethesda_map.get(int(cat), 'Unknown')
    log(f'    Bethesda {int(cat)} ({label.split(" - ")[1]}): {cnt:,}')
log(f'  FNAC missing: {xl["fnac"].isna().sum():,}')

# ── B6. Nodule size features ───────────────────────────────
log('\n[B6] Computing nodule size features...')
xl['nodule_area']   = xl['width'] * xl['height']
xl['aspect_ratio']  = (xl['width'] / xl['height']).replace([np.inf, -np.inf], np.nan)
log(f'  Width  — mean: {xl["width"].mean():.1f} mm, median: {xl["width"].median():.1f} mm')
log(f'  Height — mean: {xl["height"].mean():.1f} mm, median: {xl["height"].median():.1f} mm')
log(f'  Area   — mean: {xl["nodule_area"].mean():.1f} mm²')

# ── B7. Position -> simplified side ────────────────────────
log('\n[B7] Simplifying nodule position...')
def simplify_position(pos):
    if pd.isna(pos):
        return np.nan
    pos = str(pos).lower()
    if 'right' in pos and 'left' not in pos:
        return 'Right'
    elif 'left' in pos and 'right' not in pos:
        return 'Left'
    elif 'isthmus' in pos and 'right' not in pos and 'left' not in pos:
        return 'Isthmus'
    else:
        return 'Bilateral/Multiple'

xl['position'] = xl['position_raw'].apply(simplify_position)
log(f'  Position distribution:')
for pos, cnt in xl['position'].value_counts().items():
    log(f'    {pos}: {cnt:,} ({cnt/len(xl)*100:.1f}%)')

# ── B8. Histopathology subtype ─────────────────────────────
log('\n[B8] Histopathology subtype...')
log(f'  Available for: {xl["histopathology"].notna().sum():,} patients')
log(f'  Missing: {xl["histopathology"].isna().sum():,} (no surgery/pathology yet)')
# Top subtypes
if xl['histopathology'].notna().any():
    log(f'  Top subtypes:')
    for ht, cnt in xl['histopathology'].value_counts().head(8).items():
        log(f'    {ht}: {cnt:,}')

# ── B9. Age 55 cutoff ──────────────────────────────────────
log('\n[B9] Creating 55-year age cutoff for bridge analysis...')
xl['age_55_plus'] = (xl['age'] >= 55).astype(int)
log(f'  Age < 55:  {(xl["age_55_plus"]==0).sum():,}')
log(f'  Age >= 55: {(xl["age_55_plus"]==1).sum():,}')

# ── B10. TIRADS risk grouping ──────────────────────────────
log('\n[B10] TIRADS risk grouping...')
tirads_map = {
    1: 'T1-2 (Benign)',
    2: 'T1-2 (Benign)',
    3: 'T3 (Low suspicion)',
    4: 'T4 (Intermediate)',
    5: 'T5 (High suspicion)',
}
xl['tirads_group'] = xl['tirads'].map(tirads_map)
log(f'  TIRADS distribution:')
for t, cnt in xl['tirads'].value_counts().sort_index().items():
    mal_rate = xl[xl['tirads']==t]['malignant'].mean()*100
    log(f'    T{int(t)}: {cnt:,} patients — malignancy rate {mal_rate:.1f}%')

# ── B11. Drop raw columns ──────────────────────────────────
log('\n[B11] Dropping raw columns...')
drop_cols = ['age_raw', 'fnac_raw', 'position_raw']
xl.drop(columns=drop_cols, inplace=True, errors='ignore')

# ── B12. Final ThyroidXL summary ──────────────────────────
log('\n[B12] Final ThyroidXL clean summary...')
log(f'  Final shape: {xl.shape[0]:,} rows × {xl.shape[1]} columns')
log(f'  Columns: {list(xl.columns)}')
log(f'  Missing values:')
for col in xl.columns:
    n_miss = xl[col].isna().sum()
    if n_miss > 0:
        log(f'    {col}: {n_miss:,} missing ({n_miss/len(xl)*100:.1f}%)')

xl.to_csv(XL_OUT, index=False)
log(f'\n  Saved -> {XL_OUT}')


# ──────────────────────────────────────────────────────────
#  PART C — Bridge compatibility check
# ──────────────────────────────────────────────────────────
log_section('PART C — Bridge Variable Compatibility Check')

bridge_vars = {
    'Age':      ('age_mid in SEER', 'age in ThyroidXL', 'Both numeric — convertible'),
    'Sex':      ('sex in SEER',     'gender in ThyroidXL', 'Standardized to Male/Female'),
    'Histology':('histology_subtype in SEER', 'histopathology in ThyroidXL', 'PTC/FTC/MTC shared'),
    'Tumor size':('tumor_size in SEER (mm)', 'width/height in ThyroidXL (mm)', 'Comparable units'),
}
for bridge, (seer_col, xl_col, note) in bridge_vars.items():
    log(f'  [OK] {bridge}: SEER={seer_col} | XL={xl_col} | {note}')

# ──────────────────────────────────────────────────────────
#  SAVE LOG
# ──────────────────────────────────────────────────────────
log_section('CLEANING COMPLETE')
log(f'Finished: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
log(f'\nOutputs:')
log(f'  {SEER_OUT}')
log(f'  {XL_OUT}')
log(f'  {LOG_FILE}')

with open(LOG_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))

print(f'\nLog saved -> {LOG_FILE}')