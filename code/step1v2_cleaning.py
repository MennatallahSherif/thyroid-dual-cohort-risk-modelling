"""
STEP 1 v2 — Data cleaning under Analysis Plan v1.1 (24 Oct 2026).
No outcome model is fitted here.

Usage  : python step1v2_cleaning.py <raw_dir> <processed_dir> [reference year for birth-year ages: 2024 (default) | 2023 | 2025 | exclude]
Inputs : id2info_eng.json (ThyroidXL raw metadata), Seer_original_data.csv (SEER raw extraction)
Outputs: thyroidxl_clean_v2.csv, seer_clean_v2.csv, flow_counts_v2.csv, cleaning_log_v2.txt

Changes vs step1_cleaning.py (v1.0.1):
  ThyroidXL  - conclusion codes decoded as result x source (4 = benign, histopathology; not "suspicious")
             - primary outcome: histopathology text overrides FNAC (Plan 2.2, D4); released label kept as sensitivity
             - S-D1 flag: outcome derived from histopathology text; follicular thyroid tumor = uncertain (excluded)
             - age_55_plus not created (D12)
  SEER       - S1: nodal codes 95-97 (examined) and 95/97 (positive) are non-quantitative
             - S2: examined = 90 kept as 90 (flagged)
             - S3: corrected ICD-O-3 map
             - S4: unknown cause of death flagged; endpoint variants for primary / sensitivity A / B
             - S5: T collapsed to T1-T4; N0a/N0b/N0 -> N0; N1 NOS kept separate; TX/T0/NX/MX/88 -> missing
"""
import json, os, sys
import numpy as np, pandas as pd

IN_DIR  = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else '.'
# A4 (resolved 24 Oct 2026): ThyroidXL was collected over two years from February 2023 (Viet et al., MICCAI 2025);
# no per-patient examination year is released. Birth-year-coded ages use the midpoint year 2024 (maximum error about 1 year).
# Sensitivity: 2023, 2025, or 'exclude' (drop the birth-year-coded records). v1.0.1 used 2022 (before data collection).
REF_YEAR = sys.argv[3] if len(sys.argv) > 3 else '2024'
os.makedirs(OUT_DIR, exist_ok=True)
LOG = []
def log(m=''):
    print(m); LOG.append(str(m))
FLOW = []
def flow(cohort, step, n, events=None, note=''):
    FLOW.append(dict(cohort=cohort, step=step, n=int(n), events=None if events is None else int(events), note=note))

# =====================================================================
# PART B — ThyroidXL
# =====================================================================
log('=== ThyroidXL ===')
raw = json.load(open(os.path.join(IN_DIR, 'id2info_eng.json'), encoding='utf-8'))
rec = []
for pid, info in raw.items():
    n1 = info.get('nodule_1') or {}
    rec.append(dict(patient_id=int(pid), age_raw=info.get('age'), gender=info.get('gender'),
                    conclusion=info.get('conclusion'), conclusion_text=n1.get('Conclusion'),
                    tirads=n1.get('TIRADS'), fnac_raw=n1.get('FNAC'), histopathology=n1.get('Histopathology'),
                    width=n1.get('Width'), height=n1.get('Height'), position_raw=n1.get('Position'),
                    has_nodule2=int(info.get('nodule_2') is not None), n_images=len(info.get('images', []))))
xl = pd.DataFrame(rec)
flow('ThyroidXL', 'Raw patients', len(xl), (xl.conclusion.isin([1, 2])).sum(), 'events = released malignant')
assert xl.has_nodule2.sum() == 0, 'unexpected second nodule'

# conclusion decoding (verified against the Vietnamese source: 4 = "Lanh tinh MBH (4)")
CODE = {1: ('Malignant', 'Histopathology'), 2: ('Malignant', 'FNAC'), 3: ('Benign', 'FNAC'), 4: ('Benign', 'Histopathology')}
xl['released_result'] = xl.conclusion.map(lambda c: CODE[c][0])
xl['label_source'] = xl.conclusion.map(lambda c: CODE[c][1])
xl['malignant_released'] = xl.conclusion.isin([1, 2]).astype(int)
# consistency of code with its text label
chk = xl.conclusion_text.str.extract(r'\((\w+)\)\s*-\s*(\d)')
assert (chk[1].astype(int) == xl.conclusion).all()

# histopathology text classification
def hist_class(h):
    if pd.isna(h) or str(h).strip() == '':
        return np.nan
    h = str(h).strip().lower()
    if 'carcinoma' in h:
        return 'malignant'
    if 'tumor' in h:          # follicular thyroid tumor: uncertain malignant potential
        return 'uncertain'
    return 'benign'
xl['histopathology'] = xl.histopathology.where(xl.histopathology.isna(), xl.histopathology.astype(str).str.strip())
xl['hist_class'] = xl.histopathology.apply(hist_class)

# primary outcome: histopathology overrides FNAC; otherwise released label
xl['malignant'] = np.where(xl.hist_class == 'malignant', 1,
                  np.where(xl.hist_class == 'benign', 0, xl.malignant_released)).astype(int)
xl['label_changed'] = (xl.malignant != xl.malignant_released).astype(int)
xl['sd1'] = xl.hist_class.isin(['malignant', 'benign']).astype(int)
xl['sd1_malignant'] = np.where(xl.sd1 == 1, (xl.hist_class == 'malignant').astype(int), np.nan)

# age correction (Plan A4; see REF_YEAR above)
def fix_age(a):
    if pd.isna(a): return np.nan
    if 1900 <= a <= 2010: return np.nan if REF_YEAR == 'exclude' else int(REF_YEAR) - int(a)
    if a > 110: return np.nan
    return a
xl['age_birthyear_corrected'] = xl.age_raw.between(1900, 2010).astype(int)
xl['age'] = xl.age_raw.apply(fix_age)

# FNAC / Bethesda
def clean_fnac(v):
    if pd.isna(v): return np.nan
    v = str(v).strip()
    return int(v[0]) if v[:1].isdigit() else np.nan
xl['fnac'] = xl.fnac_raw.apply(clean_fnac)

# geometry and position (unchanged)
xl['nodule_area'] = xl.width * xl.height
xl['aspect_ratio'] = (xl.width / xl.height).replace([np.inf, -np.inf], np.nan)
def simplify_position(p):
    if pd.isna(p): return np.nan
    p = str(p).lower()
    if 'right' in p and 'left' not in p: return 'Right'
    if 'left' in p and 'right' not in p: return 'Left'
    if 'isthmus' in p and 'right' not in p and 'left' not in p: return 'Isthmus'
    return 'Bilateral/Multiple'
xl['position'] = xl.position_raw.apply(simplify_position)
xl['pos_isthmus'] = (xl.position == 'Isthmus').astype(int)
xl['pos_left'] = (xl.position == 'Left').astype(int)

FEATURES_A = ['tirads', 'age', 'gender', 'width', 'height', 'nodule_area', 'aspect_ratio', 'pos_isthmus', 'pos_left']
xl['in_setA'] = xl[FEATURES_A].notna().all(axis=1).astype(int)
xl['in_setB'] = (xl.in_setA.astype(bool) & xl.fnac.notna()).astype(int)
A = xl[xl.in_setA == 1]
flow('ThyroidXL', 'Excluded: unrecoverable age', (xl.in_setA == 0).sum(), xl.loc[xl.in_setA == 0, 'malignant'].sum())
flow('ThyroidXL', 'Set A analytic cohort (primary outcome)', len(A), A.malignant.sum())
flow('ThyroidXL', 'Set A, released labels (sensitivity)', len(A), A.malignant_released.sum())
flow('ThyroidXL', 'Labels changed by hierarchy', A.label_changed.sum(), None, ';'.join(map(str, A.loc[A.label_changed == 1, 'patient_id'])))
S1 = A[A.sd1 == 1]
flow('ThyroidXL', 'S-D1 histopathology-confirmed', len(S1), S1.sd1_malignant.sum(), f'benign = {int((S1.sd1_malignant == 0).sum())}')
flow('ThyroidXL', 'S-D1 excluded: follicular thyroid tumor', (A.hist_class == 'uncertain').sum(), None)
S1B = S1[S1.in_setB == 1]
flow('ThyroidXL', 'Set B within S-D1', len(S1B), S1B.sd1_malignant.sum(), f'benign = {int((S1B.sd1_malignant == 0).sum())}; rule >=100/>=100')
B = xl[xl.in_setB == 1]
flow('ThyroidXL', 'Set B analytic cohort', len(B), B.malignant.sum())
log(pd.crosstab([A.fnac.fillna(0).astype(int).map(lambda x: 'NA' if x == 0 else x)], [A.label_source, A.released_result], margins=True))
xl.drop(columns=['position_raw']).to_csv(os.path.join(OUT_DIR, 'thyroidxl_clean_v2.csv'), index=False)

# =====================================================================
# PART A — SEER
# =====================================================================
log('\n=== SEER ===')
s = pd.read_csv(os.path.join(IN_DIR, 'Seer_original_data.csv'), low_memory=False)
s = s.rename(columns={
    'Age recode with <1 year olds and 90+': 'age_bin', 'Sex': 'sex',
    'SEER cause-specific death classification': 'cod_class', 'Vital status recode (study cutoff used)': 'vital',
    'Survival months': 'surv_raw', 'Derived EOD 2018 T Recode (2018+)': 't_raw',
    'Derived EOD 2018 N Recode (2018+)': 'n_raw', 'Derived EOD 2018 M Recode (2018+)': 'm_raw',
    'Regional nodes examined (1988+)': 'ex_raw', 'Regional nodes positive (1988+)': 'pos_raw',
    'Histologic Type ICD-O-3': 'icdo3', 'RX Summ--Surg Prim Site (1998+)': 'surgery_code', 'Radiation recode': 'radiation',
    'Tumor Size Summary (2016+)': 'tumor_size_raw', 'Year of diagnosis': 'year'})
flow('SEER', 'Raw extraction', len(s), (s.cod_class == 'Dead (attributable to this cancer dx)').sum())

# age (unchanged midpoint rule; '00 years' -> 0.5; 85-89 and 90+ -> 87 as in v1.0.1)
def age_mid(b):
    b = str(b).strip()
    if '90+' in b or '85-89' in b: return 87.0
    if b.startswith('00'): return 0.5
    lo, hi = b.replace(' years', '').split('-'); return (int(lo) + int(hi)) / 2
s['age_mid'] = s.age_bin.apply(age_mid)
s['male'] = (s.sex == 'Male').astype(int)

# survival and endpoints (S4)
s['surv_months'] = pd.to_numeric(s.surv_raw, errors='coerce')
s['cod_unknown'] = (s.cod_class == 'Dead (missing/unknown COD)').astype(int)
s['dead'] = (s.vital == 'Dead').astype(int)
s['css_event'] = (s.cod_class == 'Dead (attributable to this cancer dx)').astype(int)             # primary (unknown COD excluded)
s['css_event_sensA'] = ((s.css_event == 1) | (s.cod_unknown == 1)).astype(int)                  # unknown COD = CSS
s['css_event_sensB'] = s.css_event                                                              # unknown COD = other death
s['other_death'] = ((s.dead == 1) & (s.css_event == 0) & (s.cod_unknown == 0)).astype(int)
s['other_death_sensB'] = ((s.dead == 1) & (s.css_event == 0)).astype(int)

# histology (S3)
PTC = [8050, 8260, 8337, 8339, 8340, 8341, 8342, 8343, 8344, 8350]
FTC = [8290, 8330, 8331, 8332, 8335]
MTC = [8345, 8510]; MIXED_MED = [8346, 8347]
ATC = [8020, 8021, 8022, 8030, 8031, 8032, 8033, 8034, 8035]
def histo(c):
    if c in PTC: return 'PTC'
    if c in FTC: return 'FTC'
    if c in MTC: return 'MTC'
    if c in MIXED_MED: return 'Mixed medullary'
    if c in ATC: return 'ATC'
    return 'Other'
s['histology'] = s.icdo3.apply(histo)
s['dtc'] = s.histology.isin(['PTC', 'FTC']).astype(int)
s['ftc'] = (s.histology == 'FTC').astype(int)

# staging (S5)
TMAP = {'T1a': 'T1', 'T1b': 'T1', 'T2': 'T2', 'T3a': 'T3', 'T3b': 'T3', 'T3': 'T3', 'T4a': 'T4', 'T4b': 'T4', 'T4': 'T4'}
NMAP = {'N0': 'N0', 'N0a': 'N0', 'N0b': 'N0', 'N1a': 'N1a', 'N1b': 'N1b', 'N1': 'N1 NOS'}
MMAP = {'M0': 'M0', 'M1': 'M1'}
s['T'] = s.t_raw.map(TMAP); s['N'] = s.n_raw.map(NMAP); s['M'] = s.m_raw.map(MMAP)   # TX, T0, NX, 88 -> NaN

# nodes (S1, S2)
ex = pd.to_numeric(s.ex_raw, errors='coerce'); po = pd.to_numeric(s.pos_raw, errors='coerce')
s['nodes_special_code'] = (ex.between(95, 97) | po.isin([95, 97])).astype(int)
s['nodes_examined_90plus'] = (ex == 90).astype(int)
quant = (ex.between(1, 90)) & (po.between(0, 90)) & (po <= ex)
s['nodal_quant'] = quant.astype(int)
s['D'] = ex.where(quant); s['NPLN'] = po.where(quant)
s['LNR'] = s.NPLN / s.D
s['LODDS'] = np.log((s.NPLN + 0.5) / (s.D - s.NPLN + 0.5))
s['nodes_category'] = np.select([quant, s.nodes_special_code == 1, ex == 0, ex.isin([98, 99]) | po.isin([98, 99])],
                                ['quantitative', 'special code 95-97', 'none examined', 'unknown/NA'], 'other')

# ---- flow: primary DTC nodal cohort
def ev(d, col='css_event'): return d[col].sum()
flow('SEER', 'Histology: DTC', s.dtc.sum(), ev(s[s.dtc == 1]))
for h in ['ATC', 'MTC', 'Mixed medullary', 'Other']:
    flow('SEER', f'Histology: {h}', (s.histology == h).sum(), ev(s[s.histology == h]))
d = s[s.dtc == 1]
for cat in ['quantitative', 'special code 95-97', 'none examined', 'unknown/NA', 'other']:
    sub = d[d.nodes_category == cat]
    flow('SEER', f'DTC nodes: {cat}', len(sub), ev(sub))
d = d[d.nodal_quant == 1]
d = d[d.surv_months.notna()]; flow('SEER', 'DTC quantitative nodal, survival time available', len(d), ev(d))
du = d[d.cod_unknown == 1]; flow('SEER', 'Excluded (primary): unknown cause of death', len(du), None, 'kept in sensitivity A/B')
d = d[d.cod_unknown == 0]; flow('SEER', 'DTC nodal, known cause of death', len(d), ev(d))
for v in ['T', 'N', 'M']:
    flow('SEER', f'  missing {v}', d[v].isna().sum(), ev(d[d[v].isna()]))
cc = d[d[['age_mid', 'male', 'T', 'M', 'N']].notna().all(axis=1)]
pct_missing = 100 * (1 - len(cc) / len(d))
flow('SEER', 'DTC nodal complete case (primary analytic cohort)', len(cc), ev(cc), f'{pct_missing:.1f}% excluded for covariates; MI sensitivity if >5%')
# EPP for M1: age 1 + sex 1 + histology 1 + T 3 + M 1 + N (levels-1)
n_levels = cc.N.nunique(); p_M1 = 1 + 1 + 1 + (cc['T'].nunique() - 1) + (cc.M.nunique() - 1) + (n_levels - 1)
epp = ev(cc) / p_M1
flow('SEER', 'EPP for M1 (largest model)', p_M1, ev(cc), f'EPP = {epp:.1f}; rule >=20 -> unpenalised primary')
log(f'N levels in cohort: {cc.N.value_counts().to_dict()}; T: {cc["T"].value_counts().to_dict()}; M: {cc.M.value_counts().to_dict()}')
log(f'Histology in cohort: {cc.histology.value_counts().to_dict()}')
s.to_csv(os.path.join(OUT_DIR, 'seer_clean_v2.csv'), index=False)

F = pd.DataFrame(FLOW); F.to_csv(os.path.join(OUT_DIR, 'flow_counts_v2.csv'), index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 80)
log('\n' + F.to_string(index=False))
open(os.path.join(OUT_DIR, 'cleaning_log_v2.txt'), 'w', encoding='utf-8').write('\n'.join(LOG))
