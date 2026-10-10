"""Compare regenerated outputs with the locked revision values. Prints PASS / FAIL per check and writes outputs/tables/verification_report.csv.
Tolerances: counts exact; C-index and AUC +/-0.0006; percentages +/-0.011 (Aalen-Johansen jitter allowance +/-0.02)."""
import pandas as pd, numpy as np, os
T='../outputs/tables/'; rows=[]
VERSION='verify v3 (61 checks expected)'; print(VERSION)
def chk(name,got,exp,tol):
    ok=(got is not None) and abs(float(got)-float(exp))<=tol
    rows.append(dict(check=name,expected=exp,got=None if got is None else round(float(got),5),tol=tol,result='PASS' if ok else 'FAIL'))
def rd(f):
    p=T+f; return pd.read_csv(p) if os.path.exists(p) else None
def val(df,col,key,keycol='Model',default=None):
    if df is None: return default
    r=df[df[keycol].astype(str).str.strip()==key]; return float(r.iloc[0][col]) if len(r) else default
lnr=rd('table_lnr_lodds_performance.csv')
for m,e in [('N stage (AJCC baseline)',0.8754),('LNR + age',0.8642),('LODDS + age',0.8405),('LNR + LODDS + age',0.8271),('Ordinal N-stage (1 df)',0.8835)]: chk('Table 2 C-index: '+m,val(lnr,'C-index',m),e,6e-4)
chk('Table 2 n (nodal cohort)',val(lnr,'N','LNR + age'),33457,0); chk('Table 2 n (ordinal N)',val(lnr,'N','Ordinal N-stage (1 df)'),32316,0)
age=rd('table_age_threshold_models.csv')
for m,e,n in [('Binary age (55yr cutoff)',0.7378,61474),('Continuous age (linear)',0.835,61474),('Binary age + stage',0.9455,59433),('Continuous age + stage',0.9538,59433),('MTC — binary age',0.56,1041),('MTC — continuous age',0.6256,1041)]:
    chk('Table 4 C-index: '+m,val(age,'C-index',m),e,6e-4); chk('Table 4 n: '+m,val(age,'N',m),n,0)
mp=rd('table_model_performance.csv')
if mp is None: rows.append(dict(check='mp: output file missing', expected='present', got=None, tol=None, result='FAIL'))
if mp is not None:
    for fs,m,e in [('Without FNAC','XGBoost',0.9336),('Without FNAC','Logistic Regression',0.9316),('With FNAC','XGBoost',0.9954),('Without FNAC','TIRADS alone (baseline)',0.9097)]:
        r=mp[(mp['Feature set']==fs)&(mp['Model']==m)]; chk(f'Table 5 CV AUC: {fs} {m}',float(r['CV AUC'].iloc[0]) if len(r) else None,e,6e-4)
b=rd('TableS6_bethesda_subgroup.csv')
if b is None: rows.append(dict(check='b: output file missing', expected='present', got=None, tol=None, result='FAIL'))
if b is not None:
    for m,e in [('XGBoost',0.852),('Logistic Regression',0.891),('Random Forest',0.858),('TIRADS alone (baseline)',0.722)]: chk('Bethesda III-V AUC: '+m,val(b,'AUC',m),e,6e-4)
d5=rd('TableS8_D5_sensitivity.csv')
for m,e in [('LNR + age',0.866),('LODDS + age',0.862),('N stage full',0.86)]: chk('D>=5 C-index: '+m,val(d5,'Cindex',m),e,6e-4)
chk('D>=5 n',val(d5,'n','LNR + age'),15821,0)
cal=rd('Table_calibration_24m_summary.csv')
for m,e in [('LNR + age',6.474),('LODDS + age',4.813),('N stage (7 df) + age',5.242)]: chk('24-month calibration slope: '+m,val(cal,'Calibration_slope',m),e,0.01)
chk('24-month mean predicted risk',val(cal,'Mean_predicted_24m','LNR + age'),0.0174,6e-4)
s1=rd('TableS1_competing_risk.csv')
if s1 is None: rows.append(dict(check='s1: output file missing', expected='present', got=None, tol=None, result='FAIL'))
if s1 is not None:
    for g,km,aj in [('<35',0.10,0.11),('35-44',0.55,0.56),('45-54',0.97,0.98),('55-64',2.69,2.69),('65-74',5.59,5.52),('>=75',16.08,15.26)]:
        r=s1[s1['Age group']==g]; chk('S1 1-KM 48m '+g,float(r['1-KM at 48 months (%)'].iloc[0]) if len(r) else None,km,0.011); chk('S1 AJ 48m '+g,float(r['AJ CIF at 48 months (%)'].iloc[0]) if len(r) else None,aj,0.021)
s2=rd('TableS2_age_rcs.csv')
if s2 is None: rows.append(dict(check='s2: output file missing', expected='present', got=None, tol=None, result='FAIL'))
if s2 is not None:
    chk('S2 linear C-index',float(s2.iloc[1]['C-index']),0.835,6e-4); chk('S2 RCS C-index',float(s2.iloc[2]['C-index']),0.835,6e-4)
    chk('S2 spline LR-test p',float(str(s2.iloc[2]['LR test of spline term']).split('p = ')[1]),0.055,0.002)
s15=rd('TableS15_sensitivity_excl_suspicious.csv')
if s15 is None: rows.append(dict(check='s15: output file missing', expected='present', got=None, tol=None, result='FAIL'))
if s15 is not None:
    a=s15[s15.Analysis.str.startswith('Set A')]; chk('S15 Set A n',a.n.iloc[0],3994,0); chk('S15 Set A XGBoost AUC',val(a,'AUC','XGBoost'),0.935,6e-4)
    b=s15[s15.Analysis.str.startswith('Bethesda')]; chk('S15 Bethesda III-V n',b.n.iloc[0],136,0); chk('S15 Bethesda III-V XGBoost AUC',val(b,'AUC','XGBoost'),0.857,6e-4)
s14=rd('TableS14_ridge_lambda_sensitivity.csv')
if s14 is None: rows.append(dict(check='s14: output file missing', expected='present', got=None, tol=None, result='FAIL'))
if s14 is not None:
    for m,e in [('LNR + age',0.0021),('LODDS + age',0.0108),('LNR + LODDS + age',0.0332)]: chk('S14 lambda range: '+m,val(s14,'Range across 0.01-1.0',m),e,6e-4)
t7=rd('Table7_bridge_wilson.csv')
if t7 is None: rows.append(dict(check='t7: output file missing', expected='present', got=None, tol=None, result='FAIL'))
if t7 is not None:
    r=t7[(t7.Dimension=='Age')&(t7.Subgroup=='<35')].iloc[0]; chk('Table 7 SEER <35 n',r.SEER_n,11061,0); chk('Table 7 XL <35 n (left-closed bins)',r.XL_n,634,0)
    r=t7[(t7.Dimension=='Age')&(t7.Subgroup=='75+')].iloc[0]; chk('Table 7 XL 75+ n',r.XL_n,46,0); chk('Table 7 XL 75+ malignancy %',r.XL_mal_pct,21.7,0.06)
out=pd.DataFrame(rows); out.to_csv(T+'verification_report.csv',index=False); print(out.to_string(index=False)); print('\nPASS',int((out.result=='PASS').sum()),'| FAIL',int((out.result=='FAIL').sum()),'| checks',len(out))
