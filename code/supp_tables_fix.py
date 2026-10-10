"""Rebuild Table S3a, S4, S5 on one analytic definition each (post age-bin fix).
Analytic cohort for survival tables: SEER patients with known survival months (n = 61,474).
Nodal tables: subset with valid LNR/LODDS (n = 33,457). S3a: ThyroidXL Set A (n = 4,088)."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
OUT='../outputs/tables/'
s=pd.read_csv('../data/processed/seer_clean.csv')
imp=s.NPLN.notna()&s.D.notna()&(s.NPLN>s.D); s.loc[imp,['LNR','LODDS','NPLN','D']]=np.nan
s['n_ordinal']=s.n_stage.map({'N0':0,'N0a':0,'N0b':0,'N1a':1,'N1b':2})
node=s[s.LNR.notna()&s.LODDS.notna()&s.survival_months.notna()].copy()
dm=pd.get_dummies(node.n_stage,prefix='n',drop_first=True).astype(float); node=pd.concat([node,dm],axis=1); dcols=list(dm.columns)
def cidx(d,preds,pen):
    c=CoxPHFitter(penalizer=pen).fit(d[preds+['survival_months','css_event']],'survival_months','css_event')
    return concordance_index(d.survival_months,-c.predict_partial_hazard(d),d.css_event)
def boot(d,preds,pen,B=300,seed=42):
    rng=np.random.default_rng(seed); d=d.reset_index(drop=True); cs=[]
    for _ in range(B):
        x=d.iloc[rng.integers(0,len(d),len(d))]
        try: cs.append(cidx(x,preds,pen))
        except Exception: pass
    return np.percentile(cs,[2.5,97.5])
# ---------------- S4 : df-controlled N-stage comparison
specs=[('N-stage ordinal (N0/N1a/N1b, 1 df) + continuous age',1,['n_ordinal','age_mid'],'Equal-df comparison (rows use different samples; see Table S11 for the common sample)'),
       ('LNR continuous (1 df) + continuous age',1,['LNR','age_mid'],'Equal-df comparison (rows use different samples; see Table S11 for the common sample)'),
       ('LODDS continuous (1 df) + continuous age',1,['LODDS','age_mid'],'Equal-df comparison (rows use different samples; see Table S11 for the common sample)'),
       ('N-stage dummy (7 df) + continuous age [original]',len(dcols),['age_mid']+dcols,'Original paper model: all eight recorded N categories (N0, N0a, N0b, N1, N1a, N1b, NX, 88), reference = code 88; 7 dummy terms (the submitted text said 5 df)')]
rows=[]
for lab,df_,preds,note in specs:
    d=node[preds+['survival_months','css_event']].dropna()
    r=dict(Model=lab,df=df_,n=len(d),CSS_events=int(d.css_event.sum()))
    for pen,tag in [(0.1,'ridge (penalizer 0.1)'),(0.0,'unpenalised')]:
        c=cidx(d,preds,pen); lo,hi=boot(d,preds,pen)
        r[f'C-index {tag}']=round(c,3); r[f'Bootstrap 95% CI {tag}']=f'{lo:.3f}-{hi:.3f}'
    r['Note']=note; rows.append(r); print(r)
pd.DataFrame(rows).to_csv(OUT+'TableS4_nstage_comparison.csv',index=False)
# ---------------- S5 : subtype sensitivity (analytic cohort n = 61,474)
a=s[s.survival_months.notna()&s.age_mid.notna()].copy()
rows=[]
adeq={'PTC':'Limited — directional only','FTC':'Adequate for directional conclusions','MTC':'Adequate','ATC':'Robust — high event rate','Other':'Not assessed'}
for h in ['PTC','FTC','MTC','ATC','Other']:
    d=a[a.histology_subtype==h]
    r=dict(Subtype=h,N=len(d),**{'CSS events':int(d.css_event.sum())},**{'Event rate (%)':round(d.css_event.mean()*100,2)})
    if h in ('PTC','FTC','MTC'):
        c=cidx(d,['age_mid'],0.1)
        rng=np.random.default_rng(42); d2=d.reset_index(drop=True); cs=[]
        for _ in range(300):
            x=d2.iloc[rng.integers(0,len(d2),len(d2))]
            cs.append(concordance_index(x.survival_months,-x.age_mid,x.css_event))   # identical to Cox C-index for a single positive-coefficient predictor
        lo,hi=np.percentile(cs,[2.5,97.5])
        r['C-index continuous age']=round(c,3); r['Bootstrap 95% CI']=f'{lo:.3f}-{hi:.3f}'
    else:
        r['C-index continuous age']='N/A (see Supp Fig S4)' if h=='ATC' else 'N/A'; r['Bootstrap 95% CI']='—'
    r['Follow-up adequacy']=adeq[h]; rows.append(r); print(r)
t=pd.DataFrame(rows); print('total N',t.N.sum(),'events',t['CSS events'].sum())
t.to_csv(OUT+'TableS5_sensitivity_subtypes.csv',index=False)
# ---------------- S3a : prevalence-adjusted TIRADS >=4, from the same Set A data as Table 6
x=pd.read_csv('../data/processed/thyroidxl_clean.csv')
x['age']=pd.to_numeric(x.age,errors='coerce')
x['pos_isthmus']=(x.position=='Isthmus').astype(int); x['pos_left']=(x.position=='Left').astype(int)
FA=['tirads','age','gender','width','height','nodule_area','aspect_ratio','age_55_plus','pos_isthmus','pos_left']
A=x[FA+['malignant']].dropna(); y=A.malignant.values; pred=(A.tirads>=4).values
sens=(pred&(y==1)).sum()/(y==1).sum(); spec=((~pred)&(y==0)).sum()/(y==0).sum(); dev=y.mean()
rows=[]
for lab,p in [('5 (community)',.05),('10 (community)',.10),('15 (community)',.15),('20 (community)',.20),('30.1 (development)',dev)]:
    ppv=sens*p/(sens*p+(1-spec)*(1-p)); npv=spec*(1-p)/(spec*(1-p)+(1-sens)*p)
    rows.append({'Prevalence (%)':lab,'Sensitivity':f'{sens*100:.1f}%','Specificity':f'{spec*100:.1f}%','PPV (TIRADS>=4)':f'{ppv*100:.1f}%','NPV (TIRADS>=4)':f'{npv*100:.1f}%'})
pd.DataFrame(rows).to_csv(OUT+'TableS3a_prevalence_adjusted_tirads.csv',index=False); print(pd.DataFrame(rows).to_string()); print('Set A n',len(A))
