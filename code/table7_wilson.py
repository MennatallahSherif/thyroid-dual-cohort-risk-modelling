"""Table 7 (cross-cohort bridge) with Wilson 95% CIs, recomputed from the cleaned data using the grouping logic of step5
(left-closed ThyroidXL age bins; SEER age groups from the 5-year bin midpoint; invalid NPLN > D ratios are irrelevant here).
SEER proportions are crude CSS proportions over the available follow-up (full cohort, n = 61,615)."""
import pandas as pd, numpy as np, os
from statsmodels.stats.proportion import proportion_confint
P='../data/processed/'; OUT='../outputs/tables/'
s=pd.read_csv(P+'seer_clean.csv'); x=pd.read_csv(P+'thyroidxl_clean.csv')
x['age']=pd.to_numeric(x.age,errors='coerce'); x=x[x.age.notna()&x.age.between(10,100)].copy()
BINS=[0,35,45,55,65,75,100]; LAB=['<35','35-44','45-54','55-64','65-74','75+']
s['g']=pd.cut(s.age_mid,BINS,labels=LAB); x['g']=pd.cut(x.age,BINS,labels=LAB,right=False)
s['sex_std']=s.sex.str.strip(); x['sex_std']=x.gender.map({1:'Male',2:'Female'})
def cell(d,col):
    n=len(d); k=int(d[col].sum()); 
    if n==0: return n,k,np.nan,np.nan,np.nan
    lo,hi=proportion_confint(k,n,method='wilson'); return n,k,k/n*100,lo*100,hi*100
rows=[]
def add(dim,sub,sd,xd):
    sn,sk,sr,slo,shi=cell(sd,'css_event'); 
    if xd is None: xn=xk=xr=xlo=xhi=np.nan
    else: xn,xk,xr,xlo,xhi=cell(xd,'malignant')
    rows.append(dict(Dimension=dim,Subgroup=sub,SEER_n=sn,SEER_CSS_events=sk,SEER_CSS_pct=round(sr,2),SEER_CI=f'{slo:.2f}-{shi:.2f}',
        XL_n=xn,XL_malignant=xk,XL_mal_pct=(None if xd is None else round(xr,1)),XL_CI=(None if xd is None else f'{xlo:.1f}-{xhi:.1f}')))
for g in LAB: add('Age',g,s[s.g==g],x[x.g==g])
for sx in ['Female','Male']: add('Sex',sx,s[s.sex_std==sx],x[x.sex_std==sx])
for h in ['PTC','FTC','MTC']: add('Histology',h,s[s.histology_subtype==h],None)
for sx in ['Female','Male']:
    add('Age x Sex',f'{sx}, <55 yr',s[(s.sex_std==sx)&(s.age_55_plus==0)],x[(x.sex_std==sx)&(x.age<55)])
    add('Age x Sex',f'{sx}, >=55 yr',s[(s.sex_std==sx)&(s.age_55_plus==1)],x[(x.sex_std==sx)&(x.age>=55)])
t=pd.DataFrame(rows); t.to_csv(OUT+'Table7_bridge_wilson.csv',index=False); print(t.to_string(index=False))
