import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter, AalenJohansenFitter
from lifelines.utils import concordance_index
s=pd.read_csv('../data/processed/seer_clean.csv')
def cox(df,preds,pen=0.1):
    d=df[preds+['survival_months','css_event']].dropna(); c=CoxPHFitter(penalizer=pen).fit(d,'survival_months','css_event')
    return round(concordance_index(d.survival_months,-c.predict_partial_hazard(d),d.css_event),4),[round(float(np.exp(c.params_[p])),4) for p in preds],len(d)
a=s[s.survival_months.notna()&s.age_mid.notna()].copy()
print('stage_group values:',a.stage_group.value_counts(dropna=False).to_dict())
mp={'1':1,'2':2,'3':3,'4A':4,'4B':4,'4C':4,'4':4}
a['stage_num']=a.stage_group.astype(str).map(mp)
print('(a) AGE+STAGE  manuscript: binary+stage C=0.946 HR1.258 n=59,431 | continuous+stage C=0.954 HR 1.010/yr')
print('   binary  ',cox(a,['age_55_plus','stage_num'])); print('   contin. ',cox(a,['age_mid','stage_num']))
a['stage_num2']=a.stage_group.astype(str).map({'1':1,'2':2,'3':3,'4A':4,'4B':5,'4C':6,'4':4})
print('   alt coding 4A..4C=4,5,6 binary',cox(a,['age_55_plus','stage_num2']),'contin.',cox(a,['age_mid','stage_num2']))
# (b) RCS
x=a.age_mid; k=np.percentile(x,[5,50,95]); print('(b) knots 5/50/95:',k,' 10/50/90:',np.percentile(x,[10,50,90]))
def rcs(x,k):
    t=lambda u:np.maximum(u,0)**3
    return t(x-k[0])-t(x-k[1])*(k[2]-k[0])/(k[2]-k[1])+t(x-k[2])*(k[1]-k[0])/(k[2]-k[1])
a['rcs1']=rcs(a.age_mid,k)/((k[2]-k[0])**2)
print('   linear age',cox(a,['age_mid'])[0],'| RCS (age + nonlinear term)',cox(a,['age_mid','rcs1'])[0],' (manuscript: both 0.835)')
# (c) 90+ midpoint
b=a.copy(); b.loc[b.age_bin.astype(str).str.startswith('90'),'age_mid']=92.5
print('(c) continuous age, 90+ = 87:',cox(a,['age_mid']),' | 90+ = 92.5:',cox(b,['age_mid']))
# (d) lambda sensitivity, nodal cohort
imp=s.NPLN.notna()&s.D.notna()&(s.NPLN>s.D); s2=s.copy(); s2.loc[imp,['LNR','LODDS','NPLN','D']]=np.nan
s2['n_ordinal']=s2.n_stage.map({'N0':0,'N0a':0,'N0b':0,'N1a':1,'N1b':2})
nd=s2[s2.LNR.notna()&s2.LODDS.notna()&s2.survival_months.notna()].copy(); dm=pd.get_dummies(nd.n_stage,prefix='n',drop_first=True).astype(float); nd=pd.concat([nd,dm],axis=1)
print('(d) ridge lambda sensitivity (apparent C-index); manuscript claims variation <0.002 across 0.01, 0.1, 1.0')
for lab,pr in [('ordinal N + age',['n_ordinal','age_mid']),('N stage 7df + age',['age_mid']+list(dm.columns)),('LNR + age',['LNR','age_mid']),('LODDS + age',['LODDS','age_mid']),('LNR+LODDS+age',['LNR','LODDS','age_mid'])]:
    r={pen:cox(nd,pr,pen)[0] for pen in [0.0,0.01,0.1,1.0]}; print('  ',lab,r,' range 0.01-1.0:',round(max(r[0.01],r[0.1],r[1.0])-min(r[0.01],r[0.1],r[1.0]),4))
# (f) mean LNR by histology (raw seer_clean LNR, before excluding NPLN>D)
print('(f) mean LNR by histology (all LNR values in seer_clean):'); print(s.groupby('histology_subtype').LNR.agg(['count','mean',lambda v:(v>1).sum()]).round(3).to_string())
print('    after excluding NPLN>D:'); print(s2.groupby('histology_subtype').LNR.agg(['count','mean']).round(3).to_string())
# (e) 48-month KM vs Aalen-Johansen by age group
def lo(bn):
    bn=str(bn).strip(); return 90 if bn.startswith('90') else int(bn[:2])
a['g']=pd.cut(a.age_bin.apply(lo),[-1,34,44,54,64,74,200],labels=['<35','35-44','45-54','55-64','65-74','>=75'])
a['ev']=np.where(a.css_event==1,1,np.where(a.os_event==1,2,0))
print('(e) 48-month: crude CSS % | 1-KM | AJ CIF   (manuscript Table 3 KM col: 0.09,0.34,0.74,2.03,4.32,16.08; AJ: 0.09,0.34,0.74,2.00,4.27,15.26; supp fig S1: KM 0.10,0.55,0.97,2.69,5.59,16.08 AJ 0.10,0.56,0.98,2.70,5.54,15.26)')
for g,d in a.groupby('g',observed=True):
    km=KaplanMeierFitter().fit(d.survival_months,d.css_event); one=1-float(km.predict(48))
    aj=AalenJohansenFitter(seed=42,calculate_variance=False).fit(d.survival_months,d.ev,event_of_interest=1)
    cif=aj.cumulative_density_.loc[:48].iloc[-1,0]
    print(f'   {g:6s} n={len(d):6d}  crude {d.css_event.mean()*100:6.2f} | 1-KM {one*100:6.2f} | AJ {cif*100:6.2f} | other-cause deaths {int((d.ev==2).sum())}',flush=True)
