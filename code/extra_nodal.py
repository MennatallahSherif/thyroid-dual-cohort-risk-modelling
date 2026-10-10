import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
s=pd.read_csv('../data/processed/seer_clean.csv')
imp=s.NPLN.notna()&s.D.notna()&(s.NPLN>s.D); s.loc[imp,['LNR','LODDS','NPLN','D']]=np.nan
nd=s[s.LNR.notna()&s.LODDS.notna()&s.survival_months.notna()&s.css_event.notna()].copy()
print('node subset',len(nd),'events',nd.css_event.sum())
ordm={'N0':0,'N0a':0,'N0b':0,'N1a':1,'N1b':2}; nd['n_ord']=nd.n_stage.map(ordm)
def cox(df,preds):
    d=df[preds+['survival_months','css_event']].dropna()
    c=CoxPHFitter(penalizer=0.1).fit(d,'survival_months','css_event')
    return concordance_index(d.survival_months,-c.predict_partial_hazard(d),d.css_event),len(d)
def boot(df,preds,B=300,seed=42):
    rng=np.random.default_rng(seed); d=df[preds+['survival_months','css_event']].dropna().reset_index(drop=True); cs=[]
    for _ in range(B):
        x=d.iloc[rng.integers(0,len(d),len(d))]
        try:
            c=CoxPHFitter(penalizer=0.1).fit(x,'survival_months','css_event'); cs.append(concordance_index(x.survival_months,-c.predict_partial_hazard(x),x.css_event))
        except Exception: pass
    return np.percentile(cs,[2.5,97.5])
# ---- m5: LNR cutpoints (common fair sample = rows with ordinal N available)
fair=nd[nd.n_ord.notna()].copy()
rows=[]
for cut in [0.20,0.25,0.30]:
    for D,lab in [(fair,'ordinal-N sample'),]:
        D=D.copy(); D['g1']=((D.LNR>0)&(D.LNR<=cut)).astype(int); D['g2']=(D.LNR>cut).astype(int)
        c,n=cox(D,['g1','g2','age_mid']); lo,hi=boot(D,['g1','g2','age_mid'])
        ev=D.groupby(np.where(D.LNR==0,'LNR=0',np.where(D.LNR<=cut,f'0<LNR<={cut}',f'LNR>{cut}'))).css_event.agg(['size','sum'])
        rows.append(dict(Model=f'LNR categorical cut {cut}',Cindex=round(c,3),CI=f'{lo:.3f}-{hi:.3f}',n=n,groups='; '.join(f'{k}:{int(v["size"])}/{int(v["sum"])}ev' for k,v in ev.iterrows())))
for preds,lab in [(['n_ord','age_mid'],'Ordinal N (1df)'),(['LNR','age_mid'],'LNR continuous'),(['LODDS','age_mid'],'LODDS continuous')]:
    c,n=cox(fair,preds); lo,hi=boot(fair,preds); rows.append(dict(Model=lab+' (same sample)',Cindex=round(c,3),CI=f'{lo:.3f}-{hi:.3f}',n=n,groups=''))
pd.DataFrame(rows).to_csv('../outputs/tables/TableS7_LNR_cutpoints.csv',index=False); print(pd.DataFrame(rows).to_string())
# ---- D>=5 sensitivity
d5=nd[nd.D>=5].copy(); print('D>=5 n',len(d5),'events',d5.css_event.sum())
nd_d=pd.get_dummies(d5.n_stage,prefix='n',drop_first=True).astype(float); d5e=pd.concat([d5,nd_d],axis=1); dc=[c for c in nd_d.columns]
rows=[]
for lab,D,preds in [('LNR + age',d5,['LNR','age_mid']),('LODDS + age',d5,['LODDS','age_mid']),('N stage full',d5e,['age_mid']+dc),('Ordinal N',d5,['n_ord','age_mid'])]:
    c,n=cox(D,preds); lo,hi=boot(D,preds); rows.append(dict(Model=lab,Cindex=round(c,3),CI=f'{lo:.3f}-{hi:.3f}',n=n))
pd.DataFrame(rows).to_csv('../outputs/tables/TableS8_D5_sensitivity.csv',index=False); print(pd.DataFrame(rows).to_string())
print('D<5 share',(nd.D<5).sum(),len(nd),round((nd.D<5).mean()*100,1))
# ---- R1-8: CSS events per histology
full=s[s.survival_months.notna()]
h=full.groupby('histology_subtype').agg(n=('css_event','size'),css_events=('css_event','sum'),os_events=('os_event','sum')); h['css_pct']=(h.css_events/h.n*100).round(2)
hn=nd.groupby('histology_subtype').agg(n_nodal=('css_event','size'),css_events_nodal=('css_event','sum'))
hh=h.join(hn); hh.to_csv('../outputs/tables/TableS9_css_events_by_histology.csv'); print(hh)
# ---- follow-up
print('median survival_months (all)',full.survival_months.median(),'max',full.survival_months.max())
k=KaplanMeierFitter().fit(full.survival_months,1-full.os_event); print('reverse-KM median follow-up (OS censoring)',k.median_survival_time_)
alive=full[full.os_event==0]; print('median FU among alive',alive.survival_months.median(),'n alive',len(alive))
print('person-years',full.survival_months.sum()/12)
print('CSS events',full.css_event.sum(),'of',len(full),round(full.css_event.mean()*100,2),'| of 61615:',round(s.css_event.sum()/len(s)*100,2),s.css_event.sum())
# ---- age strata CSS events (138-fold check)
full=full.copy(); 
for lab,sub in [('analytic n=61474',full),('full n=61615',s)]:
    g=sub.groupby('age_bin').agg(n=('css_event','size'),ev=('css_event','sum')); g['pct']=(g.ev/g.n*100).round(2); print(lab); print(g)

# ---- Aligned age-stratum table (full SEER cohort, n=61,615; groups from age_bin so every patient is assigned)
def lo_age(b):
    b=str(b).strip(); return 90 if b.startswith('90') else int(b[:2])
s['age_lo']=s['age_bin'].apply(lo_age)
s['age_group']=pd.cut(s['age_lo'],[-1,34,44,54,64,74,200],labels=['<35','35-44','45-54','55-64','65-74','75+'])
ag=s.groupby('age_group',observed=True).agg(seer_n=('css_event','size'),seer_css_events=('css_event','sum')).reset_index()
ag['seer_css_pct']=(ag.seer_css_events/ag.seer_n*100).round(2)
old=pd.read_csv('../outputs/tables/table_age_crosscohort.csv')[['age_group','xl_n','xl_mal_rate','xl_mal_pct']]
ag['age_group']=ag['age_group'].astype(str); ag=ag.merge(old,on='age_group')
ag.to_csv('../outputs/tables/Table_age_crosscohort_aligned.csv',index=False); print(ag.to_string()); print('total',ag.seer_n.sum(),ag.seer_css_events.sum())
