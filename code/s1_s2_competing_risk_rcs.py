"""
Supplementary Table S1 / Fig. S1 (Aalen-Johansen competing-risk sensitivity) and Supplementary Table S2 / Fig. S5
(restricted cubic spline vs linear vs binary age).

PROVENANCE: the original script was never part of the submitted bundle. The method below follows the code that generated
these results during the review (recovered from the revision session, 'fix2_competing_risks.py' and the follow-up block):
 - survival cohort = patients with known survival months and age_mid;
 - competing-risk coding: 0 censored, 1 cancer-specific death, 2 other-cause death (os_event = 1 and css_event = 0);
 - age groups from the 5-year-bin midpoint (<35, 35-44, 45-54, 55-64, 65-74, >=75);
 - 48-month landmark: 1 - Kaplan-Meier (other deaths censored) versus Aalen-Johansen CIF (calculate_variance = False);
 - RCS: 3 knots at the 5th/50th/95th percentiles of age (22, 52, 77), second spline term as in Harrell, ridge Cox (penalizer 0.1).
A fixed jitter seed (42) is added so that the Aalen-Johansen tie-jitter is reproducible (it moves values by about +/-0.01 pp).
Outputs: outputs/tables/TableS1_competing_risk.csv, TableS2_age_rcs.csv; outputs/figures/fig_competing_risk_age_bridge.png, fig_rcs_age_model.png
"""
import numpy as np, pandas as pd, warnings, os
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from lifelines import AalenJohansenFitter, KaplanMeierFitter, CoxPHFitter
TAB='../outputs/tables/'; FIG='../outputs/figures/'; os.makedirs(TAB,exist_ok=True); os.makedirs(FIG,exist_ok=True)
df=pd.read_csv('../data/processed/seer_clean.csv')
df['cr_event']=0; df.loc[df.css_event==1,'cr_event']=1; df.loc[(df.os_event==1)&(df.css_event==0),'cr_event']=2
d=df[df.survival_months.notna()&df.age_mid.notna()].copy()
groups=[('<35',d.age_mid<35),('35-44',(d.age_mid>=35)&(d.age_mid<45)),('45-54',(d.age_mid>=45)&(d.age_mid<55)),
        ('55-64',(d.age_mid>=55)&(d.age_mid<65)),('65-74',(d.age_mid>=65)&(d.age_mid<75)),('>=75',d.age_mid>=75)]
rows=[]
for lab,m in groups:
    s=d[m]; km=KaplanMeierFitter().fit(s.survival_months,s.css_event)
    km_r=(1-km.survival_function_[km.survival_function_.index<=48].iloc[-1,0])*100
    aj=AalenJohansenFitter(calculate_variance=False,seed=42).fit(s.survival_months,s.cr_event.astype(int),event_of_interest=1)
    aj_r=aj.cumulative_density_[aj.cumulative_density_.index<=48].iloc[-1,0]*100
    rows.append({'Age group':lab,'n':len(s),'CSS events':int(s.css_event.sum()),'Other-death events':int(((s.os_event==1)&(s.css_event==0)).sum()),
                 '1-KM at 48 months (%)':round(km_r,2),'AJ CIF at 48 months (%)':round(aj_r,2),'KM minus AJ (pp)':round(km_r-aj_r,2)})
t1=pd.DataFrame(rows); t1.to_csv(TAB+'TableS1_competing_risk.csv',index=False); print(t1.to_string(index=False))
fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(t1)); w=0.38
b1=ax.bar(x-w/2,t1['1-KM at 48 months (%)'],w,color='#2563EB',alpha=.85,label='1-KM (cause-specific, other deaths censored)')
b2=ax.bar(x+w/2,t1['AJ CIF at 48 months (%)'],w,color='#DC2626',alpha=.85,label='Aalen-Johansen CIF (competing-risk corrected)')
for b,v in list(zip(b1,t1['1-KM at 48 months (%)']))+list(zip(b2,t1['AJ CIF at 48 months (%)'])): ax.text(b.get_x()+b.get_width()/2,b.get_height()+.1,f'{v:.2f}%',ha='center',va='bottom',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(t1['Age group']); ax.set_xlabel('Age group'); ax.set_ylabel('48-month cumulative CSS incidence (%)')
ax.set_title(f'KM vs Aalen-Johansen at 48 months by age group (SEER 2018-2022, n = {len(d):,})'); ax.legend(); plt.tight_layout(); plt.savefig(FIG+'fig_competing_risk_age_bridge.png',dpi=150); plt.close()
# ---- RCS
k=np.percentile(d.age_mid,[5,50,95]); print('RCS knots',k)
def rcs(x,t1,t2,t3):
    p=lambda u:np.maximum(u,0); return (p(x-t1)**3-p(x-t2)**3*(t3-t1)/(t3-t2)+p(x-t3)**3*(t2-t1)/(t3-t2))/(t3-t1)**2
d['age_rcs2']=rcs(d.age_mid,*k)
def fit(cols):
    c=CoxPHFitter(penalizer=0.1).fit(d[cols+['survival_months','css_event']],'survival_months','css_event'); return c
cb,cl,cs=fit(['age_55_plus']),fit(['age_mid']),fit(['age_mid','age_rcs2'])
t2=pd.DataFrame([{'Model':'Binary (age >= 55)','C-index':round(cb.concordance_index_,4),'HR (ridge-penalised)':round(float(np.exp(cb.params_['age_55_plus'])),4),'n':len(d)},
                 {'Model':'Linear continuous','C-index':round(cl.concordance_index_,4),'HR (ridge-penalised)':round(float(np.exp(cl.params_['age_mid'])),4),'n':len(d)},
                 {'Model':f'RCS, 3 knots at {k[0]:.0f}/{k[1]:.0f}/{k[2]:.0f} years','C-index':round(cs.concordance_index_,4),'HR (ridge-penalised)':'nonlinear','n':len(d)}])
t2['Delta C vs linear']=(t2['C-index']-round(cl.concordance_index_,4)).round(4)
# The C-index depends only on the rank order of a single monotone predictor, so it cannot test linearity.
# Linearity is tested with a likelihood-ratio test of the spline term in UNPENALISED Cox models.
from scipy.stats import chi2
u_lin=CoxPHFitter(penalizer=0.0).fit(d[['age_mid','survival_months','css_event']],'survival_months','css_event')
u_rcs=CoxPHFitter(penalizer=0.0).fit(d[['age_mid','age_rcs2','survival_months','css_event']],'survival_months','css_event')
lr=2*(u_rcs.log_likelihood_-u_lin.log_likelihood_); p_lr=float(chi2.sf(lr,1))
t2['Log-likelihood (unpenalised)']=['',round(u_lin.log_likelihood_,1),round(u_rcs.log_likelihood_,1)]
t2['AIC (unpenalised)']=['',round(u_lin.AIC_partial_,1),round(u_rcs.AIC_partial_,1)]
t2['LR test of spline term']=['','',f'chi2 = {lr:.2f}, 1 df, p = {p_lr:.3f}']
t2.to_csv(TAB+'TableS2_age_rcs.csv',index=False); print(t2.to_string(index=False))
g=np.linspace(15,87,200); b_l,b_s=u_rcs.params_['age_mid'],u_rcs.params_['age_rcs2']; ref=52; r2=rcs(np.array([ref]),*k)[0]
fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(g,np.exp(b_l*(g-ref)+b_s*(rcs(g,*k)-r2)),'#DC2626',lw=2.5,label='RCS (3 knots)')
ax.plot(g,np.exp(u_lin.params_['age_mid']*(g-ref)),'#2563EB',lw=2,ls='--',label='Linear continuous'); ax.axvline(55,color='gray',ls='--',lw=1,label='AJCC-8 55-yr cutoff')
ax.set_xlabel('Age (years)'); ax.set_ylabel('Relative hazard (ref = age 52, unpenalised Cox)'); ax.legend(); ax.set_title('RCS vs linear age (SEER 2018-2022, CSS; LR test p = %.3f)'%p_lr); plt.tight_layout(); plt.savefig(FIG+'fig_rcs_age_model.png',dpi=150); plt.close()
