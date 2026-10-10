import numpy as np, pandas as pd, os, re
from sklearn.metrics import roc_auc_score
from statsmodels.stats.proportion import proportion_confint
exec(open('step4_preop_model.py', encoding='utf-8').read().split("# SECTION C ")[0])
TAB = TABS
# ---- OOF predictions Set A
X=df_A[FEATURES_A].values; y=df_A['malignant'].values; t=df_A['tirads'].values
oof=out_of_fold_predictions(X,y,t)
# ---- M3: prevalence-adjusted PPV/NPV (Bayes: fix sens/spec at threshold, vary prevalence)
rows=[]
for name in ['TIRADS alone (baseline)']+MODEL_NAMES:
    p=oof[name]
    for thr in [0.2,0.5]:
        pred=p>=thr
        sens=(pred&(y==1)).sum()/(y==1).sum(); spec=((~pred)&(y==0)).sum()/(y==0).sum()
        for prev in [0.05,0.10,0.20,y.mean()]:
            ppv=sens*prev/(sens*prev+(1-spec)*(1-prev)); npv=spec*(1-prev)/(spec*(1-prev)+(1-sens)*prev)
            rows.append(dict(Model=name,Threshold=thr,Prevalence=round(prev*100,1),Sens=round(sens,3),Spec=round(spec,3),PPV=round(ppv,3),NPV=round(npv,3)))
pd.DataFrame(rows).to_csv(f'{TAB}/TableS3b_prev_adj_ML.csv',index=False)
# sens-targeted threshold (95% sens) too
rows2=[]
for name in MODEL_NAMES:
    p=oof[name]
    thr=np.quantile(p[y==1],0.05); pred=p>=thr
    sens=(pred&(y==1)).sum()/(y==1).sum(); spec=((~pred)&(y==0)).sum()/(y==0).sum()
    for prev in [0.05,0.10,0.20,y.mean()]:
        rows2.append(dict(Model=name,Rule='threshold at 95% sens',Prevalence=round(prev*100,1),Sens=round(sens,3),Spec=round(spec,3),
          PPV=round(sens*prev/(sens*prev+(1-spec)*(1-prev)),3),NPV=round(spec*(1-prev)/(spec*(1-prev)+(1-sens)*prev),3)))
pd.DataFrame(rows2).to_csv(f'{TAB}/TableS3c_prev_adj_ML_95sens.csv',index=False)
# ---- m3: Wilson CIs for TIRADS thresholds
rows=[]
for tv in sorted(df_A.tirads.unique()):
    pred=(df_A.tirads>=tv).values
    tp=(pred&(y==1)).sum();fp=(pred&(y==0)).sum();fn=((~pred)&(y==1)).sum();tn=((~pred)&(y==0)).sum()
    def w(k,n):
        if n==0: return 'NA'
        lo,hi=proportion_confint(k,n,method='wilson'); return f'{k/n*100:.1f} ({lo*100:.1f}-{hi*100:.1f})'
    rows.append({'Cutpoint':f'TIRADS>={tv}','Sens':w(tp,tp+fn),'Spec':w(tn,tn+fp),'PPV':w(tp,tp+fp),'NPV':w(tn,tn+fn)})
pd.DataFrame(rows).to_csv(f'{TAB}/Table6_tirads_wilson.csv',index=False)
# ---- M4: Bethesda split (Set B OOF)
dfb=df_B; yb=dfb.malignant.values; Xb=dfb[FEATURES_B].values
oofb=out_of_fold_predictions(Xb,yb,dfb.tirads.values)
rows=[]
for lab,mask in [('III',dfb.fnac==3),('IV',dfb.fnac==4),('IV+V',dfb.fnac.isin([4,5])),('III-V',dfb.fnac.isin([3,4,5])),('V',dfb.fnac==5)]:
    m=mask.values; ys=yb[m]
    r=dict(Subgroup=lab,n=int(m.sum()),Malignant=int(ys.sum()),Prev=round(ys.mean()*100,1))
    for name in ['TIRADS alone (baseline)']+MODEL_NAMES:
        if len(np.unique(ys))<2: r[name]='NA'; continue
        a=roc_auc_score(ys,oofb[name][m]); lo,hi=bootstrap_auc_ci(ys,oofb[name][m]); r[name]=f'{a:.3f} ({lo:.3f}-{hi:.3f})'
    rows.append(r)
pd.DataFrame(rows).to_csv(f'{TAB}/TableS6b_bethesda_split.csv',index=False)
# bridge: malignant prevalence by Bethesda (full)
print(pd.crosstab(dfb.fnac,dfb.malignant))
# ---- label provenance in analytic Set A
print('Set A n',len(df_A),'malignant',y.sum())
print('done')
