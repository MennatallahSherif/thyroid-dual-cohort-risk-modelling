import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from lifelines.utils import concordance_index
s=pd.read_csv('../data/processed/seer_clean.csv'); a=s[s.survival_months.notna()&s.age_mid.notna()].reset_index(drop=True)
rows=[]
for lab,d in [('All',a),('PTC',a[a.histology_subtype=='PTC']),('FTC',a[a.histology_subtype=='FTC']),('MTC',a[a.histology_subtype=='MTC'])]:
    d=d.reset_index(drop=True)
    for spec,col in [('binary age (>=55)','age_55_plus'),('continuous age','age_mid')]:
        c0=concordance_index(d.survival_months,-d[col],d.css_event); rng=np.random.default_rng(42); cs=[]
        for _ in range(300):
            x=d.iloc[rng.integers(0,len(d),len(d))]; cs.append(concordance_index(x.survival_months,-x[col],x.css_event))
        lo,hi=np.percentile(cs,[2.5,97.5]); rows.append(dict(Stratum=lab,Specification=spec,n=len(d),CSS_events=int(d.css_event.sum()),C_index=round(c0,4),CI_full_sample_bootstrap_300=f'{lo:.3f}-{hi:.3f}')); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv('../outputs/tables/Table4_age_CI_full_sample.csv',index=False)
