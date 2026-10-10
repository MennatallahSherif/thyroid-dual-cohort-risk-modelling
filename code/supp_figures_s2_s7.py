"""Supplementary Fig. S2 (prevalence adjustment for ACR-TIRADS >= 4) and Supplementary Fig. S7 (Set B calibration irregularity).
Both figures were in the submitted supplement but no code for them was in the submitted bundle; this script regenerates them.
Outputs: outputs/figures/fig_s2_tirads_prevalence_adjusted.png, fig_s7_setB_calibration_irregularity.png"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('step4_preop_model.py', encoding='utf-8').read().split("# SECTION C ")[0])
import os
FIGD = os.path.join(BASE, 'outputs', 'figures')
# ---- S2: PPV / NPV of TIRADS >= 4 across prevalence (sensitivity and specificity held constant)
A = xl[FEATURES_A + ['malignant']].dropna()
sel = xl.loc[A.index]; y = sel.malignant.values; pred = (sel.tirads >= 4).values
sens = (pred & (y == 1)).sum() / (y == 1).sum(); spec = ((~pred) & (y == 0)).sum() / (y == 0).sum(); dev = y.mean()
p = np.linspace(0.02, 0.50, 200)
ppv = sens * p / (sens * p + (1 - spec) * (1 - p)); npv = spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p)
fig, ax = plt.subplots(figsize=(7.5, 4.8)); ax.plot(p * 100, ppv * 100, color='#2563EB', lw=2.2, label='PPV'); ax.plot(p * 100, npv * 100, color='#EA580C', lw=2.2, label='NPV')
ax.axvline(dev * 100, color='gray', ls='--', lw=1, label=f'ThyroidXL prevalence ({dev*100:.1f}%)')
for q in (5, 10, 15, 20): ax.axvline(q, color='#D1D5DB', lw=0.8)
ax.set_xlabel('Malignancy prevalence (%)'); ax.set_ylabel('Predictive value (%)'); ax.set_ylim(0, 102)
ax.set_title(f'ACR-TIRADS >= 4 (sensitivity {sens*100:.1f}%, specificity {spec*100:.1f}% held constant)'); ax.legend(loc='center right'); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIGD, 'fig_s2_tirads_prevalence_adjusted.png'), dpi=200); plt.close()
# ---- S7: Set B out-of-fold predicted probabilities are bimodal because of the Bethesda distribution
Xb = df_B[FEATURES_B].values; yb = df_B.malignant.values; tb = df_B.tirads.values
oof = out_of_fold_predictions(Xb, yb, tb)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].hist(oof['XGBoost'], bins=20, range=(0, 1), color='#EA580C', alpha=.85); axes[0].set_yscale('log')
axes[0].set_xlabel('Out-of-fold predicted probability (XGBoost, Set B)'); axes[0].set_ylabel('Patients (log scale)'); axes[0].set_title('Few patients have intermediate predicted probabilities')
bt = df_B.assign(y=yb).groupby('fnac').y.agg(['size', 'mean']); lab = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}
axes[1].bar([lab[int(i)] for i in bt.index], bt['size'], color='#2563EB', alpha=.85); axes[1].set_yscale('log'); axes[1].set_xlabel('Bethesda category'); axes[1].set_ylabel('Patients (log scale)')
for i, (n, m) in enumerate(zip(bt['size'], bt['mean'])): axes[1].text(i, n * 1.15, f'{m*100:.0f}% malignant', ha='center', fontsize=8)
axes[1].set_title('Bimodal Bethesda distribution (II and VI dominate)')
for a in axes: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIGD, 'fig_s7_setB_calibration_irregularity.png'), dpi=200); plt.close(); print('ok')
