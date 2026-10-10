#!/bin/sh
# Run the full pipeline from the raw files. Expected layout:  project/{data/raw, code}  (outputs/, logs/ are created)
# data/raw must contain Seer_original_data.csv and id2info_eng.json.  Run from the code/ directory:  sh run_all.sh
set -e
python step1_cleaning.py
python step2_lnr_lodds.py
python step3_age_analysis.py
python step4_preop_model.py
python step5_bridge_synthesis.py
python step6_final_tables.py
python extra_preop.py            # prevalence-adjusted ML, Bethesda split, Wilson CIs for Table 6
python extra_nodal.py            # LNR cutpoints, D>=5, CSS events by histology, aligned age table
python sensitivity_suspicious.py   # preoperative models with the 94 suspicious cases excluded (Table S15)
python lambda_sensitivity.py       # Table S14
python hr_sensitivity.py         # ridge vs unpenalised HRs and common-sample C-indices
python calibration_dca_24m.py    # 24-month calibration and censoring-aware DCA (overwrites the step2 calibration/DCA figures)
python s1_s2_competing_risk_rcs.py
python supp_figures_s2_s7.py     # Supplementary Figs S2 and S7
python supp_tables_fix.py        # Tables S3a, S4, S5
python table7_wilson.py          # Table 7 with Wilson intervals
python table4_ci.py              # full-sample bootstrap CIs for Table 4
python verify_outputs.py
