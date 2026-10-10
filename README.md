# Reproducibility code: dual-cohort thyroid carcinoma study (revised)

Repository name: `thyroid-dual-cohort-risk-modelling`

Revised analysis pipeline for *Ultrasound Malignancy Prediction in Thyroid Nodules and Nodal Burden and Age Modelling in Thyroid Carcinoma*.
Verified by the authors on their own machine: `code/run_all.sh` runs end to end from the raw files and `verify_outputs.py` reports 61 PASS and 0 FAIL.

## Licence and citation
MIT licence (see `LICENSE`); cite as in `CITATION.cff`.

## Layout (run from `code/`)

```
project/
├── data/raw/        Seer_original_data.csv, id2info_eng.json   (not redistributed; see Data availability)
├── data/processed/  written by step1
├── outputs/tables/  outputs/figures/  logs/   (created by the scripts)
└── code/            everything in this folder
```

```bash
pip install -r requirements.txt
cd code && sh run_all.sh
```

## Scripts

| Script | Purpose | Origin |
|---|---|---|
| `step1_cleaning.py` | Clean both cohorts | submitted; fix: SEER age bin `00 years` mapped to 0.5 (was missing) |
| `step2_lnr_lodds.py` | Nodal metrics, Cox models, KM, bootstrap CIs (300, seed 42) | submitted, unchanged. Its calibration/DCA sections are superseded (see below) |
| `step3_age_analysis.py` | Binary/continuous age Cox models, age-plus-stage, bridge age table and figure | submitted; fixes: ThyroidXL age groups left-closed (`right=False`); bridge SEER rates from the full cohort |
| `step4_preop_model.py` | Preoperative models, Bethesda III-V subgroup | submitted, unchanged |
| `step5_bridge_synthesis.py` | Cross-cohort bridge | submitted; fixes: left-closed ThyroidXL bins; invalid ratios (positive nodes > examined) set to missing before LNR summaries |
| `step6_final_tables.py` | Assemble Tables 1-5, flow and summary figures | submitted; fixes: stale hard-coded strings (ridge-penalised HRs, validation labels), Table 4 delta AUC reference, script-path check |
| `extra_preop.py` | Prevalence-adjusted ML PPV/NPV, Bethesda split, Wilson CIs for Table 6 | revision |
| `extra_nodal.py` | LNR cutpoints, D >= 5, CSS events by histology | revision |
| `sensitivity_suspicious.py` | Preoperative models with the 94 suspicious cases excluded (Table S15) | revision |
| `lambda_sensitivity.py` | Ridge-penalty sensitivity of the apparent C-index (Table S14) | revision |
| `hr_sensitivity.py` | Ridge vs unpenalised HRs; common-sample C-indices | revision |
| `calibration_dca_24m.py` | 24-month calibration and censoring-aware decision curves; overwrites the step2 calibration/DCA figures | revision |
| `s1_s2_competing_risk_rcs.py` | Aalen-Johansen (Table S1, Fig. S1) and spline (Table S2, Fig. S5) analyses | **method recovered from the revision session; the original script was not in the submitted bundle** |
| `supp_tables_fix.py` | Tables S3a, S4 (`TableS4_nstage_comparison.csv`), S5 | revision |
| `table7_wilson.py` | Table 7 with Wilson intervals | revision (original interval code not recovered) |
| `table4_ci.py` | Full-sample bootstrap CIs for Table 4 (the submitted CIs came from 15,000/8,000-patient subsamples) | revision |
| `verify_outputs.py` | Checks regenerated numbers against the locked values | revision |
| `provenance_checks.py` | Diagnostics used to trace submitted numbers (not part of the pipeline) | revision |

## Notes

- **Hazard ratios** from `step2`/`step3` are ridge-penalised (lifelines `penalizer = 0.1`) and shrunken; unpenalised values are in `Table_HR_ridge_vs_unpenalised.csv`.
- **Spline knots** are at the 5th/50th/95th percentiles of age (22, 52, 77 years). The submitted README said 10th/50th/90th and listed spline analysis under `step3`; neither was correct.
- **Aalen-Johansen** tie-jitter is seeded (42); values can still differ by about 0.01-0.02 percentage points between library versions.
- **XGBoost** must be 2.1.3 (the version in the submitted Methods); 2.0.3 and 2.1.4 also reproduce the locked result, other versions do not.
- **Data**: SEER data are available to approved users under the SEER research agreement and must not be redistributed (including `seer_clean.csv`). ThyroidXL: see the dataset DOI in the manuscript.

- **N-stage model**: the dummy-coded AJCC-8 N-stage model has 7 degrees of freedom (eight recorded categories, reference code 88); the submitted text and labels said 5.
- **Authors' environment** (verified): Python 3.12.10, numpy 2.4.4, pandas 2.3.3, scikit-learn 1.8.0, lifelines 0.30.3, xgboost 2.1.3, scipy 1.17.1, statsmodels 0.15.0.
