# Two-Stage Risk Stratification in Thyroid Carcinoma Using Ultrasound and Nodal Metrics — analysis code (v1.1.0)

Reproducible re-analysis of two independent cohorts:

- **Diagnostic (primary):** ThyroidXL ultrasound cohort — machine-learning models versus ACR TI-RADS for nodule malignancy.
- **Prognostic (secondary):** SEER 2018–2022 differentiated thyroid carcinoma — AJCC N category, LNR and LODDS for short-term cancer-specific survival (CSS).
- **Integrative context:** within-cohort age gradients (no cross-cohort or individual-level inference).

All analyses follow the prespecified **Analysis Plan v1.1** (with deviation log) in `docs/`. Release v1.0.1 (Zenodo 10.5281/zenodo.23295450) is the record of the previous analysis and is superseded by this release.

## Repository layout

```
code/      analysis scripts (this folder)
data/      NOT distributed (see Data access); raw/ and processed_ref2024/ created locally
results/   analysis outputs written by the scripts
figures/   journal figures (PDF, PNG 600 dpi, TIFF 600 dpi)
tables/    LaTeX tables and Supplementary_Tables.xlsx
docs/      Analysis Plan v1.1, audit findings, Stage-2 report
```

## Data access

Data are not included in this repository.

- **ThyroidXL**: public dataset (Viet et al., MICCAI 2025), https://huggingface.co/datasets/hunglc007/ThyroidXL. Place `stats/id2info_eng.json` in `data/raw/`.
- **SEER**: case listing exported from SEER*Stat under the SEER Research Data Use Agreement (2018–2022 diagnoses, thyroid). Place the export as `data/raw/Seer_original_data.csv`. SEER data must not be redistributed.

## Environment

Python 3.12. Exact versions are pinned; results are reproducible only with these versions (newer scikit-learn changes model outputs).

```
pip install -r requirements_v2.txt
```

## Run order (from `code/`)

| Step | Command | Output |
|---|---|---|
| 1 Cleaning (both cohorts; flow counts) | `python step1v2_cleaning.py ..\data\raw ..\data\processed_ref2024 2024` | cleaned files, `flow_counts_v2.csv` |
| 2 Bridge feasibility check | `python step_bridge_feasibility.py ..\data\processed_ref2024 ..\results\bridge_ref2024` | age strata, slopes, decision |
| 3 Diagnostic models | `python step4v2_diagnostic.py ..\data\processed_ref2024 ..\results\diagnostic_ref2024` | AUC, paired ΔAUC, calibration, thresholds |
| 4 Age-reference sensitivity (A4) | repeat steps 1 and 3 with `2023`, `2025`, `exclude` into `processed_ref<year>` / `diagnostic_ref<year>` | |
| 5 Prognostic stage 1 | `python step2v2_prognostic_stage1.py ..\data\processed_ref2024 ..\results\prognostic` | HRs, PH tests, apparent performance |
| 6 Prognostic stage 2 | `python step2v2_prognostic_stage2.py ..\data\processed_ref2024 ..\results\stage2 main` then `sens`, `desc`, `dca` | bootstrap optimism (B = 500), sensitivities, descriptives, DCA |
| 7 Stage-2 tables | `python step2v2_stage2_report.py ..\results\stage2` | corrected performance, paired ΔC, calibration |
| 8 Verification | `python verify_outputs.py --data ..\data\processed_ref2024 --results ..\results` | `verification_report.csv` (must be 100% PASS) |
| 9 Figures | `python step7_figures.py --data ..\data\processed_ref2024 --results ..\results --out ..\figures` | 10 single-panel figures |
| 10 Tables | `python step8_tables.py --data ..\data\processed_ref2024 --results ..\results --out ..\tables` | main LaTeX tables, Supplementary S1–S19 |

Step 6 `main` takes about 45 minutes and `sens` about 35 minutes on a laptop; `main` resumes automatically if interrupted. Steps 9 and 10 read frozen outputs only and fit no model.

## Key methods (details in the Analysis Plan)

- **Diagnostic reference standard:** histopathology overrides FNAC; released labels as sensitivity. `conclusion = 4` denotes histopathology-confirmed benign.
- **Diagnostic validation:** stratified 5-fold cross-validation (seed 42), training-fold-only scaling and TI-RADS rates, fixed hyperparameters; 2,000-resample paired bootstrap; DeLong with Holm adjustment.
- **SEER corrections:** nodal codes 95–97 treated as non-quantitative; corrected ICD-O-3 histology map; T collapsed to T1–T4; unknown cause of death excluded (sensitivity analyses A and B).
- **Prognostic models:** unpenalised Cox (events per parameter 24.3); Harrell bootstrap optimism correction (B = 500, seed 20261024); Uno's C at 24 months; paired ΔC with Holm adjustment; 24-month calibration.
- **`coxfast.py`:** Efron Cox engine used for bootstrapping; validated against lifelines (identical HRs; 24-month risks within 1e-9).

## Verification

`verify_outputs.py` checks 106 frozen values and robustness conditions (cohort counts, AUCs, ΔAUC, corrected C-indices, paired ΔC, calibration, sensitivity conclusions). Missing outputs count as failures; the script exits with a non-zero status if any check fails.

## Citation

Please cite the article (details to be added) and this release (Zenodo DOI: 10.5281/zenodo.23622490).

## Licence

MIT.
