# Data

No data are redistributed in this repository. SEER registry data must not be redistributed, and neither must any file derived from them at the patient level (for example `seer_clean.csv`).

Place the raw files in `data/raw/` before running `code/run_all.sh`:

| File | Source | How to obtain |
|---|---|---|
| `Seer_original_data.csv` | SEER Program, thyroid cancers diagnosed 2018-2022 (19 columns as exported; see the rename map at the top of `code/step1_cleaning.py`) | Obtain under the SEER research data agreement from https://seer.cancer.gov |
| `id2info_eng.json` | ThyroidXL dataset (Vietnamese tertiary referral ultrasound dataset) | See the dataset citation in the manuscript's reference list |

`data/processed/` is written by `step1_cleaning.py` and must stay out of version control (it is listed in `.gitignore`).
