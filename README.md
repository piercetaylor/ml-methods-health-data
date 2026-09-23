# Four machine learning methods on health data

This repository compares four methods on public health datasets: supervised classification of hospital readmission, clustering of hepatitis C laboratory profiles, association rules for breast tumor measurements, and regression of reported alcohol intake. The analyses share data preparation, evaluation, and reproducibility checks. The work originated in DATA_SCI 8140, Advanced Methods in Health Data Science, in Fall 2025.

| Analysis | Data | Main result | Details |
|---|---|---|---|
| Thirty-day readmission | Diabetes 130-US Hospitals, UCI 296 | A neural network scored 0.658 ROC-AUC on held-out patients, close to logistic regression at 0.658. | [Classification](docs/01-supervised-classification.md) |
| Hepatitis C profiles | HCV laboratory panel, UCI 571 | Two clusters separated most cirrhosis cases; hepatitis and fibrosis were not reliably recovered. | [Clustering](docs/02-clustering.md) |
| Breast tumor measurements | Wisconsin diagnostic breast cancer, UCI 17 | At minimum support 0.4, all 321 diagnosis-consequent rules described benign tumors. | [Association rules](docs/03-association-rules.md) |
| Reported alcohol intake | BUPA liver disorders, UCI 60 | A five-assay linear model scored 0.216 held-out R²; the mean across twenty splits was 0.136. | [Regression](docs/04-regression.md) |

These estimates answer different questions and are not comparable as one model ranking. The [full analysis record](docs/legacy-readme.md) gives cohort decisions, sensitivity checks, and interpretations. Four [notebooks](notebooks/README.md) present the analyses using the same pipeline functions.

## Data and limits

The original UCI files are committed in [`data/raw/`](data/raw/). The breast cancer data are loaded through scikit-learn. The four source datasets are attributed in [`data/README.md`](data/README.md), with full references in [`docs/references.md`](docs/references.md); each source dataset is released under CC BY 4.0. The processed tables and recorded results are committed in [`data/processed/`](data/processed/) and [`results/`](results/).

Readmission is evaluated across patient-level partitions, so encounters from one patient do not cross a split. Clustering and association rules are descriptive analyses without held-out performance estimates. The BUPA panel has limited predictive value for reported intake, and its `selector` column is a split flag, not a disease label. The analyses do not establish clinical utility or causality.

## Reproduce and check

The commands below use Windows PowerShell. The pinned dependencies are in [`requirements.txt`](requirements.txt).

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python data/download_data.py
.venv/Scripts/python analysis/run_all.py
.venv/Scripts/python .checks/run_all_gates.py
```

The pipeline writes figures, result tables, and processed data. Six gates check the environment, source data, schema, preparation, modeling, and reproducibility. The final gate reruns the pipeline and compares recorded outputs with the committed results. [`data/checksums.txt`](data/checksums.txt) records the source-file SHA-256 digests.

## Citation and license

Taylor, P. *Four machine learning methods on health data.* 2026. The repository license is [MIT](LICENSE). The source datasets retain their own CC BY 4.0 licenses and attribution requirements.
