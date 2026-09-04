# Four machine learning methods on health data

Four analyses on four public medical datasets, one per method family, built
on one shared pipeline. A feed-forward network predicts thirty-day hospital
readmission. Three clustering methods try to recover hepatitis C diagnostic
categories with the labels withheld. Apriori mines association rules among
cell-nucleus measurements from breast tumors. Least squares predicts reported
alcohol intake from a liver enzyme panel. Each analysis has a write-up in
`docs/` and a notebook in `notebooks/`, and this page gives the background to
each and what it found.

The work was completed as coursework for DATA_SCI 8140, Advanced Methods in
Health Data Science, in **Fall 2025**. This repository rebuilds that work from
the original public dataset with the course scaffolding removed, so that the
methods and the reasoning behind them stand as a record in their own right.

| Method | Task | Data | Write-up | Notebook |
|---|---|---|---|---|
| Supervised classification | Readmission inside thirty days | Diabetes 130-US Hospitals, UCI 296 | [docs/01](docs/01-supervised-classification.md) | [notebooks/01](notebooks/01_classification.ipynb) |
| Clustering | Recover four diagnostic categories unlabeled | HCV panel, UCI 571 | [docs/02](docs/02-clustering.md) | [notebooks/02](notebooks/02_clustering.ipynb) |
| Association rules | Co-occurring diagnostic features | Wisconsin diagnostic breast cancer, UCI 17 | [docs/03](docs/03-association-rules.md) | [notebooks/03](notebooks/03_association_rules.ipynb) |
| Regression | Daily alcohol intake from a liver panel | BUPA liver disorders, UCI 60 | [docs/04](docs/04-regression.md) | [notebooks/04](notebooks/04_regression.ipynb) |

## The four projects

### Readmission inside thirty days

Strack et al. (2014) assembled 101,766 inpatient encounters of diabetic
patients at 130 US hospitals between 1999 and 2008 to ask whether measuring
glycated hemoglobin during a stay related to readmission. Each encounter
records demographics, diagnoses, medications, prior utilization and whether
the patient returned inside thirty days. The analysis predicts that return at
the moment of discharge. Encounters ending in death or hospice, 2,423 of them,
are removed because readmission cannot occur, and 2,235 more are removed for a
missing value in a retained column, leaving 97,108 encounters from 68,166
patients, of which 11.46 percent record readmission inside thirty days. The
partition is drawn over patients so that no patient has encounters on both
sides. A network with three
hidden layers of 32 units, selected from sixteen configurations by
cross-validation, scores ROC-AUC 0.658 and average precision 0.218 on 24,511
held-out encounters. Logistic regression on the same features scores 0.658 and
0.219, and over five redrawn partitions the two differ by 0.0001 on average.
The count of inpatient visits in the prior year and the discharge destination
carry the prediction; the glycated hemoglobin result is not among the 25
features the network uses most. The coursework oversampled the minority class
before splitting, and that ordering reports 0.806 for the same network.
Oversampling applied inside the training folds lowered cross-validated
ROC-AUC for every architecture tried, and the worst unbalanced configuration
beat the best balanced one by 0.037.

### Hepatitis C categories from a laboratory panel

Lichtinghagen et al. (2020) released ten routine blood assays for 615
subjects from a study of laboratory diagnostic pathways (Hoffmann et al.
2018). They are 533 blood donors, 7 donors marked as suspect, and 24 with
hepatitis, 21 with fibrosis and 30 with cirrhosis, the three disease stages in
the release. The analysis withholds the category, clusters the standardized
panel with k-means, DBSCAN and complete-linkage agglomerative clustering, and
scores each result against the withheld labels afterwards. The seven suspect
donors are removed, and 31 missing values in 26 subjects are imputed
from the five nearest subjects, so all 608 remaining subjects are analyzed.
Two k-means clusters place 23 of the 30 cirrhosis cases in a cluster of 25,
an adjusted Rand index of 0.815 against the cirrhosis label. That cluster has
a median albumin of 32.0 against 42.2 and a median bilirubin of 40.0 against
7.1, which is the laboratory picture of failing liver function. Hepatitis and
fibrosis are not recovered by any method at any k tried, because
their 45 subjects differ from donors by less than the donors differ among
themselves. The coursework selected its clustering by Calinski-Harabasz,
which ranks k-means first at four clusters; agreement with the labels ranks
k-means last, at 0.132 against 0.310 for DBSCAN.

### Co-occurring features of breast tumors

Wolberg et al. (1995) computed thirty measurements of cell nuclei from
digitized fine-needle aspirate images of 569 breast masses, 212 malignant and
357 benign (Street et al. 1993). Each measurement is discretized into three
levels by one-dimensional k-means, the levels and the two diagnoses become 92
items, and Apriori returns every itemset with support of at least 0.4 and
every rule with conviction of at least 10. At that support, which the
coursework used, no itemset can contain the malignant item, because the
malignant rate is 0.373 and an itemset cannot be more frequent than its rarest
item. All 321 rules whose consequent is the diagnosis therefore describe the
benign class. The strongest names the lowest level of mean concave points, of
radius error and of worst perimeter, and holds for 283 samples of which 282
are benign. The 19 items in those rules are all size and boundary-shape
measurements; texture, smoothness and symmetry appear in none. Lowering the
support shows where malignancy becomes describable: one malignant itemset at
0.3, thirty-one at 0.2 and 1,699 at 0.1.

### Reported alcohol intake from a liver enzyme panel

BUPA Medical Research collected five blood tests and the reported daily
intake of alcohol, in half-pints, for 345 men (Forsyth 1990). A seventh
column, `selector`, is a train and test split flag that most of the several
hundred papers using the data have misread as a disease label (McDermott and
Forsyth 2016). It is excluded here and gate 02 checks the exclusion. After
four duplicate rows are collapsed, least squares on 255 subjects gives
F = 10.07 with p = 8.6 × 10⁻⁹, so the five tests relate to intake, and a
held-out R² of 0.216 on the remaining 86 subjects, so they explain little of
it. Over twenty redraws of the partition the held-out R² averages 0.136 with a
standard deviation of 0.098 and ranges from −0.157 to 0.253. Mean corpuscular
volume and gamma-glutamyl transpeptidase are the two terms whose intervals
exclude zero, and they are the two assays used clinically to detect sustained
drinking. A random forest and a log-transformed target score 0.235 and 0.153,
so neither the linear form nor the skew of the target sets the ceiling. The
coursework clustered the first six columns of this file, which places the
intake among the features.

## How they compare

| | Classification | Clustering | Association rules | Regression |
|---|---|---|---|---|
| Supervision | Label used in fitting | Label withheld, used to score | Label carried as two ordinary items | Target used in fitting |
| Quantity estimated | Probability of readmission | A partition of the subjects | Frequent itemsets and rules | Coefficients and predictions |
| Held-out partition | 24,511 encounters, drawn over patients | None; the partition is the estimate | None; counts on all 569 | 86 subjects, redrawn twenty times |
| Governing setting | Architecture, alpha, training regime | Number of clusters; DBSCAN radius | Minimum support | None for least squares |
| Primary evidence | ROC-AUC 0.658 on held-out encounters | Adjusted Rand 0.815 at two clusters against cirrhosis | 321 rules, all naming the benign class | Held-out R² 0.216, mean 0.136 over redraws |
| Spread measured | Five patient-level partitions | Ten seeds; complete cases beside imputed | Support swept from 0.4 to 0.1 | Twenty row-level partitions |
| Where it failed here | Ceiling near 0.66 for any model | Hepatitis and fibrosis not recovered | Malignant class unreachable at support 0.4 | R² between −0.16 and 0.25 by draw |

## Data

The original data is committed under `data/raw/`, unmodified, as the UCI
Machine Learning Repository serves it at
`https://archive.ics.uci.edu/static/public/<id>/data.csv`.

| Analysis | File | UCI record | Shape |
|---|---|---|---|
| Classification | [`data/raw/diabetes_130_us_hospitals.csv`](data/raw/diabetes_130_us_hospitals.csv) | 296 | 101,766 × 50 |
| Clustering | [`data/raw/hcv_data.csv`](data/raw/hcv_data.csv) | 571 | 615 × 14 |
| Regression | [`data/raw/liver_disorders.csv`](data/raw/liver_disorders.csv) | 60 | 345 × 7 |
| Association rules | `sklearn.datasets.load_breast_cancer`, no file | 17 | 569 × 30 |

Each dataset is released under CC BY 4.0, cited in full in
[docs/references.md](docs/references.md) and attributed in
[data/README.md](data/README.md). [`data/checksums.txt`](data/checksums.txt)
records the SHA-256 digest of each file, and
[`data/download_data.py`](data/download_data.py) verifies the committed files
against it and fetches any that are absent. Gate 01 fails if a digest does not
match.

The rows each model fitted on are committed under `data/processed/`, written
by the pipeline and reproduced byte for byte by gate 05.

| File | Contents |
|---|---|
| [`readmission_cohort.csv`](data/processed/readmission_cohort.csv) | The 97,108 cleaned encounters with the 43 features, the identifiers and the label |
| [`readmission_partition.csv`](data/processed/readmission_partition.csv) | Encounter and patient identifiers with their train or test assignment |
| [`hcv_panel.csv`](data/processed/hcv_panel.csv) | The 608 subjects, imputed values in place and flagged, with the category |
| [`breast_cancer_measurements.csv`](data/processed/breast_cancer_measurements.csv) | The 569 samples, thirty measurements and the diagnosis |
| [`breast_cancer_transactions.csv`](data/processed/breast_cancer_transactions.csv) | The same samples as 92 one-hot items |
| [`liver_panel.csv`](data/processed/liver_panel.csv) | The 341 subjects, five tests and the target, `selector` dropped |
| [`liver_partition.csv`](data/processed/liver_partition.csv) | The same 341 rows with their train or test assignment |

The course copies of the data were not used. The course copy of the
readmission set is a preprocessed derivative whose binning and encoding are
undocumented, so every cleaning decision was re-derived from the original
release and is counted in the write-up.

## Reproducing it

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python data/download_data.py
.venv/Scripts/python analysis/run_all.py
```

`analysis/run_all.py` runs the four analyses and draws the twelve figures.
Each analysis writes its quantities into `results/metrics.csv`, its tables
into `results/` and the frame it fitted on into `data/processed/`. The shared
loading, splitting, evaluation and plotting code is in `src/`. The recorded
run took 1,874 seconds on a fourteen-core CPU, nearly all of it in the
classification's grid searches and repeated partitions.

The notebooks under `notebooks/` walk through each analysis in the order the
coursework did, cleaning, inputs, training and results, calling the same
functions as the pipeline and writing nothing into `results/` or
`data/processed/`. They need the kernel pinned in `requirements-notebooks.txt`:

```
.venv/Scripts/python -m pip install -r requirements-notebooks.txt
.venv/Scripts/python -m nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

## How it is checked

```
.venv/Scripts/python .checks/run_all_gates.py
.venv/Scripts/python .checks/inject_faults.py
```

Six gates guard the stages in order: environment, acquisition, schema,
preparation, modeling and reproducibility. Gate 05 re-runs the whole pipeline
into an empty directory and compares every recorded quantity, result table and
processed table against the committed ones byte for byte. `inject_faults.py`
breaks one thing at a time, a pinned version, a digest, the split flag, a
patient on both sides of the partition, a leaked feature, and confirms that
the gate guarding it fails.

## Citation

Taylor, P. *Four machine learning methods on health data.* 2026.
