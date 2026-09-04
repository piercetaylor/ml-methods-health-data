# Four machine learning methods on health data

Four analyses, one per method family, on four public medical datasets, sharing
one pipeline. A feed-forward network predicts thirty-day hospital readmission
from 97,108 encounters (Strack et al. 2014). Three clustering methods try to
recover hepatitis C diagnostic categories from a ten-assay laboratory panel for
608 subjects with the labels withheld (Lichtinghagen et al. 2020). Apriori mines
association rules among thirty cell-nucleus measurements for 569 breast tumor
samples (Wolberg et al. 1995). Least squares predicts self-reported daily
alcohol intake from a five-test liver enzyme panel for 341 men (Forsyth 1990).
Each has a write-up in `docs/`, and this page compares them.

The work was completed as coursework for DATA_SCI 8140, Advanced Methods in
Health Data Science, in **Fall 2025**. This repository rebuilds that work from
the original public dataset with the course scaffolding removed, so that the
methods and the reasoning behind them stand as a record in their own right.

| Method | Task | Data | Write-up |
|---|---|---|---|
| Supervised classification | Readmission inside thirty days | Diabetes 130-US Hospitals, UCI 296 | [docs/01](docs/01-supervised-classification.md) |
| Clustering | Recover four diagnostic categories unlabeled | HCV panel, UCI 571 | [docs/02](docs/02-clustering.md) |
| Association rules | Co-occurring diagnostic features | Wisconsin diagnostic breast cancer, UCI 17 | [docs/03](docs/03-association-rules.md) |
| Regression | Daily alcohol intake from a liver panel | BUPA liver disorders, UCI 60 | [docs/04](docs/04-regression.md) |

## What each method established

### Classification

A network with three hidden layers of 32 units, selected
from sixteen configurations by three-fold cross-validation drawn over patients,
scores ROC-AUC 0.658 and average precision 0.218 on 24,511 held-out encounters
from 17,042 patients who appear nowhere in training, against a readmission
rate of 0.115. Logistic regression on the same features scores 0.658 and 0.219.
Over five redrawn patient-level partitions the two models average 0.662 and
0.662, and the network is ahead in three draws by a mean of 0.0001. The
network establishes what the linear model establishes and nothing more. Prior
inpatient visits and the discharge destination carry most of what is
predictable. The glycated hemoglobin result the dataset was assembled around
carries almost none of it, and the ceiling is a property of the recorded facts.
The one decision that moved the result was the class balancing the coursework
applied. Searched as a training regime beside its absence, it lowered
cross-validated ROC-AUC for every architecture, by 0.037 between the worst
unbalanced and the best balanced. On the selected architecture it cost 0.063 on
the held-out encounters and raised the Brier score from 0.098 to 0.234.

### Clustering

Cirrhosis is separable from a routine laboratory panel with no
labels. Two k-means clusters on the standardized panel place 23 of the 30
cirrhosis cases in a cluster of 25, an adjusted Rand index of 0.815 against
the cirrhosis label. The profile of that cluster is failing synthetic liver
function: median albumin 32.0 against 42.2, bilirubin 40.0 against 7.1,
cholinesterase 2.47 against 8.39. Hepatitis and fibrosis are not separable by
any of the three methods at any k tried, because their 45 subjects
differ from healthy donors by less than the donors differ among themselves. In
the four-cluster comparison, Calinski-Harabasz, the index the coursework chose
by, ranks k-means first at 102.5 against 45.5 for DBSCAN, and
agreement with the withheld labels ranks it last, 0.132 against 0.310. The
index measures the quantity k-means minimizes and is not a neutral judge.

### Association rules

At the minimum support of 0.4 the coursework used, no
frequent itemset can contain the malignant diagnosis, because the malignant
rate is 0.373 and an itemset cannot be more frequent than its rarest item.
Every one of the 321 rules whose consequent is the diagnosis therefore
describes the benign class. The strongest, lowest level of mean concave points
with lowest level of radius error and of worst perimeter, holds for 283 samples
of which 282 are benign, a lift of 1.588 against a ceiling of 1.594. The 19
items those rules select are all size and boundary-shape measurements, and
texture, smoothness and symmetry appear in none. The selection was reached by
counting co-occurrences and not by fitting anything to the label.
The finding is about the benign half of the feature space and nothing else,
which is a consequence of the threshold and not of the tumors.

### Regression

Five blood tests relate to reported daily intake, F = 10.07
with p = 8.6 × 10⁻⁹ on the training partition, and explain little of it. The
held-out R² is 0.216 on the primary split of 86 subjects, 0.136 ± 0.098 over
twenty redrawn splits, and 0.058 ± 0.093 under five-fold cross-validation.
Mean corpuscular volume and gamma-glutamyl transpeptidase are the two terms
whose intervals exclude zero, at 0.167 and 0.020 half-pints per day per
released unit, and they are the two assays used clinically to detect sustained
drinking. A log-transformed target and a random forest change none of this,
so the ceiling is in the data. The seventh released column, `selector`, is a
train and test split flag that most of the published literature on this
dataset has used as a disease label (McDermott and Forsyth 2016); it is
excluded here, and gate 02 checks that it is.

## What counts as evidence differs by method

The four analyses are scored in four different ways, and the comparison
between them is as much about that as about the numbers.

The classifier has the largest held-out partition of the four and the only
one with a grouping to respect: 24,511 encounters from patients the model never
saw, scored at a threshold chosen on training data. It was then rescored under
four more partitions to see how far the number moves. Its evidence is an
out-of-sample estimate with a measured spread, and every claim about it is a
claim about new patients. The cost of that standard is visible in the leakage stage. Balancing
the classes across the whole cohort and splitting afterwards, as the
coursework did, reports 0.806 for a model whose proper score is 0.595, with
12,057 patients and 125,252 duplicated rows across the boundary. Nothing in the
model changed; the evidence did.

The clustering has no held-out partition, and it cannot have one in the usual
sense, because the thing being estimated is a partition of the data itself.
Its external evidence is agreement with labels the method never saw, which is
the strongest check available and is still a check against one particular
labeling of one particular sample. Its internal evidence is an index computed
on the same points it clustered, and the index the coursework used prefers the
method that optimizes it. The two kinds of evidence disagree here by a factor
of two in one direction and a factor of two in the other, and a reader who had
only the internal index would have chosen the wrong method with confidence.

The rule miner has no held-out partition either, and its evidence is a count.
A confidence of 0.9965 is the fraction of 283 samples that are benign, and it
describes the table it was computed on; it is not an estimate of how often the
rule would hold on the next sample. The support threshold is the parameter
that decides what evidence can exist at all, and it decides silently: the
coursework reported rules about benign tumors and no rules about malignant
ones without the threshold ever being named as the reason.

The regression gives the most classical evidence of the four, a coefficient
with a confidence interval and a p-value, and the smallest sample. The interval
is an inferential statement conditional on assumptions the residuals fail, at
a skew of 1.14 and Jarque-Bera p = 2.8 × 10⁻²⁵, and the held-out R² on 86
subjects moves by ±0.1 when the split is redrawn. Every number the regression
produces is easier to interpret than any number the classifier produces, and
every one of them is less stable.

## What the coursework did that this rebuild measures

Each of the four course exercises carried one decision that the exercise took
for granted, and the rebuild keeps the decision as a comparison and reports
what it costs.

Balancing before splitting, in the classification exercise, adds 0.21 to the
reported ROC-AUC of the same network. Balancing at all, applied correctly
inside the training folds, subtracts 0.037 to 0.063 from it, and the rebuild
finds that out by searching the regime as a hyperparameter. The oversampling
gave the network 56,000 duplicated rows to fit and stopped its training loop
roughly five to nine times later, and the extra iterations fitted nothing that
transferred.

Selecting a clustering by Calinski-Harabasz, in the clustering exercise, picks
k-means at four clusters, which agrees with the labels least of the three
methods tried. Reporting silhouette and Davies-Bouldin beside it would have
shown the disagreement without any label being consulted. Clustering the
unstandardized panel, which the exercise also did, agrees with the labels
better than clustering the standardized one, 0.533 against 0.132. The rebuild
reports both and keeps the standardized result, because choosing the
preprocessing by agreement with withheld labels would make the labels part of
the method.

Mining at a support of 0.4, in the association rules exercise, excludes the
malignant class before the first itemset is counted. The rebuild keeps the
threshold so the results are comparable and sweeps it downward to show where
each class becomes reachable: one malignant itemset at 0.3, thirty-one at 0.2,
1,699 at 0.1.

Clustering the liver data with the reported intake among the features, in the
exercise the regression rebuilds, places the outcome inside the feature space
and reads a two-cluster solution as a split the data does not label. The
rebuild treats the data as the regression the UCI record describes. The
seventh column, which that exercise dropped, is the one most of the published
literature on this dataset has used as a disease label, and the rebuild
excludes it and checks the exclusion.

## Data

The three UCI datasets are committed under `data/raw/`, unmodified, as the
repository serves them; each is released under CC BY 4.0 and cited in full in
[docs/references.md](docs/references.md), and `data/README.md` carries the
attribution. `data/checksums.txt` records the SHA-256 digest of each file and
`data/download_data.py` verifies the committed files against it, fetching them
again from the same URL if they are absent. The breast cancer measurements ship
inside scikit-learn. The cleaned tables and the partitions every analysis
fitted on are committed under `data/processed/`, so the exact rows behind each
number are readable without running anything.

The course copies of the data were not used. For the readmission set the course
copy is a preprocessed derivative whose binning and encoding are undocumented,
and every cleaning decision here was re-derived from the original release and
is counted in the write-up.

## Reproducing it

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python data/download_data.py
.venv/Scripts/python analysis/run_all.py
```

`analysis/run_all.py` runs the four analyses and draws the twelve figures.
Each analysis writes its quantities into `results/metrics.csv` under its own
prefix, its tables into `results/`, and the frame it fitted on into
`data/processed/`. Every figure is drawn from one of those tables and computes
nothing of its own, so a figure and the number quoted beside it cannot
disagree. The shared loading, splitting, evaluation and plotting code is in
`src/`. The recorded run took 1,874 seconds on a fourteen-core CPU, of which
the classification's two grid searches and five repeated partitions took
1,860; the other three analyses finish in ten seconds between them.

## How it is checked

```
.venv/Scripts/python .checks/run_all_gates.py
.venv/Scripts/python .checks/inject_faults.py
```

Six gates guard the stages in order: environment, acquisition, schema,
preparation, modeling and reproducibility. Each exits non-zero on a failed
check and prints what it looked at. Gate 05 re-runs the whole pipeline into an
empty directory and compares every recorded quantity, every result table and
every processed table against the committed ones byte for byte; `SKIP_RERUN=1`
skips it and the summary says how many checks were skipped. `inject_faults.py`
breaks one thing at a time, a pinned version, a digest, the split flag, a
patient on both sides of the partition, a leaked feature, and confirms that the
gate guarding it fails. A gate that has never failed is not evidence.

## Citation

Taylor, P. *Four machine learning methods on health data.* 2026.
