# Four machine learning methods on health data

Four analyses on four public medical datasets, one per method family, built
on one shared pipeline. A feed-forward network predicts thirty-day hospital
readmission from 97,108 encounters (Strack et al. 2014). Three clustering
methods try to recover hepatitis C diagnostic categories from a ten-assay
laboratory panel for 608 subjects with the labels withheld (Lichtinghagen et
al. 2020). Apriori mines association rules among thirty cell-nucleus
measurements for 569 breast tumor samples (Wolberg et al. 1995). Least squares
predicts self-reported daily alcohol intake from a five-test liver enzyme panel
for 341 men (Forsyth 1990). Each analysis has a write-up in `docs/` and a
notebook in `notebooks/`. This page states the four methods, compares them,
and says what each one established here.

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

## The four methods

Each method is stated in the same four terms: what it takes in, what it
returns, which setting governs it, and how its output is judged.

### Supervised classification

The input is a table of encounters with 34 categorical and 9 numeric features
and a binary label, readmitted inside thirty days or not. The model is a
feed-forward network fitted by scikit-learn's `MLPClassifier`, with the
categories one-hot encoded and the counts standardized inside the same
pipeline. It returns a probability of readmission for each encounter, and a
threshold turns that into a decision. The governing settings are the
architecture and the regularization strength, which were searched over
sixteen configurations, and the training regime, balanced or unbalanced,
which was searched beside them. The output is judged on encounters from
patients the model never saw, by ROC-AUC, average precision and the Brier
score, and then again over five redrawn partitions.

### Clustering

The input is a 608 by 10 matrix of standardized laboratory values with the
diagnostic category removed. k-means, DBSCAN and agglomerative clustering with
complete linkage each return an integer cluster label per subject, and DBSCAN
also returns an outlier label. The governing setting is the number of clusters
for k-means and agglomerative clustering, and the neighborhood radius and
minimum count for DBSCAN. The output is judged two ways. Internal indices,
Calinski-Harabasz, silhouette and Davies-Bouldin, are computed from the
clustered points alone. External agreement, the adjusted Rand index and
adjusted mutual information, is computed against the withheld category after
the clustering is finished.

### Association rules

The input is a set of 569 transactions, one per tumor sample, each holding
the items that sample carries. An item is one of three levels of one of the
thirty measurements, or one of the two diagnosis labels, 92 items in all.
Apriori returns every itemset whose support clears a threshold, and from those
every rule whose conviction clears a second threshold. The governing setting is
the minimum support, which decides which itemsets can exist at all. The output
is judged by support, confidence, lift and conviction, each a count computed on
the same 569 transactions, with no held-out partition.

### Regression

The input is 341 rows of five blood tests and a continuous target, half-pints
of alcohol reported per day. Ordinary least squares returns a coefficient per
test with a confidence interval and a p-value, and a prediction per subject. A
random forest and a least squares fit on the log-transformed target are fitted
beside it. Least squares has no governing setting; the forest uses 500 trees
with a minimum leaf of five. The output is judged by the held-out coefficient
of determination on 86 subjects, by five-fold cross-validation inside the
training rows, and by twenty redraws of the partition.

## How the four compare

| | Classification | Clustering | Association rules | Regression |
|---|---|---|---|---|
| Supervision | Label used in fitting | Label withheld, used to score | Label carried as two ordinary items | Target used in fitting |
| Quantity estimated | Probability of readmission | A partition of the subjects | Frequent itemsets and rules | Coefficients and predictions |
| Held-out partition | 24,511 encounters, drawn over patients | None; the partition is the estimate | None; counts on all 569 | 86 subjects, redrawn twenty times |
| Governing setting | Architecture, alpha, training regime | Number of clusters; DBSCAN radius | Minimum support | None for least squares |
| Primary evidence | ROC-AUC 0.658 on held-out encounters | Adjusted Rand 0.815 at two clusters against cirrhosis | 321 rules, all naming the benign class | Held-out R² 0.216, mean 0.136 over redraws |
| Spread measured | Five patient-level partitions | Ten seeds; complete cases beside imputed | Support swept from 0.4 to 0.1 | Twenty row-level partitions |
| Where it failed here | Ceiling near 0.66 for any model | Hepatitis and fibrosis not recovered | Malignant class unreachable at support 0.4 | R² between −0.16 and 0.25 by draw |

The classification and the regression are supervised, and both are scored on
rows the model did not fit. The clustering and the rule mining are
unsupervised, and neither has a held-out partition, because what they estimate
is a property of the rows they were given. That difference decides what each
number means. A ROC-AUC of 0.658 is an estimate of how the network ranks new
patients, and five partitions put a standard deviation of 0.009 on it. An
adjusted Rand index of 0.815 is a fact about how one clustering of these 608
subjects lines up with one labeling of them. Its spread comes from the seed
and from which subjects are counted, since there are no new subjects to score
on. A confidence of 0.9965 is 282 divided by 283. An R² of 0.216 is one
draw from a distribution whose standard deviation, 0.098, is close to its
mean of 0.136.

The four also differ in what governs them. In the classification the setting
that mattered was outside the architecture grid: whether the classes were
balanced before training moved cross-validated ROC-AUC by more than the whole
grid did. In the clustering the number of clusters mattered most: at four clusters
the three methods spanned adjusted Rand indices of 0.13 to 0.31, and two
clusters reached 0.815. In the rule mining one
threshold decided which class could be described. In the regression nothing
governs the fit, and the spread comes entirely from which 86 rows were held
out.

## What each method established

### Classification

The network was searched over eight architectures and two regularization
strengths by three-fold cross-validation with the folds drawn over patients,
once with the minority class oversampled inside each training fold and once
on the natural distribution. The best configuration was three hidden layers
of 32 units at an alpha of 0.01, trained unbalanced, at a cross-validated
ROC-AUC of 0.658. Refitted on the 72,597 training encounters and scored on
24,511 encounters from 17,042 patients who appear nowhere in training, it
reached ROC-AUC 0.658, average precision 0.218 and a Brier score of 0.098,
against a readmission rate of 0.115. Logistic regression on the same features
and partition reached 0.658 and 0.219. Over five redrawn patient-level
partitions the two models averaged 0.662 each, and the network was ahead in
three draws by a mean of 0.0001.

The two features that carry the model are the count of the patient's
inpatient visits in the prior year, which lowers ROC-AUC by 0.069 when
permuted, and the discharge disposition, which lowers it by 0.032. No other
feature lowers it by more than 0.007. The glycated hemoglobin result, the
measurement the dataset was assembled to study, is not among the 25 features
with the largest decrease.

Balancing the classes before training was the one decision that moved the
result. Every unbalanced configuration in the grid scored above every balanced
one, and the gap between the worst unbalanced and the best balanced was 0.037.
On the selected architecture, balancing cost 0.063 of held-out ROC-AUC and
raised the Brier score from 0.098 to 0.234. The iteration counts show the
mechanism. Oversampling gave the network about 56,000 duplicated minority rows
to fit. The early-stopping loop ran for 90 to 155 iterations under the
balanced regime against 13 to 22 under the unbalanced one. The extra
iterations fitted duplicates of rows the model had already seen, which told it
nothing about the patients it would be scored on.

### Clustering

Two k-means clusters on the standardized panel place 23 of the 30 cirrhosis
cases in a cluster of 25, an adjusted Rand index of 0.815 against the
cirrhosis label. The 25-subject cluster has a median albumin of 32.0 against
42.2 in the rest, a median bilirubin of 40.0 against 7.1 and a median
cholinesterase of 2.47 against 8.39. Albumin and cholinesterase are made by the liver and fall when it fails,
and bilirubin is cleared by the liver and rises, so the cluster is the
laboratory picture of failing synthetic function, which advanced cirrhosis
produces.

Hepatitis and fibrosis were not recovered by any of the three methods at any
number of clusters tried. Their 45 subjects sit inside the two donor clusters
of the four-cluster solution, with no cluster holding a majority of either.
They differ from healthy donors on the panel by less than the donors differ
among themselves, so a distance-based method partitions the donors before it
isolates either group.

The internal index the coursework used to select a clustering disagreed with
the labels. Calinski-Harabasz is the ratio of between-cluster to
within-cluster dispersion, and k-means minimizes within-cluster dispersion
directly, so k-means scores 102.5 on that index against 45.5 for DBSCAN.
Agreement with the withheld labels runs the other way, 0.132 for k-means
against 0.310 for DBSCAN. Silhouette and Davies-Bouldin, computed with no
label, both rank k-means last of the three.

### Association rules

At the minimum support of 0.4 the coursework used, no frequent itemset can
contain the malignant diagnosis. An itemset cannot be more frequent than its
rarest item, and the malignant rate is 0.373. All 321 rules whose consequent is
the diagnosis therefore describe the benign class. The strongest names the lowest
level of mean concave points, of radius error and of worst perimeter. It holds
for 283 samples of which 282 are benign, a lift of 1.588 against the ceiling
of 1.594 that the benign rate allows.

The 19 items those rules select are all size and boundary-shape measurements.
Radius, perimeter, area, concavity and compactness each appear in all three
released forms, and texture, smoothness and symmetry appear in no rule. The
mining step gave the diagnosis no special standing among the 92 items, so the
selection was reached by counting co-occurrences. What it says is limited to
the part of the feature space where items are common enough to clear the
threshold, which on this data is the benign half.

### Regression

The five tests jointly relate to reported daily intake, with F = 10.07 and
p = 8.6 × 10⁻⁹ on the 255 training rows, and they explain little of it. The
held-out R² is 0.216 on the primary split of 86 subjects, 0.136 ± 0.098 over
twenty redrawn splits, and 0.058 ± 0.093 under five-fold cross-validation.
Mean corpuscular volume and gamma-glutamyl transpeptidase are the two terms
whose intervals exclude zero, at 0.167 and 0.020 half-pints per day per
released unit. These are the two assays used clinically to detect sustained
drinking, and the random forest ranks the same two first. A log-transformed
target scored lower, at 0.153, and the forest scored 0.235, so neither the
skew of the target nor the linear form is what limits the fit.

The seventh released column, `selector`, is a train and test split flag that
most of the published literature on this dataset has used as a disease label
(McDermott and Forsyth 2016). It is excluded here, and gate 02 checks that the
released file still carries it, that the cleaned table does not, and that it
is neither a predictor nor the target.

## What counts as evidence differs by method

The classifier has the largest held-out partition of the four and the only one
with a grouping to respect. Its 24,511 test encounters come from patients with
no encounter in training, they are scored at a threshold chosen on training
data, and the whole procedure was repeated under four more partitions. The
cost of that standard shows in the leakage stage. Balancing the classes across
the whole cohort and splitting afterwards by row, as the coursework did,
reports 0.806 for a model whose score under a proper partition is 0.595, with
12,057 patients and 125,252 duplicated rows on both sides of the boundary. The
model is the same in both rows of that comparison; what changed is which rows
it was scored on.

The clustering has no held-out partition and cannot have one in the usual
sense, because the partition of the data is the estimate. Its external
evidence is agreement with labels the method never saw, which is the
strongest check available and is still a check against one labeling of one
sample. Its internal evidence is an index computed on the same points it
clustered, and on this panel the index the coursework used ranks the three
methods in the opposite order from the labels. A reader with only the internal
index would have chosen k-means.

The rule miner has no held-out partition either, and its evidence is a count.
A confidence of 0.9965 is the fraction of 283 samples that are benign, and it
describes the table it was computed on. The support threshold decides what
evidence can exist. Here the coursework reported rules about benign tumors and
none about malignant ones, and the threshold was the reason without being
named as one.

The regression gives the most classical evidence of the four, a coefficient
with an interval and a p-value, and the smallest sample. The interval assumes
normal residuals, and the residuals have a skew of 1.14 with a Jarque-Bera
p-value of 2.8 × 10⁻²⁵, so the intervals are narrower than the sampling
uncertainty they stand for. The held-out R² on 86 subjects moves by about 0.1
when the split is redrawn. Every number the regression produces is easier to
read than any number the classifier produces, and every one of them moves
more when the data are resampled: a standard deviation of 0.098 across
redraws for the regression's R² against 0.009 for the classifier's ROC-AUC.

## What the coursework did that this rebuild measures

Each of the four course exercises carried one decision the exercise took for
granted. The rebuild keeps each decision as a comparison and reports what it
cost.

**Balancing before splitting**, in the classification exercise, adds 0.21 to
the reported ROC-AUC of the same network. Balancing at all, applied correctly
inside the training folds, subtracts 0.037 to 0.063 from it. The rebuild found
that out by searching the regime as a hyperparameter beside the architecture.
The exercise did not ask for that comparison. I added it because the
oversampling had been applied by default and never tested against its absence
on this data.

**Selecting a clustering by Calinski-Harabasz**, in the clustering exercise,
picks k-means at four clusters, which agrees with the labels least of the three
methods tried. Reporting silhouette and Davies-Bouldin beside it would have
shown the disagreement without consulting any label. The exercise also
clustered the unstandardized panel, which agrees with the labels better than
the standardized one, 0.533 against 0.132. The rebuild reports both and keeps
the standardized result, because choosing the preprocessing by agreement with
withheld labels would make the labels part of the method.

**Mining at a support of 0.4**, in the association rules exercise, excludes the
malignant class before the first itemset is counted. The rebuild keeps the
threshold so the results are comparable and sweeps it downward. One malignant
itemset becomes frequent at 0.3, thirty-one at 0.2 and 1,699 at 0.1.

**Clustering the liver data with the reported intake among the features**, in
the exercise the regression rebuilds, places the outcome inside the feature
space and reads a two-cluster solution as a split the data does not label. The
rebuild treats the data as the regression the UCI record describes. The seventh
column, which that exercise dropped, is the one most of the published
literature has used as a disease label, and the rebuild excludes it and checks
the exclusion.

## Data

The three UCI datasets are committed under `data/raw/`, unmodified, as the
repository serves them. Each is released under CC BY 4.0, cited in full in
[docs/references.md](docs/references.md), and attributed in `data/README.md`.
`data/checksums.txt` records the SHA-256 digest of each file, and
`data/download_data.py` verifies the committed files against it and fetches
any that are absent from the same URL. The breast cancer measurements ship
inside scikit-learn. The cleaned tables and the partitions every analysis
fitted on are committed under `data/processed/`, so the rows behind each
number can be read without running anything.

The course copies of the data were not used. For the readmission set the
course copy is a preprocessed derivative whose binning and encoding are
undocumented, so every cleaning decision here was re-derived from the original
release and is counted in the write-up.

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
nothing of its own. The shared loading, splitting, evaluation and plotting
code is in `src/`. The recorded run took 1,874 seconds on a fourteen-core CPU,
of which the classification's two grid searches and five repeated partitions
took 1,860.

The four notebooks under `notebooks/` walk through each analysis in the order
the coursework did, cleaning, inputs, training and results, with the write-up
between the cells. They call the same functions as the pipeline and write
nothing into `results/` or `data/processed/`. They need the kernel pinned in
`requirements-notebooks.txt`:

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
preparation, modeling and reproducibility. Each exits non-zero on a failed
check and prints what it looked at. Gate 05 re-runs the whole pipeline into an
empty directory and compares every recorded quantity, every result table and
every processed table against the committed ones byte for byte; `SKIP_RERUN=1`
skips it and the summary says how many checks were skipped. `inject_faults.py`
breaks one thing at a time, a pinned version, a digest, the split flag, a
patient on both sides of the partition, a leaked feature, and confirms that the
gate guarding it fails.

## Citation

Taylor, P. *Four machine learning methods on health data.* 2026.
