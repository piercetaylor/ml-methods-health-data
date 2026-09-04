"""Paths, constants and the random seeds shared by the four analyses.

Every module imports its parameters from here, so a quantity recorded in
``results/metrics.csv`` can be traced to the one place its inputs were set.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"

# The two output directories may be redirected by an environment variable. The
# reproducibility gate uses that to run the whole pipeline into a scratch
# directory and compare the result against the committed one, without moving
# the committed files aside first.
RESULTS = Path(os.environ.get("ML_METHODS_RESULTS", str(ROOT / "results")))
FIGURES = Path(os.environ.get("ML_METHODS_FIGURES", str(ROOT / "figures")))
# The cleaned tables and the partition memberships every analysis reads,
# written by the pipeline and committed beside the results they produced.
PROCESSED = Path(os.environ.get("ML_METHODS_PROCESSED",
                                str(DATA / "processed")))

CHECKSUMS = DATA / "checksums.txt"
METRICS = RESULTS / "metrics.csv"

# One seed governs every stochastic step in every analysis: the train and test
# split, the k-means initializations, the discretizer, and the network weights.
# The headline of each analysis is produced at SEED. SEED_LIST holds the seeds
# the repeated-partition comparisons run over, and SEED is the first of them, so
# the run at the primary seed is one of the repeats and not a separate draw.
SEED = 20251206
SEED_LIST = tuple(SEED + offset for offset in range(5))

# --- data sources ----------------------------------------------------------
# The three downloaded sets are served by the UCI Machine Learning Repository
# as a single CSV each, at a stable URL keyed by the dataset's numeric id. The
# fourth ships inside scikit-learn and needs no download.
UCI_STATIC = "https://archive.ics.uci.edu/static/public/{id}/data.csv"

SOURCES = {
    "diabetes": {
        "uci_id": 296,
        "file": "diabetes_130_us_hospitals.csv",
        "citation": "Clore et al. (2014)",
        "expected_rows": 101766,
        "expected_columns": 50,
    },
    "hcv": {
        "uci_id": 571,
        "file": "hcv_data.csv",
        "citation": "Lichtinghagen et al. (2020)",
        "expected_rows": 615,
        "expected_columns": 14,
    },
    "bupa": {
        "uci_id": 60,
        "file": "liver_disorders.csv",
        "citation": "Forsyth (1990)",
        "expected_rows": 345,
        "expected_columns": 7,
    },
}

# --- model 1, readmission classification -----------------------------------
# The outcome is readmission inside thirty days. The released label carries
# three levels, and the two that are not "<30" are both "not readmitted inside
# thirty days" for this question.
READMIT_POSITIVE = "<30"

# Discharge dispositions that make the outcome unobservable. A patient who died
# or entered hospice cannot be readmitted, so an encounter with one of these
# codes contributes a guaranteed negative that the model could learn from the
# disposition alone. Codes 11, 19, 20 and 21 are the expired dispositions and
# 13 and 14 are the two hospice dispositions in the released mapping.
UNOBSERVABLE_DISPOSITIONS = (11, 13, 14, 19, 20, 21)

# Columns dropped before modeling, each for a stated reason the loader counts.
# `weight` is absent in almost every row, and `payer_code` is the administrative
# identifier of the paying insurer, absent in two rows in five and carrying no
# clinical measurement. `examide` and `citoglipton` take one value in every row.
DIABETES_DROP_SPARSE = ("weight", "payer_code")
DIABETES_DROP_CONSTANT = ("examide", "citoglipton")

# Released as integer keys into a published lookup table. The integers carry no
# order, so these are treated as categories.
DIABETES_CODED_CATEGORIES = ("admission_type_id", "discharge_disposition_id",
                             "admission_source_id")

# The admitting department is recorded under 72 distinct names, and the tail of
# that list holds departments seen in a handful of encounters. A level below
# this share of the recorded values is collapsed into one "other" level, so the
# encoding does not carry columns that are almost always zero. An encounter
# with no department recorded keeps its own level, because the absence is a
# property of how the encounter was documented.
SPECIALTY_MIN_SHARE = 0.01

# ICD-9 groups follow the scheme Strack et al. (2014) published with the data.
# A code outside every listed interval, including the V and E supplementary
# codes, falls to "other".
ICD9_GROUPS = (
    ("circulatory", ((390, 459), (785, 785))),
    ("respiratory", ((460, 519), (786, 786))),
    ("digestive", ((520, 579), (787, 787))),
    ("diabetes", ((250, 250),)),
    ("injury", ((800, 999),)),
    ("musculoskeletal", ((710, 739),)),
    ("genitourinary", ((580, 629), (788, 788))),
    ("neoplasms", ((140, 239),)),
)

TEST_FRACTION = 0.25
CV_FOLDS = 3

# The grid searched for the network. The course exercise required the number of
# hidden layers to be one searched parameter, and that requirement is kept. Each
# layer holds the same width, and the widths are set relative to nothing but the
# grid, so the search is over depth, width and regularization strength.
MLP_GRID = {
    "hidden_layer_sizes": ((32,), (32, 32), (32, 32, 32), (64, 64)),
    "alpha": (1e-4, 1e-2),
}
MLP_MAX_ITER = 200
PERMUTATION_REPEATS = 10

# The grid is searched once under each training regime. Balanced oversamples
# the minority class to parity inside each training fold, which is what the
# coursework did; unbalanced trains on the natural class distribution. The
# headline configuration is the best cross-validated score across both, so the
# choice of regime is a measured result and not a premise.
TRAINING_REGIMES = ("balanced", "unbalanced")

# --- model 2, HCV clustering ------------------------------------------------
HCV_FEATURES = ("ALB", "ALP", "ALT", "AST", "BIL", "CHE",
                "CHOL", "CREA", "GGT", "PROT")
HCV_CLUSTER_RANGE = (2, 3, 4, 5, 6)
# Subjects missing an assay are imputed from their nearest neighbors on the
# assays they do hold, on the standardized scale, with no reference to the
# category. The complete-case result is recorded beside it as a sensitivity.
HCV_IMPUTE_NEIGHBORS = 5
HCV_STABILITY_SEEDS = 10
HCV_K = 4          # the number of released diagnostic categories
DBSCAN_EPS_GRID = tuple(round(0.1 * step, 2) for step in range(5, 41))
DBSCAN_MIN_SAMPLES = (3, 4, 5, 6, 8, 10)

# --- model 3, association rules --------------------------------------------
N_BINS = 3
MIN_SUPPORT = 0.4
MIN_CONVICTION = 10.0
MAX_ITEMSET_LEN = 4

# --- model 4, BUPA regression ----------------------------------------------
# The seventh released column. It is a train and test split flag and not a
# clinical variable, and it is excluded from the predictors and from the target.
BUPA_EXCLUDED = "selector"
BUPA_PREDICTORS = ("mcv", "alkphos", "sgpt", "sgot", "gammagt")
BUPA_TARGET = "drinks"
# One split of 341 rows gives one test score with a wide sampling distribution,
# so the split is repeated this many times and the spread is recorded.
BUPA_SPLIT_REPEATS = 20

# --- reproducibility --------------------------------------------------------
# Wall-clock timings differ between two runs that agree on every measured
# quantity, so they are excluded from the signature the re-run reproduces.
RECORD_TIMING_PREFIX = "timing."
RECORD_SIGNATURE_DIGITS = 12
