#!/usr/bin/env python
"""Gate 02. Each cleaned table holds the columns and the types it should.

The check this gate exists for is the BUPA exclusion. The UCI record for that
dataset states that its seventh column, selector, has been widely
misinterpreted as a diagnosis when it is a train and test split flag. The
column is present in the released file, so the check that it is absent from the
model matrix and from the target is a check that can fail, and a change that let
it through would be caught here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_lib as harness  # noqa: E402
from src import config, data  # noqa: E402

harness.gate("gate 02: schema")

# --- BUPA, the excluded column ---------------------------------------------
released = pandas.read_csv(config.RAW / config.SOURCES["bupa"]["file"])
harness.check("the released BUPA file does carry the split flag",
              config.BUPA_EXCLUDED in released.columns,
              "released columns: {}".format(", ".join(released.columns)))

bupa, bupa_counts = data.load_bupa()
harness.check("the split flag is absent from the cleaned BUPA table",
              config.BUPA_EXCLUDED not in bupa.columns,
              "cleaned columns: {}".format(", ".join(bupa.columns)))
harness.check("the split flag is not among the predictors",
              config.BUPA_EXCLUDED not in config.BUPA_PREDICTORS,
              "predictors: {}".format(", ".join(config.BUPA_PREDICTORS)))
harness.check("the split flag is not the target",
              config.BUPA_TARGET != config.BUPA_EXCLUDED,
              "target: {}".format(config.BUPA_TARGET))
harness.check("every BUPA predictor and the target are present and numeric",
              all(name in bupa.columns
                  and pandas.api.types.is_numeric_dtype(bupa[name])
                  for name in tuple(config.BUPA_PREDICTORS)
                  + (config.BUPA_TARGET,)),
              "{} predictors and the target over {} rows".format(
                  len(config.BUPA_PREDICTORS), bupa_counts["rows_analyzed"]))
harness.check("the BUPA table holds no duplicated row",
              not bupa.duplicated().any(),
              "{} duplicated rows were collapsed".format(
                  bupa_counts["rows_duplicated"]))

# --- readmission ------------------------------------------------------------
diabetes, diabetes_counts = data.load_diabetes()
harness.check("the readmission label is binary",
              set(diabetes["readmitted_30d"].unique()) == {0, 1},
              "values: {}".format(sorted(int(value) for value
                                         in diabetes["readmitted_30d"].unique())))
harness.check("the three-level released label is gone",
              "readmitted" not in diabetes.columns,
              "columns naming readmission: {}".format(
                  [name for name in diabetes.columns if "readmit" in name]))
harness.check("no discharge disposition making the outcome unobservable "
              "survives",
              not diabetes["discharge_disposition_id"].isin(
                  ["code {}".format(code)
                   for code in config.UNOBSERVABLE_DISPOSITIONS]).any(),
              "{} encounters were removed".format(
                  diabetes_counts["rows_unobservable_outcome"]))
harness.check("the columns dropped for sparsity or constancy are gone",
              not (set(config.DIABETES_DROP_SPARSE)
                   | set(config.DIABETES_DROP_CONSTANT)) & set(diabetes.columns),
              "dropped: {}".format(", ".join(
                  tuple(config.DIABETES_DROP_SPARSE)
                  + tuple(config.DIABETES_DROP_CONSTANT))))
harness.check("the patient identifier is present for the grouped partition",
              "patient_nbr" in diabetes.columns,
              "{} patients over {} encounters".format(
                  diabetes_counts["patients_analyzed"],
                  diabetes_counts["rows_analyzed"]))
harness.check("no retained column holds a missing value",
              not diabetes.isna().any().any(),
              "{} rows were removed as incomplete".format(
                  diabetes_counts["rows_incomplete"]))
harness.check("the coded categories are carried as categories and not numbers",
              all(diabetes[name].dtype == object
                  for name in config.DIABETES_CODED_CATEGORIES),
              ", ".join(config.DIABETES_CODED_CATEGORIES))

# --- HCV --------------------------------------------------------------------
hcv, hcv_counts = data.load_hcv()
harness.check("every HCV assay the clustering uses is present and numeric",
              all(name in hcv.columns
                  and pandas.api.types.is_numeric_dtype(hcv[name])
                  for name in config.HCV_FEATURES),
              "{} assays over {} subjects".format(
                  len(config.HCV_FEATURES), hcv_counts["rows_analyzed"]))
harness.check("the misspelled release column was renamed",
              "GGT" in hcv.columns and "CGT" not in hcv.columns,
              "the release names gamma-glutamyl transferase CGT")
harness.check("no assay the clustering reads holds a missing value",
              not hcv[list(config.HCV_FEATURES)].isna().any().any(),
              "{} subjects were removed as incomplete".format(
                  hcv_counts["rows_incomplete"]))
harness.check("the withheld label holds the four released categories",
              hcv_counts["categories_analyzed"] == 4,
              "categories: {}".format(", ".join(
                  sorted(hcv["category_label"].unique()))))

# --- breast cancer ----------------------------------------------------------
cancer, cancer_counts = data.load_breast_cancer()
harness.check("the breast cancer table holds 30 features and a named diagnosis",
              cancer_counts["features"] == 30
              and set(cancer["diagnosis"].unique()) == {"malignant", "benign"},
              "{} malignant and {} benign".format(
                  cancer_counts["rows_malignant"],
                  cancer_counts["rows_benign"]))

harness.finish()
