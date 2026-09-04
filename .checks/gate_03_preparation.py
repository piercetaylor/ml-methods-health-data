#!/usr/bin/env python
"""Gate 03. The partitions do not overlap and the recorded counts add up.

Two overlaps are checked separately. No row of the training partition may be
identical to a row of the test partition, which catches a duplicated release
row placed on both sides. And for the readmission data no patient may appear on
both sides, which is the check the encounter-level table needs and a row-level
check would pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_lib as harness  # noqa: E402
from src import config  # noqa: E402

harness.gate("gate 03: preparation")
record = harness.metrics()

# --- the readmission partition ---------------------------------------------
harness.check("no encounter appears in both readmission partitions",
              record.number("m01.split.rows_on_both_sides") == 0,
              "{} identical rows across the boundary".format(
                  record.get("m01.split.rows_on_both_sides")))
harness.check("no patient appears in both readmission partitions",
              record.number("m01.split.groups_on_both_sides") == 0,
              "{} of {} training and {} test patients".format(
                  record.get("m01.split.groups_on_both_sides"),
                  record.get("m01.split.train_groups"),
                  record.get("m01.split.test_groups")))
harness.check("the readmission partition sizes add to the analyzed cohort",
              record.number("m01.split.train_rows")
              + record.number("m01.split.test_rows")
              == record.number("m01.rows_analyzed"),
              "{} + {} = {}".format(record.get("m01.split.train_rows"),
                                    record.get("m01.split.test_rows"),
                                    record.get("m01.rows_analyzed")))
harness.check("the test share is within one point of the configured fraction",
              abs(record.number("m01.split.test_fraction")
                  - config.TEST_FRACTION) <= 0.01,
              "configured {}, drawn {}".format(
                  config.TEST_FRACTION,
                  record.get("m01.split.test_fraction")))
# The partition is drawn over patients and is not stratified, so the two
# outcome rates are a property of the draw. A gap of more than two points would
# make the test partition a different problem from the training partition.
harness.check("the two readmission partitions carry a similar outcome rate",
              abs(record.number("m01.split.train_positive_rate")
                  - record.number("m01.split.test_positive_rate")) <= 0.02,
              "training {}, test {}".format(
                  record.get("m01.split.train_positive_rate"),
                  record.get("m01.split.test_positive_rate")))
harness.check("the class balance was corrected to an equal split",
              record.number("m01.balanced_train_rows")
              == round(record.number("m01.split.train_rows")
                       * (1 - record.number("m01.split.train_positive_rate"))) * 2,
              "{} rows after oversampling {} training rows at an outcome rate "
              "of {}".format(record.get("m01.balanced_train_rows"),
                             record.get("m01.split.train_rows"),
                             record.get("m01.split.train_positive_rate")))

# --- the readmission exclusions --------------------------------------------
harness.check("the recorded exclusions account for every released encounter",
              record.number("m01.rows_released")
              - record.number("m01.rows_unobservable_outcome")
              - record.number("m01.rows_incomplete")
              == record.number("m01.rows_analyzed"),
              "{} released, {} unobservable, {} incomplete, {} analyzed".format(
                  record.get("m01.rows_released"),
                  record.get("m01.rows_unobservable_outcome"),
                  record.get("m01.rows_incomplete"),
                  record.get("m01.rows_analyzed")))
harness.check("the weight column was as sparse as the exclusion claims",
              record.number("m01.missing_fraction_weight") >= 0.9,
              "absent in {} of the encounters".format(
                  record.get("m01.missing_fraction_weight")))
harness.check("the two dropped drug columns each held one value",
              record.number("m01.distinct_values_examide") == 1
              and record.number("m01.distinct_values_citoglipton") == 1,
              "examide {}, citoglipton {}".format(
                  record.get("m01.distinct_values_examide"),
                  record.get("m01.distinct_values_citoglipton")))

# --- the BUPA partition -----------------------------------------------------
harness.check("no subject appears in both BUPA partitions",
              record.number("m04.split.rows_on_both_sides") == 0,
              "{} identical rows across the boundary".format(
                  record.get("m04.split.rows_on_both_sides")))
harness.check("the BUPA partition sizes add to the analyzed rows",
              record.number("m04.split.train_rows")
              + record.number("m04.split.test_rows")
              == record.number("m04.rows_analyzed"),
              "{} + {} = {}".format(record.get("m04.split.train_rows"),
                                    record.get("m04.split.test_rows"),
                                    record.get("m04.rows_analyzed")))
harness.check("the duplicated BUPA rows were collapsed before the split",
              record.number("m04.rows_released")
              - record.number("m04.rows_duplicated")
              == record.number("m04.rows_analyzed"),
              "{} released less {} duplicated is {}".format(
                  record.get("m04.rows_released"),
                  record.get("m04.rows_duplicated"),
                  record.get("m04.rows_analyzed")))

# --- the two unsupervised sets ---------------------------------------------
# Neither holds a partition. What can be checked is that the exclusions were
# recorded and that the analyzed counts follow from them.
harness.check("the recorded HCV exclusion accounts for every released subject",
              record.number("m02.rows_released")
              - record.number("m02.rows_suspect_donor")
              == record.number("m02.rows_analyzed"),
              "{} released, {} suspect donors, {} analyzed"
              .format(record.get("m02.rows_released"),
                      record.get("m02.rows_suspect_donor"),
                      record.get("m02.rows_analyzed")))
harness.check("every HCV subject is either complete or imputed",
              record.number("m02.rows_complete")
              + record.number("m02.rows_incomplete")
              == record.number("m02.rows_analyzed"),
              "{} complete, {} imputed, {} analyzed".format(
                  record.get("m02.rows_complete"),
                  record.get("m02.rows_incomplete"),
                  record.get("m02.rows_analyzed")))
harness.check("the imputed HCV cells are few and are counted",
              0 < record.number("m02.cells_imputed")
              < 0.01 * record.number("m02.cells_total"),
              "{} of {} cells imputed".format(
                  record.get("m02.cells_imputed"),
                  record.get("m02.cells_total")))
harness.check("the imputed HCV rows are counted within each category",
              sum(record.number("m02.imputed_" + name)
                  for name in ("blood_donor", "hepatitis", "fibrosis",
                               "cirrhosis"))
              == record.number("m02.rows_incomplete"),
              "donors {}, hepatitis {}, fibrosis {}, cirrhosis {}".format(
                  record.get("m02.imputed_blood_donor"),
                  record.get("m02.imputed_hepatitis"),
                  record.get("m02.imputed_fibrosis"),
                  record.get("m02.imputed_cirrhosis")))
harness.check("the analyzed HCV categories sum to the analyzed subjects",
              sum(record.number("m02.analyzed_" + name)
                  for name in ("blood_donor", "hepatitis", "fibrosis",
                               "cirrhosis"))
              == record.number("m02.rows_analyzed"),
              "donors {}, hepatitis {}, fibrosis {}, cirrhosis {}".format(
                  record.get("m02.analyzed_blood_donor"),
                  record.get("m02.analyzed_hepatitis"),
                  record.get("m02.analyzed_fibrosis"),
                  record.get("m02.analyzed_cirrhosis")))
harness.check("the breast cancer classes sum to the analyzed samples",
              record.number("m03.rows_malignant")
              + record.number("m03.rows_benign")
              == record.number("m03.rows_analyzed"),
              "{} malignant and {} benign of {}".format(
                  record.get("m03.rows_malignant"),
                  record.get("m03.rows_benign"),
                  record.get("m03.rows_analyzed")))

harness.finish()
