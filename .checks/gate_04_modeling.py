"""Gate 04. Every model recorded a result, and no result depends on the outcome.

The leakage check here is not a repetition of gate 03. Gate 03 shows that the
partitions do not share rows or patients. This gate shows that no single feature
separates the classes by itself, which is what a column recorded after the
outcome would do, and that the ordering defect the coursework carried does
inflate the result when it is reproduced deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_lib as harness  # noqa: E402
from src import config  # noqa: E402

# No single feature available at discharge should separate readmission on its
# own. The largest single-feature area on the training partition is the count
# of prior inpatient visits, and a value near one would mean a column carrying
# the outcome had survived the cleaning.
LEAKAGE_CEILING = 0.75

harness.gate("gate 04: modeling")
record = harness.metrics()

# --- model 1, readmission ---------------------------------------------------
harness.check("no single feature separates readmission on its own",
              record.number("m01.max_univariate_auc") < LEAKAGE_CEILING,
              "the largest is {} at {}, ceiling {}".format(
                  record.get("m01.max_univariate_auc_feature"),
                  record.get("m01.max_univariate_auc"), LEAKAGE_CEILING))
harness.check("the network beats chance on the held-out patients",
              record.number("m01.network.test_roc_auc") > 0.5,
              "test ROC-AUC {}".format(
                  record.get("m01.network.test_roc_auc")))
harness.check("average precision beats the readmission rate",
              record.number("m01.network.test_average_precision")
              > record.number("m01.network.test_positive_rate"),
              "average precision {} against a rate of {}".format(
                  record.get("m01.network.test_average_precision"),
                  record.get("m01.network.test_positive_rate")))
harness.check("the grid searched the number of hidden layers",
              len({row["hidden_layers"] for row in harness.table("m01_grid")}) > 1,
              "layer counts searched: {}".format(sorted(
                  {row["hidden_layers"] for row in harness.table("m01_grid")})))
harness.check("the selected configuration is the best the grid scored",
              record.number("m01.best_cv_roc_auc")
              == max(float(row["cv_roc_auc_mean"])
                     for row in harness.table("m01_grid")),
              "{} hidden layers of width {}, alpha {}, cv ROC-AUC {}".format(
                  record.get("m01.best_hidden_layers"),
                  record.get("m01.best_layer_widths"),
                  record.get("m01.best_alpha"),
                  record.get("m01.best_cv_roc_auc")))
harness.check("the confusion matrix sums to the held-out encounters",
              sum(record.number("m01.network.test_" + cell)
                  for cell in ("true_negative", "false_positive",
                               "false_negative", "true_positive"))
              == record.number("m01.network.test_rows"),
              "{} encounters".format(record.get("m01.network.test_rows")))
harness.check("accuracy is reported beside the majority-class rate",
              record.number("m01.network.test_majority_class_rate") > 0.8,
              "accuracy {} against a majority rate of {}".format(
                  record.get("m01.network.test_accuracy"),
                  record.get("m01.network.test_majority_class_rate")))

# The oversampling is a design choice, so its effect is measured and the
# measurement is checked to exist. The direction is not asserted here: a gate
# that required the balanced fit to lose would fail on data where it wins.
harness.check("the balanced and unbalanced fits of one architecture are both "
              "recorded",
              len(harness.table("m01_model_comparison")) == 3
              and record.number("m01.unbalanced.test_roc_auc") > 0.5,
              "balanced {}, unbalanced {}, logistic {} on test ROC-AUC".format(
                  record.get("m01.network.test_roc_auc"),
                  record.get("m01.unbalanced.test_roc_auc"),
                  record.get("m01.logistic.test_roc_auc")))
harness.check("the unbalanced fit was scored at the same threshold",
              record.number("m01.unbalanced.test_threshold")
              == record.number("m01.network.test_threshold"),
              "threshold {} from the {}".format(
                  record.get("m01.network.test_threshold"),
                  record.get("m01.threshold_source")))

# --- the ordering defect, reproduced ---------------------------------------
harness.check("balancing before the split does put patients on both sides",
              record.number("m01.leaked.patients_on_both_sides") > 0,
              "{} patients on both sides, and {} matching row pairs "
              "across that boundary".format(
                  record.get("m01.leaked.patients_on_both_sides"),
                  record.get("m01.leaked.rows_on_both_sides")))
harness.check("the ordering defect inflates the reported area",
              record.number("m01.leaked.test_roc_auc")
              > record.number("m01.network.test_roc_auc"),
              "{} under the defect against {} with patients held apart".format(
                  record.get("m01.leaked.test_roc_auc"),
                  record.get("m01.network.test_roc_auc")))

# --- model 2, clustering ----------------------------------------------------
# A clustering has no held-out accuracy, so nothing of the kind is checked
# here. What is checked is that every recorded index is defined, that the
# clustering formed the number of clusters it was asked for, and that agreement
# with the withheld labels beats chance.
for method in ("kmeans", "dbscan", "agglomerative"):
    harness.check("{} recorded a defined Calinski-Harabasz score".format(method),
                  record.number("m02." + method + ".calinski_harabasz") > 0,
                  "{}".format(record.get("m02." + method + ".calinski_harabasz")))
harness.check("k-means formed the number of clusters the label set implies",
              record.number("m02.kmeans.clusters") == config.HCV_K,
              "{} clusters over {} subjects".format(
                  record.get("m02.kmeans.clusters"),
                  record.get("m02.rows_analyzed")))
harness.check("the DBSCAN search kept a configuration that assigns most "
              "subjects",
              record.number("m02.dbscan.assigned_fraction") >= 0.8,
              "eps {}, min_samples {}, {} of the subjects assigned".format(
                  record.get("m02.dbscan.eps"),
                  record.get("m02.dbscan.min_samples"),
                  record.get("m02.dbscan.assigned_fraction")))
harness.check("every method beats chance agreement with the withheld labels",
              all(record.number("m02." + method + ".adjusted_rand") > 0
                  for method in ("kmeans", "dbscan", "agglomerative")),
              "adjusted Rand: k-means {}, DBSCAN {}, agglomerative {}".format(
                  record.get("m02.kmeans.adjusted_rand"),
                  record.get("m02.dbscan.adjusted_rand"),
                  record.get("m02.agglomerative.adjusted_rand")))
harness.check("the internal index and the label agreement disagree on the "
              "best method",
              record.number("m02.kmeans.calinski_harabasz")
              > record.number("m02.dbscan.calinski_harabasz")
              and record.number("m02.kmeans.adjusted_rand")
              < record.number("m02.dbscan.adjusted_rand"),
              "k-means leads on Calinski-Harabasz and DBSCAN on adjusted Rand")
harness.check("the two-cluster solution against cirrhosis is the strongest "
              "recovery",
              record.number("m02.kmeans_k2_vs_cirrhosis.adjusted_rand")
              > record.number("m02.kmeans.adjusted_rand"),
              "{} at two clusters against {} at {}".format(
                  record.get("m02.kmeans_k2_vs_cirrhosis.adjusted_rand"),
                  record.get("m02.kmeans.adjusted_rand"), config.HCV_K))

# --- model 3, association rules --------------------------------------------
harness.check("the mining run found frequent itemsets and rules",
              record.number("m03.frequent_itemsets") > 0
              and record.number("m03.rules") > 0,
              "{} itemsets at support {}, {} rules at conviction {}".format(
                  record.get("m03.frequent_itemsets"), config.MIN_SUPPORT,
                  record.get("m03.rules"), config.MIN_CONVICTION))
harness.check("the malignant class is out of reach at the chosen support",
              record.get("m03.malignant_reachable_at_min_support") == "False",
              "the malignant rate is {} and the support threshold is {}".format(
                  record.get("m03.malignant_rate"), config.MIN_SUPPORT))
harness.check("no frequent itemset names the malignant class at that support",
              all(int(row["itemsets_with_malignant"]) == 0
                  for row in harness.table("m03_support_sweep")
                  if float(row["min_support"]) >= record.number(
                      "m03.malignant_rate")),
              "the sweep records where each class becomes reachable")
harness.check("every diagnosis rule beats the base rate of its consequent",
              all(float(row["lift"]) > 1
                  for row in harness.table("m03_top_rules")),
              "the smallest lift among the recorded rules is {:.4f}".format(
                  min(float(row["lift"])
                      for row in harness.table("m03_top_rules"))))

# --- model 4, regression ----------------------------------------------------
harness.check("the excluded BUPA column is not among the fitted terms",
              config.BUPA_EXCLUDED not in {row["term"]
                                           for row in
                                           harness.table("m04_coefficients")},
              "terms: {}".format(", ".join(
                  row["term"] for row in harness.table("m04_coefficients"))))
harness.check("the panel as a whole relates to daily intake",
              record.number("m04.ols.f_p_value") < 0.05,
              "F = {} on the training partition, p = {}".format(
                  record.get("m04.ols.f_statistic"),
                  record.get("m04.ols.f_p_value")))
harness.check("the cross-validated fit is recorded beside the single split",
              record.number("m04.ols.cv_r2_mean")
              < record.number("m04.ols.test_r2"),
              "cross-validated {} plus or minus {} against {} on one split"
              .format(record.get("m04.ols.cv_r2_mean"),
                      record.get("m04.ols.cv_r2_sd"),
                      record.get("m04.ols.test_r2")))
harness.check("the residual normality the intervals assume was tested",
              record.number("m04.ols.residual_jarque_bera") > 0,
              "Jarque-Bera {} with p = {} and skew {}".format(
                  record.get("m04.ols.residual_jarque_bera"),
                  record.get("m04.ols.residual_jarque_bera_p"),
                  record.get("m04.ols.residual_skew")))
harness.check("the root mean squared error is below the target deviation",
              record.number("m04.ols.test_rmse")
              < record.number("m04.ols.test_sd_observed"),
              "RMSE {} against an observed deviation of {} half-pints".format(
                  record.get("m04.ols.test_rmse"),
                  record.get("m04.ols.test_sd_observed")))

# --- every recorded result table -------------------------------------------
for name in ("m01_grid", "m01_model_comparison", "m01_roc_curve",
             "m01_pr_curve", "m01_confusion", "m01_permutation_importance",
             "m01_leakage_comparison", "m01_univariate_auc",
             "m02_model_comparison", "m02_cluster_sweep", "m02_dbscan_grid",
             "m02_contingency_k4", "m02_contingency_k2", "m02_group_profile",
             "m02_feature_variance", "m02_projection", "m03_bin_edges",
             "m03_support_sweep", "m03_itemsets_by_length", "m03_top_rules",
             "m03_selected_items", "m04_coefficients", "m04_model_comparison",
             "m04_forest_importance", "m04_predictions"):
    harness.check("results/{}.csv holds rows".format(name),
                  lambda name=name: len(harness.table(name)) > 0,
                  lambda name=name: "{} rows".format(len(harness.table(name))))

harness.finish()
