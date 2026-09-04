#!/usr/bin/env python
"""Model 1. Predicting readmission inside thirty days, with explanations.

    python analysis/m01_classification.py

The Diabetes 130-US Hospitals data (Strack et al. 2014, Clore et al. 2014) holds
101,766 inpatient encounters of diabetic patients over ten years. The outcome is
readmission inside thirty days of discharge, which 11.5 percent of the analyzed
encounters record. The prediction is made at discharge, so every feature the
model reads is available at that point.

Two properties of the data decide the design. The same patient contributes
several encounters, so the partition is drawn over patients and a patient never
appears on both sides of it. The outcome is rare, and whether to correct that by
oversampling is treated as a hyperparameter: the grid is searched once with the
minority class oversampled inside each training fold and once on the natural
distribution, and the headline is the best configuration across both.

A feed-forward network is fitted, with the number of hidden layers among the
searched hyperparameters, and logistic regression is fitted beside it as the
reference. The headline configuration is then refitted under five repeated
patient-level partitions, so its score is read against its own spread. The last
stage reproduces the ordering defect the coursework carried, balancing the
classes before splitting, and measures how much it inflates the result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy
import pandas
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import resample

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluate, splits, utils  # noqa: E402

PREFIX = "m01."
LABEL = "readmitted_30d"
GROUP = "patient_nbr"
NOT_FEATURES = ("encounter_id", GROUP, LABEL)
SCORED = ("roc_auc", "average_precision", "accuracy", "balanced_accuracy",
          "precision", "recall", "f1", "brier")


def columns_of(frame: pandas.DataFrame) -> tuple[list[str], list[str], list[str]]:
    features = [name for name in frame.columns if name not in NOT_FEATURES]
    categorical = [name for name in features if frame[name].dtype == object]
    numeric = [name for name in features if name not in categorical]
    return features, categorical, numeric


def pipeline(estimator, categorical: list[str], numeric: list[str]) -> Pipeline:
    """One-hot encode the categories, standardize the counts, then the model.

    The encoder and the scaler are steps of the pipeline and are therefore
    fitted on whatever partition the pipeline is fitted on. A scaler fitted on
    the whole cohort would carry the test partition's means into training.
    A category seen only in the test partition encodes to all zeros, which is
    the intended behavior for a level the training data never held.
    """
    return Pipeline([
        ("prepare", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore", drop="first",
                                          sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric)])),
        ("model", estimator)])


def network(parameters: dict) -> MLPClassifier:
    return MLPClassifier(max_iter=config.MLP_MAX_ITER, early_stopping=True,
                         n_iter_no_change=10, random_state=config.SEED,
                         **parameters)


def balance(block: pandas.DataFrame, seed: int) -> pandas.DataFrame:
    """Oversample the minority class of one block up to the majority count.

    This is applied to a training fold or to a training partition, and never
    to anything a result is read from. It is the correction the coursework
    applied, and here it is one of two training regimes the grid is searched
    under, so whether it helps is measured and not assumed.
    """
    majority = block[block[LABEL] == 0]
    minority = block[block[LABEL] == 1]
    upsampled = resample(minority, replace=True, n_samples=len(majority),
                         random_state=seed)
    return (pandas.concat([majority, upsampled])
            .sample(frac=1, random_state=seed).reset_index(drop=True))


def prepared(block: pandas.DataFrame, regime: str) -> pandas.DataFrame:
    """The training block as one regime sees it."""
    return balance(block, config.SEED) if regime == "balanced" else block


def univariate_auc(frame: pandas.DataFrame, features: list[str]) -> list[dict]:
    """The ROC-AUC of each feature taken alone, as a check for a leaked outcome.

    A feature recorded after the outcome, or derived from it, separates the two
    classes by itself. Each feature is scored alone on the training partition
    and the largest is recorded. A categorical feature is scored by the outcome
    rate of its own levels, which is the best a single split on it can do.
    """
    target = frame[LABEL].to_numpy()
    rows = []
    for name in features:
        column = frame[name]
        if column.dtype == object:
            rates = frame.groupby(name)[LABEL].mean()
            score = column.map(rates).to_numpy(dtype=float)
        else:
            score = column.to_numpy(dtype=float)
        area = roc_auc_score(target, score)
        rows.append({"feature": name,
                     "auc": round(float(max(area, 1 - area)), 6)})
    return sorted(rows, key=lambda row: -row["auc"])


def search(train: pandas.DataFrame, features: list[str], categorical: list[str],
           numeric: list[str]) -> tuple[dict, list[dict], numpy.ndarray]:
    """Grid search the network by grouped cross-validation of the training set.

    The folds are drawn over patients on the unbalanced training partition and
    are the same folds under both regimes, so a difference between the regimes
    is not a difference between draws. Under the balanced regime the minority
    class is oversampled inside each training fold only. Balancing the training
    partition first and folding it afterwards would place copies of one
    encounter in the training fold and the validation fold together, which is
    the defect the last stage of this script measures.

    The out-of-fold scores of the winning configuration are returned, so the
    decision threshold can be chosen without touching the test partition.
    """
    folds = list(StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True,
                                      random_state=config.SEED)
                 .split(train, train[LABEL], groups=train[GROUP]))
    grid = [{"hidden_layer_sizes": size, "alpha": alpha}
            for size in config.MLP_GRID["hidden_layer_sizes"]
            for alpha in config.MLP_GRID["alpha"]]

    rows = []
    best = None
    best_out_of_fold = None
    for regime in config.TRAINING_REGIMES:
        for parameters in grid:
            out_of_fold = numpy.full(len(train), numpy.nan)
            scores = []
            iterations = []
            for fold_train, fold_validation in folds:
                model = pipeline(network(parameters), categorical, numeric)
                fold = prepared(train.iloc[fold_train], regime)
                model.fit(fold[features], fold[LABEL])
                iterations.append(model.named_steps["model"].n_iter_)
                predicted = model.predict_proba(
                    train.iloc[fold_validation][features])[:, 1]
                out_of_fold[fold_validation] = predicted
                scores.append(roc_auc_score(
                    train.iloc[fold_validation][LABEL], predicted))
            row = {"training": regime,
                   "hidden_layers": len(parameters["hidden_layer_sizes"]),
                   "layer_widths": "-".join(str(width) for width
                                            in parameters["hidden_layer_sizes"]),
                   "alpha": parameters["alpha"],
                   "cv_roc_auc_mean": round(float(numpy.mean(scores)), 6),
                   "cv_roc_auc_sd": round(float(numpy.std(scores, ddof=1)), 6),
                   "mean_iterations": round(float(numpy.mean(iterations)), 1)}
            rows.append(row)
            print("  {:<10} {:>10} alpha={:<8} cv ROC-AUC {:.4f}".format(
                regime, row["layer_widths"], row["alpha"],
                row["cv_roc_auc_mean"]))
            if best is None or (row["cv_roc_auc_mean"]
                                > best["row"]["cv_roc_auc_mean"]):
                best = {"row": row, "parameters": parameters, "regime": regime}
                best_out_of_fold = out_of_fold
    return best, rows, best_out_of_fold


def main() -> int:
    record = utils.Metrics()
    with utils.Timer(record, "m01"):
        frame, counts = data.load_diabetes()
        record.update(counts, PREFIX)
        features, categorical, numeric = columns_of(frame)
        record.set(PREFIX + "features_categorical", len(categorical))
        record.set(PREFIX + "features_numeric", len(numeric))

        train, test = splits.grouped_split(frame, GROUP)
        record.update(splits.report(train, test, GROUP, LABEL),
                      PREFIX + "split.")
        # The partition is published beside the cleaned cohort, so the exact
        # training and test membership behind every number can be re-read.
        utils.write_processed(
            pandas.concat([train[["encounter_id", GROUP]].assign(
                              partition="train"),
                           test[["encounter_id", GROUP]].assign(
                              partition="test")]),
            "readmission_partition")

        # --- leakage audit ---------------------------------------------------
        audit = univariate_auc(train, features)
        utils.write_table(audit[:20], "m01_univariate_auc")
        record.set(PREFIX + "max_univariate_auc", audit[0]["auc"])
        record.set(PREFIX + "max_univariate_auc_feature", audit[0]["feature"])

        # --- the grid search -------------------------------------------------
        print("grid search over {} configurations, {} regimes, {} folds "
              "each".format(len(config.MLP_GRID["hidden_layer_sizes"])
                            * len(config.MLP_GRID["alpha"]),
                            len(config.TRAINING_REGIMES), config.CV_FOLDS))
        best, grid_rows, out_of_fold = search(train, features, categorical,
                                              numeric)
        utils.write_table(grid_rows, "m01_grid")
        regime = best["regime"]
        other = [name for name in config.TRAINING_REGIMES if name != regime][0]
        record.set(PREFIX + "best_training", regime)
        record.set(PREFIX + "best_hidden_layers", best["row"]["hidden_layers"])
        record.set(PREFIX + "best_layer_widths", best["row"]["layer_widths"])
        record.set(PREFIX + "best_alpha", best["row"]["alpha"])
        record.set(PREFIX + "best_cv_roc_auc", best["row"]["cv_roc_auc_mean"])
        record.set(PREFIX + "best_cv_roc_auc_sd", best["row"]["cv_roc_auc_sd"])
        for name in config.TRAINING_REGIMES:
            top = max((row for row in grid_rows if row["training"] == name),
                      key=lambda row: row["cv_roc_auc_mean"])
            record.set(PREFIX + "best_cv_roc_auc_" + name, top["cv_roc_auc_mean"])
            record.set(PREFIX + "best_layer_widths_" + name, top["layer_widths"])

        # The threshold is chosen on the out-of-fold scores of the training
        # partition under the winning regime and applied unchanged to every
        # model scored on the test partition.
        threshold = evaluate.best_threshold(train[LABEL], out_of_fold)
        record.set(PREFIX + "threshold_source",
                   "training out-of-fold scores, {} regime".format(regime))

        # --- refit and score -------------------------------------------------
        balanced = balance(train, config.SEED)
        record.set(PREFIX + "balanced_train_rows", len(balanced))
        blocks = {"balanced": balanced, "unbalanced": train}

        headline = pipeline(network(best["parameters"]), categorical, numeric)
        headline.fit(blocks[regime][features], blocks[regime][LABEL])
        record.set(PREFIX + "network.training", regime)
        record.set(PREFIX + "network.iterations",
                   int(headline.named_steps["model"].n_iter_))
        test_score = headline.predict_proba(test[features])[:, 1]
        record.update(evaluate.classification(test[LABEL], test_score, threshold),
                      PREFIX + "network.test_")

        # --- the same architecture under the other regime --------------------
        # Scored on the same held-out encounters at the same threshold, so the
        # difference between the two rows is the training regime alone.
        alternate = pipeline(network(best["parameters"]), categorical, numeric)
        alternate.fit(blocks[other][features], blocks[other][LABEL])
        record.set(PREFIX + "alternate.training", other)
        record.set(PREFIX + "alternate.iterations",
                   int(alternate.named_steps["model"].n_iter_))
        record.update(
            evaluate.classification(
                test[LABEL], alternate.predict_proba(test[features])[:, 1],
                threshold),
            PREFIX + "alternate.test_")

        # --- the reference model ---------------------------------------------
        # Logistic regression on the same features, the same partition and the
        # same regime as the headline. It is reported because a network that
        # does not beat it has not earned the extra capacity.
        reference = pipeline(LogisticRegression(max_iter=1000,
                                                random_state=config.SEED),
                             categorical, numeric)
        reference.fit(blocks[regime][features], blocks[regime][LABEL])
        record.set(PREFIX + "logistic.training", regime)
        record.update(
            evaluate.classification(
                test[LABEL], reference.predict_proba(test[features])[:, 1],
                threshold),
            PREFIX + "logistic.test_")

        utils.write_table(
            [{"model": name,
              **{key: round(record.number(PREFIX + key_prefix + key), 6)
                 for key in SCORED}}
             for name, key_prefix in (
                 ("network, {} training, selected".format(regime),
                  "network.test_"),
                 ("network, {} training".format(other), "alternate.test_"),
                 ("logistic regression, {} training".format(regime),
                  "logistic.test_"))],
            "m01_model_comparison")

        # --- the curves and the confusion matrix -----------------------------
        false_positive, true_positive, _ = roc_curve(test[LABEL], test_score)
        step = max(1, len(false_positive) // 500)
        utils.write_table(
            [{"false_positive_rate": round(float(x), 6),
              "true_positive_rate": round(float(y), 6)}
             for x, y in zip(false_positive[::step], true_positive[::step])],
            "m01_roc_curve")
        precision, recall, _ = precision_recall_curve(test[LABEL], test_score)
        step = max(1, len(precision) // 500)
        utils.write_table(
            [{"recall": round(float(x), 6), "precision": round(float(y), 6)}
             for x, y in zip(recall[::step], precision[::step])],
            "m01_pr_curve")
        utils.write_table(
            [{"observed": observed, "predicted": predicted,
              "encounters": int(record.number(PREFIX + "network.test_" + key))}
             for observed, predicted, key in (
                 ("not readmitted", "not readmitted", "true_negative"),
                 ("not readmitted", "readmitted", "false_positive"),
                 ("readmitted", "not readmitted", "false_negative"),
                 ("readmitted", "readmitted", "true_positive"))],
            "m01_confusion")

        # --- explanation -----------------------------------------------------
        # Permutation importance measures the fall in test ROC-AUC when one
        # column is shuffled, so it reports what the fitted model uses and not
        # what the outcome depends on. A correlated pair can share its
        # importance and both appear unimportant. It is measured on the whole
        # test partition.
        record.set(PREFIX + "importance_rows", len(test))
        importance = permutation_importance(
            headline, test[features], test[LABEL],
            scoring="roc_auc", n_repeats=config.PERMUTATION_REPEATS,
            random_state=config.SEED, n_jobs=1)
        utils.write_table(
            sorted([{"feature": name,
                     "mean_decrease": round(float(mean), 6),
                     "sd": round(float(deviation), 6)}
                    for name, mean, deviation in zip(
                        features, importance.importances_mean,
                        importance.importances_std)],
                   key=lambda row: -row["mean_decrease"])[:25],
            "m01_permutation_importance")

        # --- the same configuration under repeated partitions ----------------
        # One partition gives one score. The patient-level split is redrawn at
        # each seed in SEED_LIST, the primary seed first, and the headline
        # network and the reference are refitted under the headline regime and
        # scored on each draw. The row at the primary seed is the fit above
        # and must reproduce it, which the modeling gate checks.
        repeats = []
        for seed in config.SEED_LIST:
            again_train, again_test = splits.grouped_split(frame, GROUP,
                                                           seed=seed)
            again_block = (balance(again_train, config.SEED)
                           if regime == "balanced" else again_train)
            for name, estimator in (
                    ("network", network(best["parameters"])),
                    ("logistic", LogisticRegression(max_iter=1000,
                                                    random_state=config.SEED))):
                model = pipeline(estimator, categorical, numeric)
                model.fit(again_block[features], again_block[LABEL])
                scored = evaluate.classification(
                    again_test[LABEL],
                    model.predict_proba(again_test[features])[:, 1], threshold)
                repeats.append({"seed": seed, "model": name,
                                "test_rows": scored["rows"],
                                "test_positive_rate": round(
                                    scored["positive_rate"], 6),
                                "roc_auc": round(scored["roc_auc"], 6),
                                "average_precision": round(
                                    scored["average_precision"], 6)})
        utils.write_table(repeats, "m01_repeated_splits")
        record.set(PREFIX + "repeats.count", len(config.SEED_LIST))
        for name in ("network", "logistic"):
            for measure in ("roc_auc", "average_precision"):
                values = numpy.array([row[measure] for row in repeats
                                      if row["model"] == name])
                key = PREFIX + "repeats." + name + "." + measure
                record.set(key + "_mean", float(values.mean()))
                record.set(key + "_sd", float(values.std(ddof=1)))
                record.set(key + "_min", float(values.min()))
                record.set(key + "_max", float(values.max()))
        record.set(PREFIX + "repeats.network.roc_auc_at_primary_seed",
                   [row["roc_auc"] for row in repeats
                    if row["model"] == "network"][0])
        differences = [
            next(row["roc_auc"] for row in repeats
                 if row["seed"] == seed and row["model"] == "network")
            - next(row["roc_auc"] for row in repeats
                   if row["seed"] == seed and row["model"] == "logistic")
            for seed in config.SEED_LIST]
        record.set(PREFIX + "repeats.network_minus_logistic_roc_auc_mean",
                   float(numpy.mean(differences)))
        record.set(PREFIX + "repeats.network_wins",
                   int(sum(value > 0 for value in differences)))

        # --- the ordering defect, measured -----------------------------------
        # The coursework balanced the classes over the whole cohort and split
        # the balanced table afterwards, by row. Each oversampled encounter is
        # then a duplicate of a training row, and the same patient's other
        # encounters sit on both sides regardless. The identical architecture
        # is fitted under that ordering and scored on the partition that
        # ordering produces. Its counterpart is the balanced fit under the
        # proper partition, whichever of the two fits above that was, so the
        # two rows differ in the ordering and in nothing else.
        counterpart = "network" if regime == "balanced" else "alternate"
        record.set(PREFIX + "leaked.counterpart", counterpart)
        leaked = balance(frame, config.SEED)
        leaked_train, leaked_test = splits.row_split(leaked)
        leaked_model = pipeline(network(best["parameters"]), categorical,
                                numeric)
        leaked_model.fit(leaked_train[features], leaked_train[LABEL])
        leaked_score = leaked_model.predict_proba(leaked_test[features])[:, 1]
        record.update(
            evaluate.classification(leaked_test[LABEL], leaked_score, 0.5),
            PREFIX + "leaked.test_")
        record.set(PREFIX + "leaked.rows_on_both_sides",
                   int(pandas.merge(leaked_train, leaked_test,
                                    how="inner").shape[0]))
        record.set(PREFIX + "leaked.patients_on_both_sides",
                   len(set(leaked_train[GROUP]) & set(leaked_test[GROUP])))
        utils.write_table(
            [{"partition": "patients held apart, balanced training",
              "roc_auc": round(
                  record.number(PREFIX + counterpart + ".test_roc_auc"), 6),
              "accuracy": round(
                  record.number(PREFIX + counterpart + ".test_accuracy"), 6),
              "patients_on_both_sides": int(
                  record.number(PREFIX + "split.groups_on_both_sides"))},
             {"partition": "balanced first, split by row",
              "roc_auc": round(record.number(PREFIX + "leaked.test_roc_auc"), 6),
              "accuracy": round(
                  record.number(PREFIX + "leaked.test_accuracy"), 6),
              "patients_on_both_sides": int(
                  record.number(PREFIX + "leaked.patients_on_both_sides"))}],
            "m01_leakage_comparison")

    record.save()
    print("model 1: {} training selected; test ROC-AUC {:.4f}, average "
          "precision {:.4f}; the coursework ordering reports {:.4f}".format(
              regime,
              record.number(PREFIX + "network.test_roc_auc"),
              record.number(PREFIX + "network.test_average_precision"),
              record.number(PREFIX + "leaked.test_roc_auc")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
