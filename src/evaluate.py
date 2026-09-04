"""Evaluation shared by the four analyses.

Each function takes what a model produced and returns a dictionary of named
quantities, which the analysis records under its own prefix. Writing the
evaluation once means the classification and the regression results are
comparable to themselves across runs, and that a metric is defined in one
place.

The three task families do not share one interface, and no attempt is made to
give them one. A clustering has no held-out accuracy to report and a regression
has no confusion matrix, so :func:`clustering` and :func:`regression` return
different keys from :func:`classification` and are called separately.
"""

from __future__ import annotations

import numpy
from sklearn import metrics


def classification(y_true, y_score, threshold: float) -> dict:
    """Discrimination, calibration and the confusion matrix at one threshold.

    ROC-AUC and average precision are computed from the scores and do not
    depend on the threshold. Average precision is reported beside ROC-AUC
    because it is sensitive to the positive rate, and on an outcome this rare
    ROC-AUC alone reads better than the model performs. Accuracy is reported
    with the majority-class rate beside it, so the two can be compared.
    """
    y_true = numpy.asarray(y_true)
    y_score = numpy.asarray(y_score)
    predicted = (y_score >= threshold).astype(int)
    matrix = metrics.confusion_matrix(y_true, predicted, labels=[0, 1])
    (true_negative, false_positive), (false_negative, true_positive) = matrix
    prevalence = float(y_true.mean())
    return {
        "roc_auc": float(metrics.roc_auc_score(y_true, y_score)),
        "average_precision": float(
            metrics.average_precision_score(y_true, y_score)),
        "brier": float(metrics.brier_score_loss(y_true, y_score)),
        "threshold": float(threshold),
        "accuracy": float(metrics.accuracy_score(y_true, predicted)),
        "majority_class_rate": float(max(prevalence, 1 - prevalence)),
        "balanced_accuracy": float(
            metrics.balanced_accuracy_score(y_true, predicted)),
        "precision": float(metrics.precision_score(y_true, predicted,
                                                   zero_division=0)),
        "recall": float(metrics.recall_score(y_true, predicted,
                                             zero_division=0)),
        "f1": float(metrics.f1_score(y_true, predicted, zero_division=0)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "positive_rate": prevalence,
        "rows": int(y_true.size),
    }


def best_threshold(y_true, y_score) -> float:
    """The decision threshold maximizing Youden's J on the data given.

    This is chosen on the training partition and applied unchanged to the test
    partition. Choosing it on the test partition would tune a parameter on the
    data used to report the result.
    """
    false_positive, true_positive, thresholds = metrics.roc_curve(y_true, y_score)
    return float(thresholds[numpy.argmax(true_positive - false_positive)])


def regression(y_true, y_predicted) -> dict:
    """Fit and error for a continuous target.

    The coefficient of determination is reported against the mean predictor,
    so a value at or below zero states that the model does not beat predicting
    the training mean. Root mean squared error and mean absolute error are in
    the units of the target.
    """
    y_true = numpy.asarray(y_true, dtype=float)
    y_predicted = numpy.asarray(y_predicted, dtype=float)
    residual = y_true - y_predicted
    return {
        "r2": float(metrics.r2_score(y_true, y_predicted)),
        "rmse": float(numpy.sqrt(numpy.mean(residual ** 2))),
        "mae": float(numpy.mean(numpy.abs(residual))),
        "mean_observed": float(y_true.mean()),
        "sd_observed": float(y_true.std(ddof=1)),
        "rows": int(y_true.size),
    }


def clustering(features, labels) -> dict:
    """Internal validity of one partition of the feature space.

    All three indices are computed on the points that were assigned to a
    cluster. DBSCAN labels an outlier -1, and an index computed with the
    outliers treated as a cluster of their own describes a group that the
    method declined to form. The number left out is returned, because an index
    computed on 60 percent of the data is not comparable to one computed on all
    of it.

    Calinski-Harabasz is the index the coursework used. It is a ratio of
    between-cluster to within-cluster dispersion, which is the quantity k-means
    minimizes, so it favors k-means over a method optimizing anything else.
    Silhouette and Davies-Bouldin are reported beside it for that reason.
    """
    features = numpy.asarray(features, dtype=float)
    labels = numpy.asarray(labels)
    assigned = labels >= 0
    kept = features[assigned]
    kept_labels = labels[assigned]
    distinct, sizes = numpy.unique(kept_labels, return_counts=True)
    result = {
        "clusters": int(distinct.size),
        "unassigned": int((~assigned).sum()),
        "assigned_fraction": float(assigned.mean()),
        "smallest_cluster": int(sizes.min()) if distinct.size else 0,
    }
    if distinct.size < 2:
        # Every index below is undefined on one cluster. Recording nothing
        # would let a single-cluster result pass as an absent measurement.
        result.update({"calinski_harabasz": float("nan"),
                       "silhouette": float("nan"),
                       "davies_bouldin": float("nan")})
        return result
    result.update({
        "calinski_harabasz": float(
            metrics.calinski_harabasz_score(kept, kept_labels)),
        "silhouette": float(metrics.silhouette_score(kept, kept_labels)),
        "davies_bouldin": float(
            metrics.davies_bouldin_score(kept, kept_labels)),
    })
    return result


def agreement(reference, labels) -> dict:
    """How far one clustering recovers a labeling it never saw.

    A cluster index and a class index are arbitrary names, so they cannot be
    compared as values. All four measures here are invariant to renaming.
    Adjusted Rand and adjusted mutual information are corrected for chance and
    take the value zero on a random assignment. Homogeneity asks whether a
    cluster holds one class and completeness whether a class sits in one
    cluster.
    """
    return {
        "adjusted_rand": float(metrics.adjusted_rand_score(reference, labels)),
        "adjusted_mutual_information": float(
            metrics.adjusted_mutual_info_score(reference, labels)),
        "homogeneity": float(metrics.homogeneity_score(reference, labels)),
        "completeness": float(metrics.completeness_score(reference, labels)),
    }
