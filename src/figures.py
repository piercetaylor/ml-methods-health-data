"""Every figure, each drawn from a table in ``results/``.

A figure and the prose beside it cannot disagree, because both read the same
recorded table. No figure recomputes anything.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot  # noqa: E402

from . import config, utils  # noqa: E402

STYLE = {
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
}
INK = "#25333f"
ACCENT = "#b1442e"
MUTED = "#8ea3b0"


def _save(figure, name: str) -> str:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / (name + ".png")
    figure.savefig(path)
    pyplot.close(figure)
    return path.name


def fig01_leakage(_record) -> str:
    """Test ROC-AUC under the two partition orderings."""
    rows = utils.read_table("m01_leakage_comparison")
    figure, axes = pyplot.subplots(figsize=(5.6, 3.0))
    labels = ["patients held apart\nbalanced inside folds",
              "balanced first\nsplit by row"]
    values = [float(row["roc_auc"]) for row in rows]
    bars = axes.bar(labels, values, color=[INK, ACCENT], width=0.55)
    for bar, row in zip(bars, rows):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                  "{:.4f}".format(float(row["roc_auc"])), ha="center")
        axes.text(bar.get_x() + bar.get_width() / 2, 0.04,
                  "{} patients on\nboth sides".format(
                      row["patients_on_both_sides"]),
                  ha="center", color="white", fontsize=8)
    axes.axhline(0.5, color=MUTED, linestyle=":", linewidth=1)
    axes.set_ylim(0, 1.05)
    axes.set_ylabel("test ROC-AUC")
    axes.set_title("The same network under two partition orderings")
    return _save(figure, "fig01_readmission_leakage")


def fig02_curves(record) -> str:
    """The ROC and precision-recall curves of the balanced network."""
    roc = utils.read_table("m01_roc_curve")
    curve = utils.read_table("m01_pr_curve")
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(7.2, 3.4))
    left.plot([float(row["false_positive_rate"]) for row in roc],
              [float(row["true_positive_rate"]) for row in roc],
              color=INK, linewidth=1.6)
    left.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=1)
    left.set_xlabel("false positive rate")
    left.set_ylabel("true positive rate")
    left.set_title("ROC, area {:.4f}".format(
        record.number("m01.network.test_roc_auc")))

    prevalence = record.number("m01.network.test_positive_rate")
    right.plot([float(row["recall"]) for row in curve],
               [float(row["precision"]) for row in curve],
               color=INK, linewidth=1.6)
    right.axhline(prevalence, color=MUTED, linestyle=":", linewidth=1)
    right.text(0.55, prevalence + 0.012,
               "readmission rate {:.3f}".format(prevalence), color=MUTED,
               fontsize=8)
    right.set_xlabel("recall")
    right.set_ylabel("precision")
    right.set_title("Precision and recall, average {:.4f}".format(
        record.number("m01.network.test_average_precision")))
    return _save(figure, "fig02_readmission_curves")


def fig03_importance(record) -> str:
    """The fifteen features whose permutation lowers the network ROC-AUC most."""
    rows = utils.read_table("m01_permutation_importance")[:15][::-1]
    figure, axes = pyplot.subplots(figsize=(6.0, 4.2))
    axes.barh([row["feature"] for row in rows],
              [float(row["mean_decrease"]) for row in rows],
              xerr=[float(row["sd"]) for row in rows],
              color=INK, error_kw={"ecolor": MUTED, "elinewidth": 1})
    axes.set_xlabel("mean decrease in ROC-AUC over {} permutations".format(
        config.PERMUTATION_REPEATS))
    axes.set_title("What the balanced network uses, measured on {} "
                   "held-out encounters".format(
                       int(record.number("m01.importance_rows"))))
    return _save(figure, "fig03_readmission_importance")


def fig04_variance(record) -> str:
    """Each assay's share of the total unstandardized variance."""
    rows = utils.read_table("m02_feature_variance")
    figure, axes = pyplot.subplots(figsize=(5.6, 3.0))
    shares = [float(row["share_of_total"]) for row in rows]
    axes.bar([row["assay"] for row in rows], shares,
             color=[ACCENT if share > 0.1 else INK for share in shares])
    axes.set_ylabel("share of total variance")
    axes.set_title("Two assays carry {:.0%} of the unstandardized variance"
                   .format(record.number("m02.variance_share_top_two")))
    return _save(figure, "fig04_hcv_variance")


def fig05_sweep(_record) -> str:
    """Internal index and label agreement against the number of clusters."""
    rows = utils.read_table("m02_cluster_sweep")
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(7.2, 3.2))
    for method, color in (("k-means", INK), ("agglomerative", ACCENT)):
        block = [row for row in rows if row["method"] == method]
        ks = [int(row["k"]) for row in block]
        left.plot(ks, [float(row["calinski_harabasz"]) for row in block],
                  marker="o", color=color, label=method)
        right.plot(ks, [float(row["adjusted_rand"]) for row in block],
                   marker="o", color=color, label=method)
    left.set_xlabel("clusters")
    left.set_ylabel("Calinski-Harabasz")
    left.set_title("Internal index")
    right.set_xlabel("clusters")
    right.set_ylabel("adjusted Rand against the labels")
    right.set_title("Agreement with the withheld labels")
    right.axvline(config.HCV_K, color=MUTED, linestyle=":", linewidth=1)
    left.legend(frameon=False)
    return _save(figure, "fig05_hcv_cluster_sweep")


def fig06_projection(record) -> str:
    """The subjects in two principal components, by cluster and by category."""
    rows = utils.read_table("m02_projection")
    x = [float(row["pc1"]) for row in rows]
    y = [float(row["pc2"]) for row in rows]
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(7.4, 3.4),
                                            sharex=True, sharey=True)
    palette = ["#25333f", "#b1442e", "#3f7f7a", "#c58a2e", "#6a5b8c"]
    clusters = sorted({int(row["cluster_k4"]) for row in rows})
    for cluster in clusters:
        mask = [index for index, row in enumerate(rows)
                if int(row["cluster_k4"]) == cluster]
        left.scatter([x[index] for index in mask], [y[index] for index in mask],
                     s=8, alpha=0.7, color=palette[cluster % len(palette)],
                     label="cluster {}".format(cluster))
    for position, category in enumerate(
            sorted({row["category"] for row in rows})):
        mask = [index for index, row in enumerate(rows)
                if row["category"] == category]
        right.scatter([x[index] for index in mask], [y[index] for index in mask],
                      s=8, alpha=0.7, color=palette[position % len(palette)],
                      label=category)
    left.set_title("k-means at {} clusters".format(config.HCV_K))
    right.set_title("The withheld diagnostic category")
    for axes in (left, right):
        axes.set_xlabel("first component")
        axes.legend(frameon=False, fontsize=7, markerscale=1.4)
    left.set_ylabel("second component")
    figure.suptitle("Two components hold {:.0%} of the standardized variance"
                    .format(record.number("m02.pca_variance_explained")),
                    fontsize=9)
    return _save(figure, "fig06_hcv_projection")


def fig07_profile(_record) -> str:
    """Quartiles of each assay in the two-cluster solution and in the labels."""
    rows = utils.read_table("m02_group_profile")
    assays = list(config.HCV_FEATURES)
    figure, axeses = pyplot.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    for axes, grouping in zip(axeses, ("cluster", "category")):
        block = [row for row in rows if row["grouping"] == grouping]
        groups = sorted({row["group"] for row in block})
        width = 0.8 / len(groups)
        for position, group in enumerate(groups):
            values = {row["assay"]: row for row in block
                      if row["group"] == group}
            centers = [index + position * width - 0.4 + width / 2
                       for index in range(len(assays))]
            medians = [float(values[assay]["median"]) for assay in assays]
            lows = [float(values[assay]["median"])
                    - float(values[assay]["q1"]) for assay in assays]
            highs = [float(values[assay]["q3"])
                     - float(values[assay]["median"]) for assay in assays]
            axes.errorbar(centers, medians, yerr=[lows, highs], fmt="o",
                          markersize=4, capsize=2, linewidth=1,
                          color=INK if position == 0 else ACCENT,
                          label="{} (n={})".format(
                              group, values[assays[0]]["n"]))
        axes.set_xticks(range(len(assays)))
        axes.set_xticklabels(assays, rotation=45, ha="right")
        axes.set_yscale("log")
        axes.set_title("by two-cluster k-means" if grouping == "cluster"
                       else "by cirrhosis against every other category")
        axes.legend(frameon=False, fontsize=7)
    axeses[0].set_ylabel("assay value, median and quartiles")
    return _save(figure, "fig07_hcv_group_profile")


def fig08_support(record) -> str:
    """Frequent itemsets naming each diagnosis, against the support threshold."""
    rows = utils.read_table("m03_support_sweep")
    supports = [float(row["min_support"]) for row in rows]
    figure, axes = pyplot.subplots(figsize=(5.6, 3.0))
    axes.plot(supports, [int(row["itemsets_with_benign"]) for row in rows],
              marker="o", color=INK, label="benign")
    axes.plot(supports, [int(row["itemsets_with_malignant"]) for row in rows],
              marker="o", color=ACCENT, label="malignant")
    axes.axvline(record.number("m03.malignant_rate"), color=MUTED,
                 linestyle=":", linewidth=1)
    axes.text(record.number("m03.malignant_rate") + 0.005, 5,
              "malignant rate {:.3f}".format(record.number("m03.malignant_rate")),
              color=MUTED, fontsize=8)
    axes.set_yscale("symlog")
    axes.set_xlabel("minimum support")
    axes.set_ylabel("frequent itemsets naming the class")
    axes.set_title("No itemset can be more frequent than its rarest item")
    axes.legend(frameon=False)
    return _save(figure, "fig08_rules_support_sweep")


def fig09_items(record) -> str:
    """The items appearing in the antecedents of the diagnosis rules."""
    rows = utils.read_table("m03_selected_items")[:19][::-1]
    figure, axes = pyplot.subplots(figsize=(6.0, 4.4))
    axes.barh([row["item"] for row in rows],
              [int(row["rules"]) for row in rows], color=INK)
    axes.set_xlabel("rules whose antecedent names the item")
    axes.set_title("{} items appear in the {} diagnosis rules".format(
        int(record.number("m03.selected_items")),
        int(record.number("m03.diagnosis_rules"))))
    return _save(figure, "fig09_rules_selected_items")


def fig10_predictions(record) -> str:
    """Observed against predicted daily intake on the held-out subjects."""
    rows = utils.read_table("m04_predictions")
    observed = [float(row["observed"]) for row in rows]
    predicted = [float(row["predicted"]) for row in rows]
    figure, axes = pyplot.subplots(figsize=(4.4, 4.0))
    axes.scatter(predicted, observed, s=18, alpha=0.75, color=INK)
    limit = max(max(observed), max(predicted)) * 1.05
    axes.plot([0, limit], [0, limit], color=MUTED, linestyle=":", linewidth=1)
    axes.set_xlim(0, limit)
    axes.set_ylim(0, limit)
    axes.set_xlabel("predicted half-pints per day")
    axes.set_ylabel("observed half-pints per day")
    axes.set_title("Least squares on the held-out subjects, R2 = {:.3f}".format(
        record.number("m04.ols.test_r2")))
    return _save(figure, "fig10_bupa_observed_predicted")


def fig11_coefficients(_record) -> str:
    """The least squares coefficients with their confidence intervals."""
    rows = [row for row in utils.read_table("m04_coefficients")
            if row["term"] != "intercept"][::-1]
    figure, axes = pyplot.subplots(figsize=(5.2, 3.0))
    centers = range(len(rows))
    estimates = [float(row["estimate"]) for row in rows]
    axes.errorbar(
        estimates, list(centers),
        xerr=[[estimate - float(row["ci_low"])
               for estimate, row in zip(estimates, rows)],
              [float(row["ci_high"]) - estimate
               for estimate, row in zip(estimates, rows)]],
        fmt="o", color=INK, ecolor=MUTED, capsize=3, linewidth=1)
    axes.axvline(0, color=ACCENT, linestyle=":", linewidth=1)
    axes.set_yticks(list(centers))
    axes.set_yticklabels([row["term"] for row in rows])
    axes.set_xlabel("half-pints per day per unit of the assay")
    axes.set_title("Coefficients with 95 percent intervals")
    return _save(figure, "fig11_bupa_coefficients")


ALL = (fig01_leakage, fig02_curves, fig03_importance, fig04_variance,
       fig05_sweep, fig06_projection, fig07_profile, fig08_support,
       fig09_items, fig10_predictions, fig11_coefficients)


def draw_all() -> list[str]:
    record = utils.Metrics()
    with matplotlib.rc_context(STYLE):
        return [function(record) for function in ALL]
