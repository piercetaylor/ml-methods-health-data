"""Every figure, each drawn from a table in ``results/``.

A figure and the prose beside it cannot disagree, because both read the same
recorded table. No figure recomputes anything.

The figures follow the conventions of a journal article. Matplotlib's default
color cycle and typeface are used unchanged. Panels carry no titles; what each
shows is stated in the caption beside it in ``docs/``. Multi-panel figures label
their panels (a), (b) in the top left. Widths are one column at 3.4 inches or
two at 7.0, and every file is written at 300 dots per inch.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as pyplot  # noqa: E402

from . import config, utils  # noqa: E402

STYLE = {
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "errorbar.capsize": 2,
    "figure.constrained_layout.use": True,
}
SINGLE = 3.4
DOUBLE = 7.0
GRAY = "0.55"


def _save(figure, name: str) -> str:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / (name + ".png")
    figure.savefig(path)
    pyplot.close(figure)
    return path.name


def _label(axes, letter: str) -> None:
    """A bold panel letter above the top left corner of the axes."""
    axes.set_title("({})".format(letter), loc="left", fontweight="bold",
                   fontsize=9, pad=6)


def fig01_leakage(_record) -> str:
    """Test ROC-AUC under the two partition orderings."""
    rows = utils.read_table("m01_leakage_comparison")
    figure, axes = pyplot.subplots(figsize=(SINGLE, 2.6))
    labels = ["patients held apart," + "\n" + "balanced training",
              "balanced first," + "\n" + "split by row"]
    values = [float(row["roc_auc"]) for row in rows]
    bars = axes.bar(labels, values, width=0.55, color=["C0", "C3"])
    for bar, row in zip(bars, rows):
        axes.annotate("{:.3f}".format(float(row["roc_auc"])),
                      (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      xytext=(0, 3), textcoords="offset points", ha="center",
                      fontsize=7)
        axes.annotate("{} patients\non both sides".format(
                          row["patients_on_both_sides"]),
                      (bar.get_x() + bar.get_width() / 2, 0.06),
                      ha="center", va="bottom", color="white", fontsize=6.5)
    axes.axhline(0.5, color=GRAY, linestyle=":", linewidth=0.8)
    axes.set_ylim(0, 0.85)
    axes.set_ylabel("test ROC-AUC")
    return _save(figure, "fig01_readmission_leakage")


def fig02_curves(record) -> str:
    """The ROC and precision-recall curves of the selected network."""
    roc = utils.read_table("m01_roc_curve")
    curve = utils.read_table("m01_pr_curve")
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(DOUBLE, 2.8))
    left.plot([float(row["false_positive_rate"]) for row in roc],
              [float(row["true_positive_rate"]) for row in roc],
              color="C0", label="network, AUC = {:.3f}".format(
                  record.number("m01.network.test_roc_auc")))
    left.plot([0, 1], [0, 1], color=GRAY, linestyle=":", linewidth=0.8,
              label="chance")
    left.set_xlabel("false positive rate")
    left.set_ylabel("true positive rate")
    left.set_xlim(0, 1)
    left.set_ylim(0, 1)
    left.set_aspect("equal")
    left.legend(loc="lower right")
    _label(left, "a")

    prevalence = record.number("m01.network.test_positive_rate")
    right.plot([float(row["recall"]) for row in curve],
               [float(row["precision"]) for row in curve],
               color="C0", label="network, AP = {:.3f}".format(
                   record.number("m01.network.test_average_precision")))
    right.axhline(prevalence, color=GRAY, linestyle=":", linewidth=0.8,
                  label="readmission rate {:.3f}".format(prevalence))
    right.set_xlabel("recall")
    right.set_ylabel("precision")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.set_aspect("equal")
    right.legend(loc="upper right")
    _label(right, "b")
    return _save(figure, "fig02_readmission_curves")


def fig03_importance(_record) -> str:
    """The fifteen features whose permutation lowers the network ROC-AUC most."""
    rows = utils.read_table("m01_permutation_importance")[:15][::-1]
    figure, axes = pyplot.subplots(figsize=(SINGLE, 3.4))
    axes.barh([row["feature"] for row in rows],
              [float(row["mean_decrease"]) for row in rows],
              xerr=[float(row["sd"]) for row in rows],
              color="C0", error_kw={"ecolor": "0.3", "elinewidth": 0.8})
    axes.axvline(0, color="0.3", linewidth=0.7)
    axes.set_xlabel("mean decrease in test ROC-AUC, {} permutations".format(
        config.PERMUTATION_REPEATS))
    return _save(figure, "fig03_readmission_importance")


def fig04_variance(_record) -> str:
    """Each assay's share of the total unstandardized variance."""
    rows = utils.read_table("m02_feature_variance")
    figure, axes = pyplot.subplots(figsize=(SINGLE, 2.4))
    shares = [float(row["share_of_total"]) for row in rows]
    axes.bar([row["assay"] for row in rows], shares, color="C0", width=0.7)
    axes.set_ylabel("share of total variance")
    axes.set_ylim(0, max(shares) * 1.1)
    for position, share in enumerate(shares[:2]):
        axes.annotate("{:.2f}".format(share), (position, share),
                      xytext=(0, 2), textcoords="offset points",
                      ha="center", fontsize=7)
    return _save(figure, "fig04_hcv_variance")


def fig05_sweep(_record) -> str:
    """Internal index and label agreement against the number of clusters."""
    rows = utils.read_table("m02_cluster_sweep")
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(DOUBLE, 2.6))
    for method, color, marker in (("k-means", "C0", "o"),
                                  ("agglomerative", "C1", "s")):
        block = [row for row in rows if row["method"] == method]
        ks = [int(row["k"]) for row in block]
        left.plot(ks, [float(row["calinski_harabasz"]) for row in block],
                  marker=marker, color=color, label=method)
        right.plot(ks, [float(row["adjusted_rand"]) for row in block],
                   marker=marker, color=color, label=method)
    for axes in (left, right):
        axes.set_xlabel("number of clusters, k")
        axes.set_xticks([int(row["k"]) for row in rows
                         if row["method"] == "k-means"])
        axes.axvline(config.HCV_K, color=GRAY, linestyle=":", linewidth=0.8)
    left.set_ylabel("Calinski-Harabasz index")
    right.set_ylabel("adjusted Rand index against the labels")
    left.legend()
    _label(left, "a")
    _label(right, "b")
    return _save(figure, "fig05_hcv_cluster_sweep")


def fig06_projection(record) -> str:
    """The subjects in two principal components, by cluster and by category."""
    rows = utils.read_table("m02_projection")
    x = [float(row["pc1"]) for row in rows]
    y = [float(row["pc2"]) for row in rows]
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(DOUBLE, 3.0),
                                            sharex=True, sharey=True)
    clusters = sorted({int(row["cluster_k4"]) for row in rows})
    for cluster in clusters:
        mask = [index for index, row in enumerate(rows)
                if int(row["cluster_k4"]) == cluster]
        left.scatter([x[index] for index in mask], [y[index] for index in mask],
                     s=7, alpha=0.7, color="C{}".format(cluster),
                     linewidths=0, label="cluster {}".format(cluster))
    order = ("Blood Donor", "Hepatitis", "Fibrosis", "Cirrhosis")
    for position, category in enumerate(order):
        mask = [index for index, row in enumerate(rows)
                if row["category"].strip() == category]
        right.scatter([x[index] for index in mask], [y[index] for index in mask],
                      s=7, alpha=0.7, color="C{}".format(position),
                      linewidths=0, label=category.lower())
    share = record.number("m02.pca_variance_explained")
    for axes in (left, right):
        axes.set_xlabel("first principal component")
        axes.legend(markerscale=1.6, handletextpad=0.3)
    left.set_ylabel("second principal component")
    left.annotate("{:.0%} of the standardized variance".format(share),
                  (0.02, 0.02), xycoords="axes fraction", fontsize=6.5,
                  color="0.3")
    _label(left, "a")
    _label(right, "b")
    return _save(figure, "fig06_hcv_projection")


def fig07_profile(_record) -> str:
    """Quartiles of each assay in the two-cluster solution and in the labels."""
    rows = utils.read_table("m02_group_profile")
    assays = list(config.HCV_FEATURES)
    figure, axeses = pyplot.subplots(1, 2, figsize=(DOUBLE, 2.8), sharey=True)
    for axes, grouping, letter in zip(axeses, ("cluster", "category"), "ab"):
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
            name = ("cluster {}".format(group) if grouping == "cluster"
                    else group)
            axes.errorbar(centers, medians, yerr=[lows, highs], fmt="o",
                          markersize=3, linewidth=0.9,
                          color="C{}".format(position),
                          label="{}, n = {}".format(
                              name, values[assays[0]]["n"]))
        axes.set_xticks(range(len(assays)))
        axes.set_xticklabels(assays, rotation=45, ha="right")
        axes.set_yscale("log")
        axes.set_ylim(0.3, 400)
        axes.legend(loc="lower center", ncol=2, columnspacing=1.0)
        _label(axes, letter)
    axeses[0].set_ylabel("assay value, median and quartiles")
    return _save(figure, "fig07_hcv_group_profile")


def fig08_support(record) -> str:
    """Frequent itemsets naming each diagnosis, against the support threshold."""
    rows = utils.read_table("m03_support_sweep")
    supports = [float(row["min_support"]) for row in rows]
    figure, axes = pyplot.subplots(figsize=(SINGLE, 2.6))
    axes.plot(supports, [int(row["itemsets_with_benign"]) for row in rows],
              marker="o", color="C0", label="benign")
    axes.plot(supports, [int(row["itemsets_with_malignant"]) for row in rows],
              marker="s", color="C3", label="malignant")
    rate = record.number("m03.malignant_rate")
    axes.axvline(rate, color=GRAY, linestyle=":", linewidth=0.8)
    axes.annotate("malignant rate\n{:.3f}".format(rate), (rate, 2000),
                  xytext=(4, 0), textcoords="offset points", fontsize=6.5,
                  color="0.3")
    axes.set_yscale("symlog")
    axes.set_xlabel("minimum support")
    axes.set_ylabel("frequent itemsets naming the class")
    axes.set_xticks(supports)
    axes.legend(loc="upper right")
    return _save(figure, "fig08_rules_support_sweep")


def fig09_items(_record) -> str:
    """The items appearing in the antecedents of the diagnosis rules."""
    rows = utils.read_table("m03_selected_items")[:19][::-1]
    figure, axes = pyplot.subplots(figsize=(SINGLE, 3.8))
    axes.barh([row["item"].replace("_0", "") for row in rows],
              [int(row["rules"]) for row in rows], color="C0")
    axes.set_xlabel("diagnosis rules whose antecedent holds the item")
    return _save(figure, "fig09_rules_selected_items")


def fig10_predictions(record) -> str:
    """Observed against predicted intake, and the spread of R² over splits."""
    rows = utils.read_table("m04_predictions")
    observed = [float(row["observed"]) for row in rows]
    predicted = [float(row["predicted"]) for row in rows]
    repeats = utils.read_table("m04_repeated_splits")
    figure, (left, right) = pyplot.subplots(
        1, 2, figsize=(DOUBLE, 2.9), gridspec_kw={"width_ratios": [1, 1.1]})
    left.scatter(predicted, observed, s=14, alpha=0.75, color="C0",
                 linewidths=0)
    limit = max(max(observed), max(predicted)) * 1.05
    left.plot([0, limit], [0, limit], color=GRAY, linestyle=":", linewidth=0.8)
    left.set_xlim(0, limit)
    left.set_ylim(0, limit)
    left.set_aspect("equal")
    left.set_xlabel("predicted half-pints per day")
    left.set_ylabel("observed half-pints per day")
    left.annotate("R² = {:.3f}, n = {}".format(
                      record.number("m04.ols.test_r2"), len(rows)),
                  (0.96, 0.05), xycoords="axes fraction", fontsize=7,
                  ha="right")
    _label(left, "a")

    values = [float(row["r2"]) for row in repeats]
    right.hist(values, bins=10, color="C0", alpha=0.85, edgecolor="white",
               linewidth=0.5)
    right.axvline(values[0], color="C3", linewidth=1.0,
                  label="primary split, {:.3f}".format(values[0]))
    right.axvline(record.number("m04.repeats.r2_mean"), color="0.3",
                  linestyle="--", linewidth=0.9,
                  label="mean of {} splits, {:.3f}".format(
                      len(values), record.number("m04.repeats.r2_mean")))
    right.set_xlabel("test R² of least squares")
    right.set_ylabel("splits")
    right.legend(loc="upper left")
    _label(right, "b")
    return _save(figure, "fig10_bupa_observed_predicted")


def fig11_coefficients(_record) -> str:
    """The least squares coefficients with their confidence intervals."""
    rows = [row for row in utils.read_table("m04_coefficients")
            if row["term"] != "intercept"][::-1]
    figure, axes = pyplot.subplots(figsize=(SINGLE, 2.4))
    centers = list(range(len(rows)))
    estimates = [float(row["estimate"]) for row in rows]
    axes.errorbar(
        estimates, centers,
        xerr=[[estimate - float(row["ci_low"])
               for estimate, row in zip(estimates, rows)],
              [float(row["ci_high"]) - estimate
               for estimate, row in zip(estimates, rows)]],
        fmt="o", color="C0", ecolor="0.3", linewidth=0.9, markersize=4)
    axes.axvline(0, color=GRAY, linestyle=":", linewidth=0.8)
    axes.set_yticks(centers)
    axes.set_yticklabels([row["term"] for row in rows])
    axes.set_xlabel("half-pints per day per released unit, 95% interval")
    return _save(figure, "fig11_bupa_coefficients")


def fig12_repeats(record) -> str:
    """ROC-AUC and average precision of the two models over repeated splits."""
    rows = utils.read_table("m01_repeated_splits")
    seeds = sorted({int(row["seed"]) for row in rows})
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(DOUBLE, 2.6))
    for axes, measure, letter in ((left, "roc_auc", "a"),
                                  (right, "average_precision", "b")):
        for model, color, marker in (("network", "C0", "o"),
                                     ("logistic", "C1", "s")):
            values = [float(next(row[measure] for row in rows
                                 if int(row["seed"]) == seed
                                 and row["model"] == model))
                      for seed in seeds]
            axes.plot(range(len(seeds)), values, marker=marker, color=color,
                      label="{}, mean {:.3f}".format(
                          "network" if model == "network"
                          else "logistic regression",
                          record.number("m01.repeats.{}.{}_mean".format(
                              model, measure))))
        axes.set_xticks(range(len(seeds)))
        axes.set_xticklabels(["draw {}".format(index + 1)
                              for index in range(len(seeds))])
        axes.legend(loc="best")
        _label(axes, letter)
    left.set_ylabel("test ROC-AUC")
    right.set_ylabel("test average precision")
    right.axhline(record.number("m01.network.test_positive_rate"), color=GRAY,
                  linestyle=":", linewidth=0.8)
    return _save(figure, "fig12_readmission_repeated_splits")


ALL = (fig01_leakage, fig02_curves, fig03_importance, fig04_variance,
       fig05_sweep, fig06_projection, fig07_profile, fig08_support,
       fig09_items, fig10_predictions, fig11_coefficients, fig12_repeats)


def draw_all() -> list[str]:
    record = utils.Metrics()
    with matplotlib.rc_context(STYLE):
        return [function(record) for function in ALL]
