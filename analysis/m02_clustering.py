#!/usr/bin/env python
"""Model 2. Recovering hepatitis C diagnostic categories without labels.

    python analysis/m02_clustering.py

The HCV data (Lichtinghagen et al. 2020) holds ten laboratory values for 615
subjects, each labeled as a blood donor or as carrying hepatitis, fibrosis or
cirrhosis. The labels are withheld from the clustering and used afterwards to
measure how much of the diagnostic structure the unlabeled values recover.
Every subject is analyzed; the few missing assay values are imputed without
reference to the label, and the complete cases are scored alone as a check.

Three methods are compared at the four clusters the label set implies: k-means,
DBSCAN and agglomerative clustering with complete linkage. Internal validity is
reported by three indices, and agreement with the withheld labels by four
measures that are invariant to how clusters and classes are numbered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, evaluate, utils  # noqa: E402

PREFIX = "m02."


def fit_kmeans(features, k: int, seed: int = config.SEED):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(features)


def search_dbscan(features, reference) -> tuple[dict, list[dict]]:
    """Search the DBSCAN neighborhood parameters and return the best and the grid.

    A DBSCAN run that assigns few points scores well on an internal index while
    describing little of the data, so a configuration is admitted only when it
    assigns at least four fifths of the subjects and forms at least two
    clusters. The best is then the admitted configuration with the highest
    Calinski-Harabasz score, which is the index the coursework used.
    """
    rows = []
    for eps in config.DBSCAN_EPS_GRID:
        for minimum in config.DBSCAN_MIN_SAMPLES:
            labels = DBSCAN(eps=eps, min_samples=minimum).fit_predict(features)
            summary = evaluate.clustering(features, labels)
            if summary["clusters"] < 2:
                continue
            row = {"eps": eps, "min_samples": minimum}
            row.update({key: round(value, 6) if isinstance(value, float)
                        else value for key, value in summary.items()})
            row.update({key: round(value, 6) for key, value
                        in evaluate.agreement(reference, labels).items()})
            rows.append(row)
    admitted = [row for row in rows if row["assigned_fraction"] >= 0.8]
    if not admitted:
        raise RuntimeError("no DBSCAN configuration assigned 80 percent of the "
                           "subjects to at least two clusters")
    best = max(admitted, key=lambda row: row["calinski_harabasz"])
    return best, sorted(rows, key=lambda row: -row["calinski_harabasz"])


def main() -> int:
    record = utils.Metrics()
    with utils.Timer(record, "m02"):
        frame, counts = data.load_hcv()
        record.update(counts, PREFIX)

        features = list(config.HCV_FEATURES)
        raw = frame[features].to_numpy(dtype=float)
        scaled = StandardScaler().fit_transform(raw)
        reference = frame["category_index"].to_numpy()

        # --- standardization, and why it is not chosen on the result --------
        # The ten assays are reported in different units and their variances
        # differ by four orders of magnitude, so an unstandardized Euclidean
        # distance is very nearly the distance in GGT and creatinine alone.
        # Both clusterings are recorded. The standardized one is used for
        # everything that follows, and that choice is made before the labels
        # are consulted: picking the preprocessing by how well the clusters
        # agree with the withheld labels would put those labels back into an
        # analysis whose whole claim is that it never saw them.
        variances = dict(zip(features, raw.var(axis=0, ddof=1)))
        utils.write_table(
            [{"assay": name, "variance": round(float(value), 4),
              "share_of_total": round(float(value) / sum(variances.values()), 6)}
             for name, value in sorted(variances.items(),
                                       key=lambda item: -item[1])],
            "m02_feature_variance")
        record.set(PREFIX + "variance_share_top_two",
                   sum(sorted(variances.values())[-2:]) / sum(variances.values()))

        for label, matrix in (("unscaled", raw), ("scaled", scaled)):
            model = fit_kmeans(matrix, config.HCV_K)
            record.update(evaluate.clustering(matrix, model.labels_),
                          PREFIX + "kmeans_" + label + ".")
            record.update(evaluate.agreement(reference, model.labels_),
                          PREFIX + "kmeans_" + label + ".")

        # --- the number of clusters -----------------------------------------
        sweep = []
        for k in config.HCV_CLUSTER_RANGE:
            for name, labels in (
                    ("k-means", fit_kmeans(scaled, k).labels_),
                    ("agglomerative", AgglomerativeClustering(
                        n_clusters=k, linkage="complete").fit_predict(scaled))):
                row = {"method": name, "k": k}
                row.update({key: round(value, 6) if isinstance(value, float)
                            else value
                            for key, value in evaluate.clustering(
                                scaled, labels).items()})
                row.update({key: round(value, 6) for key, value
                            in evaluate.agreement(reference, labels).items()})
                sweep.append(row)
        utils.write_table(sweep, "m02_cluster_sweep")

        # --- the three methods at four clusters ------------------------------
        best_dbscan, dbscan_grid = search_dbscan(scaled, reference)
        utils.write_table(dbscan_grid[:20], "m02_dbscan_grid")
        record.set(PREFIX + "dbscan.eps", best_dbscan["eps"])
        record.set(PREFIX + "dbscan.min_samples", best_dbscan["min_samples"])
        record.set(PREFIX + "dbscan.configurations_searched", len(dbscan_grid))

        kmeans = fit_kmeans(scaled, config.HCV_K)
        agglomerative = AgglomerativeClustering(
            n_clusters=config.HCV_K, linkage="complete").fit(scaled)
        dbscan = DBSCAN(eps=best_dbscan["eps"],
                        min_samples=best_dbscan["min_samples"]).fit(scaled)

        comparison = []
        for name, labels in (("k-means", kmeans.labels_),
                             ("DBSCAN", dbscan.labels_),
                             ("agglomerative", agglomerative.labels_)):
            key = PREFIX + name.replace("-", "").lower() + "."
            internal = evaluate.clustering(scaled, labels)
            external = evaluate.agreement(reference, labels)
            record.update(internal, key)
            record.update(external, key)
            row = {"method": name}
            row.update({name_: round(value, 6) if isinstance(value, float)
                        else value for name_, value in internal.items()})
            row.update({name_: round(value, 6)
                        for name_, value in external.items()})
            comparison.append(row)
        utils.write_table(comparison, "m02_model_comparison")

        # --- what the clusters hold -----------------------------------------
        # A cluster index and a class index name nothing in common, so the two
        # cannot be compared as values. The contingency table is the comparison
        # that can be made, and it is what the agreement measures summarize.
        frame["cluster_k4"] = kmeans.labels_
        contingency = pandas.crosstab(frame["cluster_k4"],
                                      frame["category_label"])
        utils.write_table(
            [{"cluster": int(index), **{str(column): int(value)
                                        for column, value in row.items()}}
             for index, row in contingency.iterrows()],
            "m02_contingency_k4")

        # The coursework also compared a two-cluster solution against the
        # cirrhosis cases held apart from every other category, and that
        # comparison is kept because it is where the recovery is strongest.
        two = fit_kmeans(scaled, 2)
        frame["cluster_k2"] = two.labels_
        frame["cirrhosis"] = (frame["category_index"] == 3).astype(int)
        record.update(evaluate.agreement(frame["cirrhosis"], two.labels_),
                      PREFIX + "kmeans_k2_vs_cirrhosis.")
        record.update(evaluate.clustering(scaled, two.labels_),
                      PREFIX + "kmeans_k2.")
        utils.write_table(
            [{"cluster": int(index),
              **{("cirrhosis" if column else "other"): int(value)
                 for column, value in row.items()}}
             for index, row in pandas.crosstab(frame["cluster_k2"],
                                               frame["cirrhosis"]).iterrows()],
            "m02_contingency_k2")

        # --- the imputed subjects, held out as a sensitivity -----------------
        # The results above use every subject, with the 26 incomplete ones
        # imputed. The same two k-means solutions are refitted on the complete
        # cases alone, so the effect of keeping the imputed subjects is a
        # recorded difference and not an assumption either way.
        complete = ~frame["imputed"].to_numpy()
        scaled_complete = StandardScaler().fit_transform(raw[complete])
        reference_complete = reference[complete]
        cirrhosis_complete = frame.loc[complete, "cirrhosis"].to_numpy()
        record.set(PREFIX + "complete_case.rows", int(complete.sum()))
        sensitivity = []
        for name, labels_all, labels_complete, truth_all, truth_complete in (
                ("k-means, k=4, against four categories",
                 kmeans.labels_,
                 fit_kmeans(scaled_complete, config.HCV_K).labels_,
                 reference, reference_complete),
                ("k-means, k=2, against cirrhosis",
                 two.labels_,
                 fit_kmeans(scaled_complete, 2).labels_,
                 frame["cirrhosis"].to_numpy(), cirrhosis_complete)):
            row = {"comparison": name,
                   "rows_all": int(len(labels_all)),
                   "rows_complete": int(len(labels_complete))}
            for suffix, labels, truth in (("_all", labels_all, truth_all),
                                          ("_complete", labels_complete,
                                           truth_complete)):
                for key, value in evaluate.agreement(truth, labels).items():
                    row[key + suffix] = round(value, 6)
            sensitivity.append(row)
        utils.write_table(sensitivity, "m02_imputation_sensitivity")
        record.set(PREFIX + "complete_case.kmeans.adjusted_rand",
                   sensitivity[0]["adjusted_rand_complete"])
        record.set(PREFIX + "complete_case.kmeans_k2_vs_cirrhosis.adjusted_rand",
                   sensitivity[1]["adjusted_rand_complete"])

        # --- stability over the k-means seed --------------------------------
        # k-means is run from ten initializations at one seed. The two headline
        # solutions are refitted at ten further seeds and the agreement measure
        # is recorded at each, so the reported value is read against its spread.
        stability = []
        for offset in range(config.HCV_STABILITY_SEEDS):
            seed = config.SEED + offset
            stability.append({
                "seed": seed,
                "adjusted_rand_k4": round(evaluate.agreement(
                    reference, fit_kmeans(scaled, config.HCV_K, seed).labels_
                )["adjusted_rand"], 6),
                "adjusted_rand_k2_vs_cirrhosis": round(evaluate.agreement(
                    frame["cirrhosis"], fit_kmeans(scaled, 2, seed).labels_
                )["adjusted_rand"], 6)})
        utils.write_table(stability, "m02_seed_stability")
        for column in ("adjusted_rand_k4", "adjusted_rand_k2_vs_cirrhosis"):
            values = [row[column] for row in stability]
            record.set(PREFIX + "stability." + column + "_min", min(values))
            record.set(PREFIX + "stability." + column + "_max", max(values))
        record.set(PREFIX + "stability.seeds", len(stability))

        # --- the tables the figures are drawn from ---------------------------
        profile = []
        for grouping, column in (("cluster", "cluster_k2"),
                                 ("category", "cirrhosis")):
            for value, block in frame.groupby(column):
                for assay in features:
                    series = block[assay]
                    profile.append({
                        "grouping": grouping,
                        "group": ("cirrhosis" if value else "other")
                                 if column == "cirrhosis" else int(value),
                        "assay": assay, "n": int(series.size),
                        "mean": round(float(series.mean()), 4),
                        "sd": round(float(series.std(ddof=1)), 4),
                        "q1": round(float(series.quantile(0.25)), 4),
                        "median": round(float(series.median()), 4),
                        "q3": round(float(series.quantile(0.75)), 4)})
        utils.write_table(profile, "m02_group_profile")

        projection = PCA(n_components=2, random_state=config.SEED)
        coordinates = projection.fit_transform(scaled)
        record.set(PREFIX + "pca_variance_explained",
                   float(projection.explained_variance_ratio_.sum()))
        utils.write_table(
            [{"pc1": round(float(x), 4), "pc2": round(float(y), 4),
              "cluster_k4": int(cluster), "dbscan": int(noise),
              "category": label}
             for x, y, cluster, noise, label in zip(
                 coordinates[:, 0], coordinates[:, 1], kmeans.labels_,
                 dbscan.labels_, frame["category_label"])],
            "m02_projection")

    record.save()
    print("model 2: k-means at k={} recovers the labels with adjusted Rand "
          "{:.4f}; the two-cluster solution against cirrhosis alone gives "
          "{:.4f}".format(
              config.HCV_K, record.number(PREFIX + "kmeans.adjusted_rand"),
              record.number(PREFIX + "kmeans_k2_vs_cirrhosis.adjusted_rand")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
