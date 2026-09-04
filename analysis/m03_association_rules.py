#!/usr/bin/env python
"""Model 3. Association rules among diagnostic image features.

    python analysis/m03_association_rules.py

The Wisconsin diagnostic breast cancer measurements (Wolberg et al. 1995) hold
thirty continuous features computed from a digitized fine needle aspirate, and a
malignant or benign diagnosis for each of 569 samples. Apriori needs items, so
each feature is discretized into three levels and one-hot encoded, and the
diagnosis joins the transaction as two further items.

Association rule mining is unsupervised. Filtering the mined rules down to those
whose consequent is the diagnosis turns the output into a ranking of features by
their association with the outcome, which is a feature selection result and not
a rule mining result; both are reported here and named for what they are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import KBinsDiscretizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, utils  # noqa: E402

PREFIX = "m03."
DIAGNOSIS_ITEMS = ("diagnosis_malignant", "diagnosis_benign")


def transactions(frame: pandas.DataFrame, features: list[str]
                 ) -> tuple[pandas.DataFrame, list[dict]]:
    """Discretize every feature into three levels and one-hot encode the result.

    The levels are placed by one-dimensional k-means on each feature, so a
    boundary follows the density of that feature and is not a fixed quantile.
    The edges are returned as a table, because a rule naming level 0 of a
    feature means nothing without the interval that level covers.
    """
    discretizer = KBinsDiscretizer(n_bins=config.N_BINS, encode="ordinal",
                                   strategy="kmeans", random_state=config.SEED)
    binned = discretizer.fit_transform(frame[features].to_numpy(dtype=float))
    edges = [{"feature": name,
              **{"edge_{}".format(position): round(float(value), 6)
                 for position, value in enumerate(bin_edges)}}
             for name, bin_edges in zip(features, discretizer.bin_edges_)]

    table = pandas.DataFrame(binned, columns=features).astype(int)
    encoded = pandas.get_dummies(table, columns=features)
    for level in frame["diagnosis"].unique():
        encoded["diagnosis_" + level] = (frame["diagnosis"] == level).values
    return encoded, edges


def diagnosis_rules(rules: pandas.DataFrame) -> pandas.DataFrame:
    """The rules whose consequent is the diagnosis and nothing else."""
    keep = [len(consequent) == 1 and next(iter(consequent)) in DIAGNOSIS_ITEMS
            for consequent in rules["consequents"]]
    return rules[keep]


def main() -> int:
    record = utils.Metrics()
    with utils.Timer(record, "m03"):
        frame, counts = data.load_breast_cancer()
        record.update(counts, PREFIX)
        features = [column for column in frame.columns if column != "diagnosis"]

        encoded, edges = transactions(frame, features)
        utils.write_table(edges, "m03_bin_edges")
        utils.write_processed(encoded.astype(int), "breast_cancer_transactions")
        record.set(PREFIX + "items", int(encoded.shape[1]))
        record.set(PREFIX + "transactions", int(encoded.shape[0]))

        # --- what the support threshold can reach ---------------------------
        # An itemset cannot be more frequent than its rarest item, so no
        # itemset containing the malignant diagnosis can reach a support above
        # the malignant rate. The coursework mined at a support of 0.4, which
        # is above that rate, so every diagnosis rule it could find was a rule
        # about benign samples. The sweep records where each class becomes
        # reachable.
        malignant_rate = counts["rows_malignant"] / counts["rows_analyzed"]
        benign_rate = counts["rows_benign"] / counts["rows_analyzed"]
        record.set(PREFIX + "malignant_rate", malignant_rate)
        record.set(PREFIX + "benign_rate", benign_rate)
        record.set(PREFIX + "malignant_reachable_at_min_support",
                   config.MIN_SUPPORT <= malignant_rate)

        sweep = []
        for support in (0.4, 0.3, 0.2, 0.1):
            itemsets = apriori(encoded, min_support=support, use_colnames=True,
                               max_len=config.MAX_ITEMSET_LEN)
            reachable = {item: int(sum(item in itemset
                                       for itemset in itemsets["itemsets"]))
                         for item in DIAGNOSIS_ITEMS}
            sweep.append({"min_support": support,
                          "frequent_itemsets": int(len(itemsets)),
                          "itemsets_with_malignant":
                              reachable["diagnosis_malignant"],
                          "itemsets_with_benign": reachable["diagnosis_benign"]})
        utils.write_table(sweep, "m03_support_sweep")

        # --- the mining run the results are quoted from ---------------------
        itemsets = apriori(encoded, min_support=config.MIN_SUPPORT,
                           use_colnames=True, max_len=config.MAX_ITEMSET_LEN)
        record.set(PREFIX + "frequent_itemsets", int(len(itemsets)))
        record.set(PREFIX + "max_itemset_length", config.MAX_ITEMSET_LEN)
        by_length = itemsets["itemsets"].map(len).value_counts().sort_index()
        utils.write_table(
            [{"itemset_length": int(length), "itemsets": int(value)}
             for length, value in by_length.items()],
            "m03_itemsets_by_length")

        rules = association_rules(itemsets, num_itemsets=len(encoded),
                                  metric="conviction",
                                  min_threshold=config.MIN_CONVICTION)
        record.set(PREFIX + "rules", int(len(rules)))
        record.set(PREFIX + "min_conviction", config.MIN_CONVICTION)

        selected = diagnosis_rules(rules)
        record.set(PREFIX + "diagnosis_rules", int(len(selected)))
        if selected.empty:
            raise RuntimeError("no rule at this threshold names the diagnosis "
                               "as its only consequent")

        # Apriori enumerates itemsets over frozensets of strings, whose
        # iteration order depends on the per-process string hash seed, and
        # several rules here tie on lift to six decimal places. Ranking on lift
        # alone would therefore order the tied rows differently between two runs
        # that agree on every number. The key below is a total order, so the
        # table is a function of the data and not of the interpreter.
        ranked = sorted(
            [{"antecedent": ", ".join(sorted(row.antecedents)),
              "consequent": next(iter(row.consequents)),
              "support": round(float(row.support), 6),
              "confidence": round(float(row.confidence), 6),
              "lift": round(float(row.lift), 6),
              "conviction": round(float(row.conviction), 6)}
             for row in selected.itertuples()],
            key=lambda rule: (-rule["lift"], -rule["support"],
                              -rule["confidence"], rule["antecedent"]))
        utils.write_table(ranked[:20], "m03_top_rules")
        record.set(PREFIX + "best_lift", float(selected["lift"].max()))
        record.set(PREFIX + "best_confidence",
                   float(selected["confidence"].max()))
        record.set(PREFIX + "median_confidence",
                   float(selected["confidence"].median()))

        # --- the feature selection reading ----------------------------------
        # Every feature appearing in the antecedent of a diagnosis rule, with
        # the number of rules it appears in. This is the coursework's last step,
        # and it is a ranking of features and not a set of rules.
        appearances: dict[str, int] = {}
        for antecedent in selected["antecedents"]:
            for item in antecedent:
                appearances[item] = appearances.get(item, 0) + 1
        # Ties on the rule count are broken by the item name for the same
        # reason: the counts are accumulated by iterating frozensets.
        utils.write_table(
            [{"item": item, "rules": value}
             for item, value in sorted(appearances.items(),
                                       key=lambda pair: (-pair[1], pair[0]))],
            "m03_selected_items")
        record.set(PREFIX + "selected_items", len(appearances))
        record.set(PREFIX + "selected_features",
                   len({item.rsplit("_", 1)[0] for item in appearances}))

    record.save()
    print("model 3: {} frequent itemsets at support {}, {} rules at conviction "
          "{}, of which {} name the diagnosis alone".format(
              int(record.number(PREFIX + "frequent_itemsets")),
              config.MIN_SUPPORT, int(record.number(PREFIX + "rules")),
              config.MIN_CONVICTION,
              int(record.number(PREFIX + "diagnosis_rules"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
