"""Train and test partitions, and the evidence that they are disjoint.

Two of the four analyses need a held-out partition. The readmission data holds
several encounters for the same patient, so its partition is drawn over
patients and not over rows. The BUPA data holds one row per subject, so a row
partition is a subject partition there.
"""

from __future__ import annotations

import pandas
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from . import config


def grouped_split(frame: pandas.DataFrame, group: str,
                  test_fraction: float = config.TEST_FRACTION,
                  seed: int = config.SEED
                  ) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Split whole groups into train and test.

    ``GroupShuffleSplit`` keeps every row of one group on one side of the
    boundary and does not stratify, so the outcome rate of the two sides is a
    property of the draw. The draw is taken once at the recorded seed and
    :func:`report` records the two rates, so a reader can see how far apart
    they fell and does not have to assume they match.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction,
                                 random_state=seed)
    train_index, test_index = next(splitter.split(frame, groups=frame[group]))
    return (frame.iloc[train_index].reset_index(drop=True),
            frame.iloc[test_index].reset_index(drop=True))


def row_split(frame: pandas.DataFrame,
              test_fraction: float = config.TEST_FRACTION,
              seed: int = config.SEED
              ) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Split rows into train and test, for a table of one row per subject."""
    train, test = train_test_split(frame, test_size=test_fraction,
                                   random_state=seed, shuffle=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def report(train: pandas.DataFrame, test: pandas.DataFrame,
           group: str | None = None, label: str | None = None) -> dict:
    """Describe one partition and check that its two sides do not overlap.

    Four quantities are returned for every partition: the two sizes, the test
    share, and the number of pairs of rows, one from each side, that agree in
    every column. A row duplicated in the release would sit on both sides while
    the index partition remained disjoint, so the overlap check compares content
    and not position. On a partition with no overlap the count is zero and the
    pair counting does not arise; where copies exist the count is of matching
    pairs and is larger than the number of distinct rows involved.

    Where a group column is given, the number of groups appearing on both sides
    is returned as well, and that is the quantity the preparation gate reads for
    the readmission data.
    """
    result = {"train_rows": len(train), "test_rows": len(test),
              "test_fraction": round(len(test) / (len(train) + len(test)), 4),
              "rows_on_both_sides": int(pandas.merge(train, test,
                                                     how="inner").shape[0])}
    if group is not None:
        result["train_groups"] = int(train[group].nunique())
        result["test_groups"] = int(test[group].nunique())
        result["groups_on_both_sides"] = len(
            set(train[group]) & set(test[group]))
    if label is not None:
        result["train_positive_rate"] = round(float(train[label].mean()), 6)
        result["test_positive_rate"] = round(float(test[label].mean()), 6)
    return result
