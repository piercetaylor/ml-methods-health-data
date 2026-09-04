"""Loading and cleaning, one function per dataset.

Each loader reads the downloaded file, applies the cleaning the analysis needs,
and returns a frame plus a dictionary of counts describing what it changed. The
counts are recorded, so every exclusion in the prose has a number behind it.
"""

from __future__ import annotations

import pandas

from . import config


def _verify(frame: pandas.DataFrame, name: str) -> None:
    """Fail on a file whose shape is not the shape the analysis was written for.

    A silently changed release would otherwise be cleaned by rules that no
    longer match it.
    """
    source = config.SOURCES[name]
    expected = (source["expected_rows"], source["expected_columns"])
    if frame.shape != expected:
        raise ValueError("{} has shape {} and {} was expected".format(
            source["file"], frame.shape, expected))


def _read(name: str, **kwargs) -> pandas.DataFrame:
    path = config.RAW / config.SOURCES[name]["file"]
    if not path.exists():
        raise FileNotFoundError(
            "{} is absent. Run: python data/download_data.py".format(path))
    frame = pandas.read_csv(path, **kwargs)
    _verify(frame, name)
    return frame


# --- model 1, readmission ---------------------------------------------------
def icd9_group(code) -> str:
    """Map one ICD-9 code to its diagnostic group.

    The groups follow the scheme Strack et al. (2014) published with the data.
    A V or E supplementary code carries a letter prefix and does not parse as a
    number, and it falls to "other" along with any code outside every interval.
    """
    if pandas.isna(code):
        return "missing"
    try:
        value = float(str(code))
    except ValueError:
        return "other"
    whole = int(value)
    for label, intervals in config.ICD9_GROUPS:
        for low, high in intervals:
            if low <= whole <= high:
                return label
    return "other"


def load_diabetes() -> tuple[pandas.DataFrame, dict]:
    """The readmission cohort, cleaned, with the counts each step removed.

    Encounters whose discharge disposition records death or a transfer to
    hospice are removed, because such a patient cannot be readmitted and the
    disposition alone would predict the outcome. The weight column and the payer
    identifier are dropped, as are the two drug columns that take one value
    throughout. The admitting department is kept with its rare levels collapsed.
    The remaining rows are complete in every retained column.
    """
    frame = _read("diabetes", na_values=["?", "Unknown/Invalid"],
                  low_memory=False)
    counts = {"rows_released": len(frame),
              "encounters_released": frame["encounter_id"].nunique(),
              "patients_released": frame["patient_nbr"].nunique()}

    unobservable = frame["discharge_disposition_id"].isin(
        config.UNOBSERVABLE_DISPOSITIONS)
    counts["rows_unobservable_outcome"] = int(unobservable.sum())
    frame = frame[~unobservable].copy()

    for column in config.DIABETES_DROP_SPARSE:
        counts["missing_fraction_" + column] = round(
            float(frame[column].isna().mean()), 4)
    for column in config.DIABETES_DROP_CONSTANT:
        counts["distinct_values_" + column] = int(frame[column].nunique())
    frame = frame.drop(columns=list(config.DIABETES_DROP_SPARSE)
                       + list(config.DIABETES_DROP_CONSTANT))

    # The label carries three levels. The question is readmission inside thirty
    # days, so the other two levels form one negative class.
    frame["readmitted_30d"] = (
        frame["readmitted"] == config.READMIT_POSITIVE).astype(int)
    frame = frame.drop(columns=["readmitted"])

    # A1Cresult and max_glu_serum are absent where the test was not ordered.
    # That is a recorded clinical decision and not a missing measurement, so it
    # becomes its own level.
    for column in ("A1Cresult", "max_glu_serum"):
        frame[column] = frame[column].fillna("not measured")

    # The admitting department is kept, with its long tail collapsed and its
    # absence given a level of its own.
    share = frame["medical_specialty"].value_counts(normalize=True)
    kept = set(share[share >= config.SPECIALTY_MIN_SHARE].index)
    counts["specialty_levels_released"] = int(share.size)
    counts["specialty_levels_kept"] = len(kept)
    counts["specialty_not_recorded_fraction"] = round(
        float(frame["medical_specialty"].isna().mean()), 4)
    frame["medical_specialty"] = [
        "not recorded" if pandas.isna(value)
        else value if value in kept else "other"
        for value in frame["medical_specialty"]]

    # The admission type, admission source and discharge disposition are
    # released as integer codes into a published lookup table. The integers
    # order nothing, so they are carried as categories and one-hot encoded, and
    # not left as numbers a model would read as a scale.
    for column in config.DIABETES_CODED_CATEGORIES:
        frame[column] = "code " + frame[column].astype(int).astype(str)

    for column in ("diag_1", "diag_2", "diag_3"):
        frame[column + "_group"] = frame[column].map(icd9_group)
    frame = frame.drop(columns=["diag_1", "diag_2", "diag_3"])

    # Age is released as a ten-year bracket. The midpoint keeps the ordering
    # that a one-hot encoding of the bracket would discard.
    frame["age_midpoint"] = frame["age"].map(
        lambda text: int(text.strip("[)").split("-")[0]) + 5)
    frame = frame.drop(columns=["age"])

    before = len(frame)
    frame = frame.dropna()
    counts["rows_incomplete"] = before - len(frame)
    counts["rows_analyzed"] = len(frame)
    counts["patients_analyzed"] = frame["patient_nbr"].nunique()
    counts["positive_rate"] = round(float(frame["readmitted_30d"].mean()), 6)
    return frame.reset_index(drop=True), counts


# --- model 2, HCV -----------------------------------------------------------
def load_hcv() -> tuple[pandas.DataFrame, dict]:
    """The hepatitis C panel, complete cases, with the released category kept.

    The released column CGT holds gamma-glutamyl transferase and is renamed GGT
    to match the assay's usual abbreviation. The category "0s=suspect Blood
    Donor" marks a donor whose values were questioned, and it holds too few rows
    to stand as its own group, so those rows are removed and counted.
    """
    frame = _read("hcv")
    frame = frame.rename(columns={"CGT": "GGT"})
    counts = {"rows_released": len(frame)}

    for name, value in frame["Category"].value_counts().items():
        counts["released_" + name.split("=")[0]] = int(value)

    suspect = frame["Category"].str.startswith("0s")
    counts["rows_suspect_donor"] = int(suspect.sum())
    frame = frame[~suspect].copy()

    before = len(frame)
    frame = frame.dropna(subset=list(config.HCV_FEATURES))
    counts["rows_incomplete"] = before - len(frame)
    counts["rows_analyzed"] = len(frame)

    # The released label is a string carrying its own ordinal prefix. The
    # integer is kept for the contingency table and the text for the figures.
    frame["category_index"] = frame["Category"].str.split("=").str[0].astype(int)
    frame["category_label"] = frame["Category"].str.split("=").str[1]
    counts["categories_analyzed"] = int(frame["category_index"].nunique())
    # The rows removed for incompleteness are not spread evenly across the
    # categories, so the analyzed count of each is recorded beside the released
    # count. The difference is a limitation of the complete-case analysis and
    # the clustering results have to be read against it.
    for label, value in frame["category_label"].value_counts().items():
        counts["analyzed_" + label.strip().lower().replace(" ", "_")] = int(value)
    return frame.reset_index(drop=True), counts


# --- model 4, BUPA ----------------------------------------------------------
def load_bupa() -> tuple[pandas.DataFrame, dict]:
    """The liver enzyme panel with the split flag removed.

    The seventh released column, selector, is a train and test split flag and
    not a clinical measurement. It is dropped here and its absence from the
    predictors is checked by the schema gate. The release also holds rows
    identical in all seven columns; those are collapsed, because a duplicated
    row placed on both sides of a split makes the two sides overlap in content
    while remaining disjoint by index.
    """
    frame = _read("bupa")
    counts = {"rows_released": len(frame)}
    if config.BUPA_EXCLUDED not in frame.columns:
        raise ValueError(
            "the released file no longer carries the {} column, so the "
            "exclusion this loader documents cannot be verified".format(
                config.BUPA_EXCLUDED))
    counts["selector_distinct_values"] = int(
        frame[config.BUPA_EXCLUDED].nunique())
    frame = frame.drop(columns=[config.BUPA_EXCLUDED])

    duplicated = frame.duplicated()
    counts["rows_duplicated"] = int(duplicated.sum())
    frame = frame[~duplicated].copy()
    counts["rows_analyzed"] = len(frame)
    counts["drinks_zero_rows"] = int((frame[config.BUPA_TARGET] == 0).sum())
    return frame.reset_index(drop=True), counts


# --- model 3, breast cancer -------------------------------------------------
def load_breast_cancer() -> tuple[pandas.DataFrame, dict]:
    """The Wisconsin diagnostic measurements as scikit-learn distributes them.

    The bundled copy is the same 569 by 30 table the UCI record serves, and it
    needs no download. Its target is coded 0 for malignant and 1 for benign,
    so a named column is added beside it and the numeric code is not used.
    """
    from sklearn.datasets import load_breast_cancer as sklearn_loader

    bundle = sklearn_loader()
    frame = pandas.DataFrame(bundle.data, columns=bundle.feature_names)
    frame["diagnosis"] = [bundle.target_names[value] for value in bundle.target]
    counts = {"rows_analyzed": len(frame),
              "features": int(len(bundle.feature_names)),
              "rows_malignant": int((frame["diagnosis"] == "malignant").sum()),
              "rows_benign": int((frame["diagnosis"] == "benign").sum()),
              "rows_incomplete": int(frame.isna().any(axis=1).sum())}
    return frame, counts
