# Data

## `raw/`

The three files under `raw/` are the CSV each dataset's record at the UCI
Machine Learning Repository serves at
`https://archive.ics.uci.edu/static/public/<id>/data.csv`, committed byte for
byte as downloaded. `checksums.txt` records the SHA-256 digest of each, and
`download_data.py` verifies the committed file against it on every run, so a
file here can be shown to be the file the repository published. Running
`download_data.py` with the files absent fetches them again from the same URL.

| File | UCI record | Rows × columns | Citation |
|---|---|---|---|
| `diabetes_130_us_hospitals.csv` | 296 | 101,766 × 50 | Clore, Cios, DeShazo and Strack (2014) |
| `hcv_data.csv` | 571 | 615 × 14 | Lichtinghagen, Klawonn and Hoffmann (2020) |
| `liver_disorders.csv` | 60 | 345 × 7 | BUPA Medical Research Ltd., donated by Forsyth (1990) |

All three are released under the Creative Commons Attribution 4.0 International
license, which permits redistribution with attribution. The full citations,
with the record URLs, are in `docs/references.md`. The files are unmodified;
every cleaning step is applied by `src/data.py` when an analysis loads them.

The fourth dataset, the Wisconsin diagnostic breast cancer measurements
(Wolberg, Street and Mangasarian 1995, UCI record 17), ships inside
scikit-learn as `sklearn.datasets.load_breast_cancer` and is read from there.

## `processed/`

The pipeline writes the table each analysis fitted on, and the partition it
fitted under, to `processed/`. These are the frames the analyses hold in
memory, written out, so the exact rows behind every recorded number can be
read without running anything. Gate 05 re-creates them in a scratch directory
and compares each against the committed copy byte for byte.

| File | What it is |
|---|---|
| `readmission_cohort.csv` | The 97,108 encounters after cleaning, with the derived columns |
| `readmission_partition.csv` | Encounter and patient identifiers with the train or test assignment |
| `hcv_panel.csv` | The 608 subjects with the imputed values in place and an `imputed` flag |
| `breast_cancer_measurements.csv` | The 569 samples as scikit-learn distributes them, with the diagnosis named |
| `breast_cancer_transactions.csv` | The 92-item one-hot table Apriori mined |
| `liver_panel.csv` | The 341 subjects with `selector` removed and duplicates collapsed |
| `liver_partition.csv` | The same rows with the train or test assignment |
