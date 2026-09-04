#!/usr/bin/env python
"""Gate 01. The downloaded files are the files whose digests were recorded.

The digest is computed over the bytes on disk, so a truncated download, a
re-served file, or an edit made by hand all fail this gate. The shape of each
file is checked beside its digest, because a file that verifies and holds a
different number of rows than the analysis was written for is still the wrong
input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_lib as harness  # noqa: E402
from src import config, utils  # noqa: E402

harness.gate("gate 01: acquisition")

try:
    recorded = utils.read_checksums()
except FileNotFoundError as error:
    recorded = {}
    harness.check("data/checksums.txt is present", False, str(error))

harness.check("a digest is recorded for every source",
              set(recorded) == {source["file"]
                                for source in config.SOURCES.values()},
              "recorded: {}".format(", ".join(sorted(recorded))))

for name, source in sorted(config.SOURCES.items()):
    path = config.RAW / source["file"]
    harness.check("{} is present".format(source["file"]), path.is_file(),
                  str(path))
    if not path.is_file():
        continue
    found = utils.digest(path)
    harness.check("{} matches its recorded digest".format(source["file"]),
                  recorded.get(source["file"]) == found,
                  "recorded {}..., found {}...".format(
                      str(recorded.get(source["file"]))[:16], found[:16]))
    frame = pandas.read_csv(path, low_memory=False)
    expected = (source["expected_rows"], source["expected_columns"])
    harness.check("{} has the expected shape".format(source["file"]),
                  frame.shape == expected,
                  "expected {}, found {}".format(expected, frame.shape))

harness.check("the UCI record identifiers are the ones the citations name",
              [config.SOURCES[key]["uci_id"]
               for key in ("diabetes", "hcv", "bupa")] == [296, 571, 60],
              "diabetes 296, HCV 571, liver disorders 60")

# The fourth dataset is not downloaded. The check that it is available is that
# scikit-learn serves it at the shape the analysis was written for.
try:
    from sklearn.datasets import load_breast_cancer
    shape = load_breast_cancer().data.shape
except Exception as error:  # noqa: BLE001
    shape = "not available: {}".format(error)
harness.check("scikit-learn serves the breast cancer data at 569 by 30",
              shape == (569, 30), str(shape))

harness.finish()
