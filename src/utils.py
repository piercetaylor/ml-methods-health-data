"""The metrics record, table writing, and the file digest.

Results are written once, to ``results/metrics.csv``, by whichever analysis
computes them. Every number quoted in the README or in ``docs/`` is read back
out of that file, so a claim in the prose and a claim in the pipeline cannot
drift apart.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config


class Metrics:
    """Append-only key/value record of everything the four analyses measured.

    The record is loaded from disk on construction and written back whole, so
    one analysis re-run in isolation keeps the quantities the other three
    recorded.
    """

    def __init__(self, path: Path = config.METRICS):
        self.path = path
        self.values: dict[str, str] = {}
        if path.exists():
            with open(path, newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self.values[row["key"]] = row["value"]

    def set(self, key: str, value: Any) -> None:
        if isinstance(value, float):
            value = f"{value:.6g}"
        self.values[key] = str(value)

    def update(self, mapping: dict, prefix: str = "") -> None:
        for key, value in mapping.items():
            self.set(prefix + key, value)

    def get(self, key: str) -> str:
        if key not in self.values:
            raise KeyError(f"metric '{key}' was never recorded")
        return self.values[key]

    def number(self, key: str) -> float:
        return float(self.get(key))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["key", "value"])
            for key in sorted(self.values):
                writer.writerow([key, self.values[key]])


def record_signature(path: Path = config.METRICS) -> str:
    """A digest of the recorded quantities that a re-run reproduces exactly.

    The wall-clock timings are excluded, because they differ between two runs
    that agree on every measured quantity.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        pairs = [
            (row["key"], row["value"])
            for row in csv.DictReader(handle)
            if not row["key"].startswith(config.RECORD_TIMING_PREFIX)
        ]
    text = "\n".join(f"{key}={value}" for key, value in sorted(pairs))
    return hashlib.sha256(text.encode()).hexdigest()[:config.RECORD_SIGNATURE_DIGITS]


def write_table(rows: Iterable[dict], name: str,
                columns: Sequence[str] | None = None) -> int:
    """Write one result table to ``results/<name>.csv`` and return its length.

    An empty table is refused. A figure is drawn from a table in ``results/``,
    so an empty one would produce an empty figure and no error.
    """
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write an empty table to results/{name}.csv")
    columns = list(columns or rows[0].keys())
    path = config.RESULTS / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_processed(frame, name: str) -> int:
    """Write one cleaned table or partition to ``data/processed/<name>.csv``.

    The analyses fit on frames built in memory, and these files are the same
    frames written out, so the exact rows behind every recorded number can be
    read without running anything. An empty frame is refused for the same
    reason an empty result table is.
    """
    if len(frame) == 0:
        raise ValueError(
            f"refusing to write an empty table to data/processed/{name}.csv")
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.PROCESSED / f"{name}.csv", index=False,
                 lineterminator="\n")
    return len(frame)


def read_table(name: str) -> list[dict]:
    path = config.RESULTS / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"results/{name}.csv is absent")
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    """The SHA-256 digest of a file as it was downloaded, read in blocks."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_checksums() -> dict[str, str]:
    """The recorded digests, keyed by file name."""
    if not config.CHECKSUMS.exists():
        raise FileNotFoundError("data/checksums.txt is absent. "
                                "Run: python data/download_data.py")
    recorded = {}
    for line in config.CHECKSUMS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value, name = line.split(None, 1)
        recorded[name.strip()] = value
    return recorded


class Timer:
    """Context manager recording the wall-clock time one stage took."""

    def __init__(self, record: Metrics, key: str):
        self.record = record
        self.key = config.RECORD_TIMING_PREFIX + key

    def __enter__(self):
        self.started = time.time()
        return self

    def __exit__(self, *exc):
        self.record.set(self.key, round(time.time() - self.started, 1))
        return False
