#!/usr/bin/env python
"""Prove that each gate can fail, by breaking one thing at a time.

    python .checks/inject_faults.py
    python .checks/inject_faults.py --only gate_02_schema

A gate that has never failed is not evidence. Each fault below changes one file,
runs the gate that should catch it, and restores the file whether the gate
caught it or not. The run reports a fault as CAUGHT when the gate exits non-zero
and as MISSED when it does not, and a missed fault is a defect in the gate.

Gate 05 is not driven from here. Its fault is a changed value in
``results/metrics.csv``, and catching it requires the full pipeline re-run that
the gate performs, so it is exercised on its own.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CHECKS = Path(__file__).resolve().parent
ROOT = CHECKS.parent


def edit(path: Path, change) -> "Fault":
    return Fault(path, change)


class Fault:
    """One reversible edit to one file."""

    def __init__(self, path: Path, change):
        self.path = path
        self.change = change
        self.original = b""

    def __enter__(self):
        self.original = self.path.read_bytes()
        self.path.write_text(self.change(self.original.decode("utf-8")),
                             encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.path.write_bytes(self.original)
        return False


def set_metric(key: str, value: str):
    """Replace one value in results/metrics.csv."""
    def change(text: str) -> str:
        replaced, count = re.subn(r"(?m)^{},.*$".format(re.escape(key)),
                                  "{},{}".format(key, value), text)
        if count != 1:
            raise RuntimeError(
                "{} appears {} times in the record".format(key, count))
        return replaced
    return change


def replace(before: str, after: str):
    def change(text: str) -> str:
        if before not in text:
            raise RuntimeError("the text to break is not present: " + before)
        return text.replace(before, after, 1)
    return change


FAULTS = (
    ("gate_00_environment",
     "requirements.txt pins a version that is not installed",
     lambda: edit(ROOT / "requirements.txt",
                  replace("pandas==", "pandas==0.0.0  # "))),
    ("gate_01_acquisition",
     "one recorded digest no longer matches the downloaded file",
     lambda: edit(ROOT / "data" / "checksums.txt",
                  replace("a6ebd", "b6ebd"))),
    ("gate_02_schema",
     "the BUPA split flag is put back among the predictors",
     lambda: edit(ROOT / "src" / "config.py",
                  replace('BUPA_PREDICTORS = ("mcv"',
                          'BUPA_PREDICTORS = ("selector", "mcv"'))),
    ("gate_03_preparation",
     "a patient is recorded on both sides of the readmission partition",
     lambda: edit(ROOT / "results" / "metrics.csv",
                  set_metric("m01.split.groups_on_both_sides", "5"))),
    ("gate_04_modeling",
     "one feature separates readmission on its own",
     lambda: edit(ROOT / "results" / "metrics.csv",
                  set_metric("m01.max_univariate_auc", "0.99"))),
)


def run(gate: str) -> int:
    return subprocess.run([sys.executable, str(CHECKS / (gate + ".py"))],
                          cwd=ROOT, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode


def main(only: str | None) -> int:
    missed = []
    for gate, description, make in FAULTS:
        if only and gate != only:
            continue
        clean = run(gate)
        if clean != 0:
            print("  [SETUP] {} already fails before any fault is injected. "
                  "Fix that first.".format(gate))
            missed.append(gate)
            continue
        with make():
            broken = run(gate)
        caught = broken != 0
        print("  [{}] {}: {}".format("CAUGHT" if caught else "MISSED",
                                     gate, description))
        if not caught:
            missed.append(gate)

    print("\nGate 05 is exercised separately. Change one value in "
          "results/metrics.csv,\nrun .checks/gate_05_reproducibility.py, and "
          "the re-run will disagree with it.")
    if missed:
        print("\n{} fault(s) were not caught: {}".format(len(missed),
                                                         ", ".join(missed)))
        return 1
    print("\nEvery injected fault was caught by the gate that guards it.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="run one gate only")
    sys.exit(main(parser.parse_args().only))
