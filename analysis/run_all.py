#!/usr/bin/env python
"""Run the four analyses in order and draw every figure.

    python analysis/run_all.py
    python analysis/run_all.py --only m02 m04

Each analysis writes its own quantities into ``results/metrics.csv`` under its
own prefix and its own tables into ``results/``. The figures are drawn last,
from those tables, so a figure and the number quoted beside it come from the
same file. The digest printed at the end is the quantity the reproducibility
gate compares between two runs.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, figures, utils  # noqa: E402

ANALYSES = (("m01", "analysis.m01_classification"),
            ("m02", "analysis.m02_clustering"),
            ("m03", "analysis.m03_association_rules"),
            ("m04", "analysis.m04_regression"))


def main(only: list[str] | None) -> int:
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for name, module in ANALYSES:
        if only and name not in only:
            print("skipping {}".format(name))
            continue
        print("\n--- {} ---".format(module))
        code = importlib.import_module(module).main()
        if code != 0:
            print("{} exited {}. Stopping.".format(module, code))
            return code

    print("\n--- figures ---")
    for name in figures.draw_all():
        print("  {}".format(name))

    record = utils.Metrics()
    record.set("pipeline.figures", len(figures.ALL))
    record.set("timing.pipeline", round(time.time() - started, 1))
    record.save()
    print("\n{} quantities recorded in {}".format(len(record.values),
                                                  config.METRICS))
    print("record digest: {}".format(utils.record_signature(config.METRICS)))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None,
                        help="run only the named analyses, for example m02 m04")
    sys.exit(main(parser.parse_args().only))
