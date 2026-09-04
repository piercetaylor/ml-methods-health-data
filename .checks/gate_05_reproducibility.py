"""Gate 05. The pipeline reproduces its own recorded results from nothing.

The whole pipeline is run again into an empty scratch directory, and every
quantity and every table it produces is compared against the committed ones.
Wall-clock timings are excluded, because two runs that agree on every measured
quantity still differ in how long they took.

    python .checks/gate_05_reproducibility.py
    SKIP_RERUN=1 python .checks/gate_05_reproducibility.py

The re-run takes about as long as the pipeline, which is dominated by the grid
search over the network. A skipped re-run asserts nothing about
reproducibility, and the gate says so.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_lib as harness  # noqa: E402
from src import config, utils  # noqa: E402

harness.gate("gate 05: reproducibility")


def read(path: Path) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["key"]: row["value"] for row in csv.DictReader(handle)
                if not row["key"].startswith(config.RECORD_TIMING_PREFIX)}


committed = read(config.METRICS)
harness.check("the committed record holds the quantities the prose quotes",
              len(committed) > 100,
              "{} quantities excluding timings".format(len(committed)))

if os.environ.get("SKIP_RERUN"):
    harness.skip("the pipeline reproduces every recorded quantity",
                 "SKIP_RERUN is set, so no re-run was attempted")
    harness.skip("the pipeline reproduces every recorded table",
                 "SKIP_RERUN is set, so no re-run was attempted")
    harness.skip("the pipeline reproduces every processed table",
                 "SKIP_RERUN is set, so no re-run was attempted")
    harness.finish()

with tempfile.TemporaryDirectory(prefix="ml-methods-rerun-") as scratch:
    results = Path(scratch) / "results"
    processed = Path(scratch) / "processed"
    environment = dict(os.environ,
                       ML_METHODS_RESULTS=str(results),
                       ML_METHODS_FIGURES=str(Path(scratch) / "figures"),
                       ML_METHODS_PROCESSED=str(processed),
                       PYTHONIOENCODING="utf-8")
    print("  re-running the pipeline into {}".format(results))
    completed = subprocess.run(
        [sys.executable, str(config.ROOT / "analysis" / "run_all.py")],
        cwd=config.ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in completed.stdout.splitlines():
        if line.strip().startswith(("model ", "record digest", "---")):
            print("    {}".format(line.strip()))
    harness.check("the re-run completed", completed.returncode == 0,
                  "exit code {}".format(completed.returncode))

    rerun_metrics = results / "metrics.csv"
    if not rerun_metrics.exists():
        harness.check("the re-run wrote a record", False, str(rerun_metrics))
        harness.finish()

    produced = read(rerun_metrics)
    missing = sorted(set(committed) - set(produced))
    added = sorted(set(produced) - set(committed))
    differing = sorted(key for key in set(committed) & set(produced)
                       if committed[key] != produced[key])
    harness.check("the re-run recorded the same set of quantities",
                  not missing and not added,
                  "{} absent, {} unexpected".format(len(missing), len(added))
                  + ("" if not (missing or added)
                     else "; " + ", ".join((missing + added)[:6])))
    harness.check("every recorded quantity has the same value",
                  not differing,
                  "{} of {} differ".format(len(differing), len(committed))
                  + ("" if not differing else "; first: {} was {} and is now {}"
                     .format(differing[0], committed[differing[0]],
                             produced[differing[0]])))
    harness.check("the record digest is unchanged",
                  utils.record_signature(config.METRICS)
                  == utils.record_signature(rerun_metrics),
                  "{} against {}".format(
                      utils.record_signature(config.METRICS),
                      utils.record_signature(rerun_metrics)))

    committed_tables = sorted(path.name for path
                              in (config.ROOT / "results").glob("*.csv")
                              if path.name != "metrics.csv")
    unequal = [name for name in committed_tables
               if not (results / name).exists()
               or (results / name).read_bytes()
               != (config.ROOT / "results" / name).read_bytes()]
    harness.check("every recorded table is reproduced byte for byte",
                  not unequal,
                  "{} of {} tables differ".format(len(unequal),
                                                  len(committed_tables))
                  + ("" if not unequal else "; " + ", ".join(unequal[:6])))

    # The cleaned tables and partitions under data/processed are what every
    # analysis fitted on, so they are compared the same way. A cleaning step
    # that changed would show here before it showed in any result.
    committed_processed = sorted(path.name for path
                                 in (config.ROOT / "data" / "processed")
                                 .glob("*.csv"))
    unequal_processed = [
        name for name in committed_processed
        if not (processed / name).exists()
        or (processed / name).read_bytes()
        != (config.ROOT / "data" / "processed" / name).read_bytes()]
    harness.check("every processed table is reproduced byte for byte",
                  committed_processed and not unequal_processed,
                  "{} of {} tables differ".format(len(unequal_processed),
                                                  len(committed_processed))
                  + ("" if not unequal_processed
                     else "; " + ", ".join(unequal_processed[:6])))

harness.finish()
