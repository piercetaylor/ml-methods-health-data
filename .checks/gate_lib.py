"""Shared harness for the phase gates.

A gate verifies one phase of the pipeline and exits 0 or non-zero. Every check
prints what it looked at and what it found, so a passing gate is evidence and
not a silent success.

Two properties here are load-bearing. A check that cannot be run has not passed,
so a quantity that was never recorded fails the check that reads it and does not
stop the gate before the remaining checks run. And a check whose evidence cannot
be produced fails as well, because a pass with nothing printed beside it is the
silent success this harness exists to prevent.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.utils import Metrics  # noqa: E402

_STARTED = time.time()
_STATE = {"pass": 0, "fail": 0, "skip": 0, "name": "gate"}
_READERS: list = []


def gate(name: str) -> None:
    _STATE["name"] = name
    print("\n=== {} ===".format(name))


class _Record:
    """The metrics record as a gate reads it.

    ``Metrics.get`` raises on a key it does not hold, and a gate builds its
    condition before :func:`check` is entered, so one absent key would stop the
    whole gate where one failed check was wanted. Here an absent key reads as a
    value that fails every comparison, and the key is kept so that
    :func:`finish` reports what was read and never found.
    """

    def __init__(self, record: Metrics) -> None:
        self._record = record
        self.values = record.values
        self.absent: list[str] = []

    def get(self, key: str) -> str:
        if key in self.values:
            return self._record.get(key)
        self.absent.append(key)
        return "absent"

    def number(self, key: str) -> float:
        if key in self.values:
            return self._record.number(key)
        self.absent.append(key)
        return float("nan")


def check(label: str, condition, detail="") -> bool:
    """Record one check. ``condition`` and ``detail`` may each be a callable.

    A callable is evaluated inside this function, so an expression that raises
    fails the check it belongs to and does not end the gate.
    """
    try:
        text = str(detail() if callable(detail) else detail)
        evidenced = True
    except Exception as error:
        text = "the detail of this check could not be read: {}".format(error)
        evidenced = False
    try:
        ok = bool(condition() if callable(condition) else condition)
    except Exception as error:
        ok = False
        text = (text + " [error: {}]".format(error)).strip()
    ok = ok and evidenced
    _STATE["pass" if ok else "fail"] += 1
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", label,
                               " -- " + text if text else ""))
    return ok


def skip(label: str, reason: str) -> None:
    """Record a check that was not run. A skip asserts nothing."""
    _STATE["skip"] += 1
    print("  [SKIP] {} -- {}".format(label, reason))


def metrics() -> _Record:
    if not config.METRICS.exists():
        print("results/metrics.csv is absent. Run: python analysis/run_all.py")
        sys.exit(1)
    reader = _Record(Metrics())
    _READERS.append(reader)
    return reader


def table(name: str) -> list[dict]:
    from src.utils import read_table
    return read_table(name)


def finish() -> None:
    for reader in _READERS:
        absent = sorted(set(reader.absent))
        check("every quantity a check read is present in the record",
              not absent,
              ", ".join(absent) if absent
              else "{} quantities available".format(len(reader.values)))
    total = _STATE["pass"] + _STATE["fail"]
    print("\n{}: {} ({} of {} checks passed{}, {:.1f}s)".format(
        _STATE["name"], "PASS" if _STATE["fail"] == 0 else "FAIL",
        _STATE["pass"], total,
        ", {} skipped".format(_STATE["skip"]) if _STATE["skip"] else "",
        time.time() - _STARTED))
    sys.exit(0 if _STATE["fail"] == 0 else 1)
