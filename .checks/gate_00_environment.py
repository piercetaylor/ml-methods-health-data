#!/usr/bin/env python
"""Gate 00. The interpreter, the installed versions and the directory layout.

A result produced under a different version of a library is not the result this
repository records, so the pinned versions in requirements.txt are compared
against what is importable.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_lib as harness  # noqa: E402
from src import config  # noqa: E402

MODULES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
           "matplotlib": "matplotlib", "scikit-learn": "sklearn",
           "mlxtend": "mlxtend", "statsmodels": "statsmodels",
           "ucimlrepo": "ucimlrepo"}


def pinned() -> dict[str, str]:
    """The versions requirements.txt pins, read back from the committed file."""
    path = config.ROOT / "requirements.txt"
    versions = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        versions[name.strip()] = version.strip()
    return versions


harness.gate("gate 00: environment")

harness.check("the interpreter is Python 3.11 or later",
              sys.version_info >= (3, 11),
              "{}.{}.{}".format(*sys.version_info[:3]))

requirements = pinned()
harness.check("requirements.txt pins every module the pipeline imports",
              set(MODULES) <= set(requirements),
              "pinned: {}".format(", ".join(sorted(requirements))))

# The installed version is read from the distribution metadata and not from a
# module attribute, because not every distribution exposes __version__. The
# module is imported beside it, so a distribution recorded as installed and not
# importable fails the check.
for distribution, module in sorted(MODULES.items()):
    expected = requirements.get(distribution)
    try:
        importlib.import_module(module)
        found = importlib.metadata.version(distribution)
    except Exception as error:
        found = "not available: {}".format(error)
    harness.check("{} matches the pinned version".format(distribution),
                  found == expected,
                  "pinned {}, found {}".format(expected, found))

for name, path in (("data", config.DATA), ("data/raw", config.RAW),
                   ("results", config.RESULTS), ("figures", config.FIGURES),
                   ("src", config.ROOT / "src"),
                   ("analysis", config.ROOT / "analysis"),
                   ("docs", config.ROOT / "docs")):
    harness.check("{} exists".format(name), path.is_dir(), str(path))

harness.check("the download script and the digests are committed",
              (config.DATA / "download_data.py").is_file()
              and config.CHECKSUMS.is_file(),
              "data/download_data.py and data/checksums.txt")

harness.finish()
