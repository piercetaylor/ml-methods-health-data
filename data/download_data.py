#!/usr/bin/env python
"""Download the three UCI datasets and verify them against recorded digests.

    python data/download_data.py            verify, downloading what is absent
    python data/download_data.py --record   download and write data/checksums.txt

The bytes served by the repository are digested as they arrive, so the file the
pipeline reads can be shown to be the file the repository published. The fourth
dataset ships inside scikit-learn and is not downloaded.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, utils  # noqa: E402

RETRIES = 5
USER_AGENT = "ml-methods-health-data/1.0 (+https://github.com/piercetaylor)"


def fetch(url: str, destination: Path) -> Path:
    """Fetch one URL to a temporary name and move it into place when complete.

    A partial download left under the final name would pass the existence test
    on the next run and fail the digest, so the move is the last step.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = response.read()
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last = error
            time.sleep(min(2 ** attempt, 20))
    else:
        raise RuntimeError(f"{url} failed after {RETRIES} attempts: {last}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def main(record: bool) -> int:
    config.RAW.mkdir(parents=True, exist_ok=True)
    recorded = {} if record else utils.read_checksums()
    lines = []
    failures = 0

    for name, source in config.SOURCES.items():
        url = config.UCI_STATIC.format(id=source["uci_id"])
        path = config.RAW / source["file"]
        if not path.exists():
            print(f"downloading {source['file']} from {url}")
            fetch(url, path)
        found = utils.digest(path)
        size = path.stat().st_size
        if record:
            print(f"  {source['file']}: {found}  ({size} bytes)")
            lines.append(f"{found}  {source['file']}")
            continue
        expected = recorded.get(source["file"])
        if expected is None:
            print(f"  FAIL {source['file']}: no digest is recorded for it")
            failures += 1
        elif expected != found:
            print(f"  FAIL {source['file']}: recorded {expected}, found {found}")
            failures += 1
        else:
            print(f"  ok   {source['file']}: {found[:16]}... ({size} bytes)")

    if record:
        header = ("# SHA-256 of the file each dataset's UCI record serves at\n"
                  "# https://archive.ics.uci.edu/static/public/<id>/data.csv\n")
        config.CHECKSUMS.write_text(header + "\n".join(lines) + "\n",
                                    encoding="utf-8")
        print(f"\nwrote {config.CHECKSUMS}")
        return 0

    if failures:
        print(f"\n{failures} of {len(config.SOURCES)} files did not verify.")
        return 1
    print(f"\nAll {len(config.SOURCES)} files verify against data/checksums.txt.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true",
                        help="write data/checksums.txt from what was downloaded")
    sys.exit(main(parser.parse_args().record))
