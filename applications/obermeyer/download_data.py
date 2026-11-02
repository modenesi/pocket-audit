"""Download the public synthetic data of Obermeyer et al. (Science 2019) at a pinned commit.

Source: https://gitlab.com/labsysmed/dissecting-bias (synthetic patient-years generated with synthpop from the
authors' analytic file; the repository has no license file, so the data are downloaded, not redistributed).
Saves ``data_new.csv`` and ``data_dictionary.md`` to the gitignored ``data/obermeyer2019/`` and checks the
SHA-256 recorded below.

    python applications/obermeyer/download_data.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

URL = "https://gitlab.com/labsysmed/dissecting-bias.git"
COMMIT = "f7a3b4d86e0dd3c71b1bd28a6706dc5fd85548ad"      # 2022-09-03
SHA256 = "5341f90f3a1d330557620af1c734e552abe0ad57053055dd13dbccc6d8384d74"   # data/data_new.csv
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "obermeyer2019"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "data_new.csv"
    if target.exists() and sha256(target) == SHA256:
        print(f"already present and verified: {target}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "--quiet", URL, tmp], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "--quiet", COMMIT], check=True)
        for name in ("data_new.csv", "data_dictionary.md"):
            (OUT / name).write_bytes((Path(tmp) / "data" / name).read_bytes())
    digest = sha256(target)
    if digest != SHA256:
        print(f"checksum mismatch: {digest}", file=sys.stderr)
        return 1
    print(f"saved and verified: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
