#!/usr/bin/env python3
"""Validate the lightweight thesis diagnostics and refresh its checksums."""

import csv
import hashlib
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent.parent
TABLES = PACKAGE / "tables"


def rows(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


q3 = rows("q3_case_diagnostics.csv")
assert len(q3) == 29
assert sum(row["status"] == "EXACT_PASS" for row in q3) == 28
assert sum(row["status"] == "SENSITIVITY_INCLUDED" for row in q3) == 1
sensitivity = [row for row in q3 if row["status"] == "SENSITIVITY_INCLUDED"]
assert [(row["sample"], row["point"]) for row in sensitivity] == [
    ("redandblack_viewdep_vox12", "8k")
]

diagnostic256 = rows("diagnostic256_vs_512.csv")
assert len(diagnostic256) == 20
assert all(int(row["enhancement_bits_256"]) >= 8 for row in diagnostic256)
assert sum(float(row["delta_YUV611_512"]) > 0.01 for row in diagnostic256) == 19

pairs = rows("enhancement_effectiveness_all_pairs.csv")
assert len(pairs) == 140
assert sum(float(row["delta_YUV611"]) < 0 for row in pairs) == 13

files = sorted(
    path for path in PACKAGE.rglob("*")
    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
)
manifest = PACKAGE / "SHA256SUMS"
with manifest.open("w", encoding="utf-8") as handle:
    for path in files:
        if path == manifest:
            continue
        handle.write(f"{sha256(path)}  {path.relative_to(PACKAGE)}\n")
print("PASS: 29 Q3 cases, 20 diagnostic-256 samples, 140 frozen Base/Full pairs")
