#!/usr/bin/env python3
"""Compatibility CLI forwarding to `diagnostics/aggregate_reference_rd.py`."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_root.parents[1]))
    runpy.run_path(str(script_root / "diagnostics" / "aggregate_reference_rd.py"), run_name="__main__")
