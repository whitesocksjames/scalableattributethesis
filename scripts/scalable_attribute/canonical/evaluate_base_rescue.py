#!/usr/bin/env python3
"""Compatibility CLI forwarding to `../historical/evaluate_base_rescue.py`."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(script_root.parents[1]))
    runpy.run_path(str(script_root / "historical" / "evaluate_base_rescue.py"), run_name="__main__")
