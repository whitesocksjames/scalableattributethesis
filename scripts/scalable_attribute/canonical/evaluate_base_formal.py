#!/usr/bin/env python3
"""Compatibility CLI forwarding to `../evaluation/evaluate_base_formal.py`."""

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    script_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(script_root.parents[1]))
    runpy.run_path(str(script_root / "evaluation" / "evaluate_base_formal.py"), run_name="__main__")
