#!/usr/bin/env python3
"""Regenerate the results tables from run artifacts. Equivalent to `arc3cb results`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arc3cb.results import render_results  # noqa: E402

if __name__ == "__main__":
    runs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    print(render_results(runs_dir))
