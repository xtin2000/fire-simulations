"""Project directory layout."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DERIVED_DIR = PROJECT_ROOT / "data" / "derived"
RUNS_DIR = PROJECT_ROOT / "runs"
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
CSIRO_RAW_DIR = RAW_DIR / "csiro"
CSIRO_RUNS_DIR = RUNS_DIR / "csiro"
