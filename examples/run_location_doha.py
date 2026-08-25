#!/usr/bin/env python3
"""Run S2SR super-resolution for Doha, Qatar.

Wrapper around scripts/run_location.py with Doha-specific defaults:
central Doha coordinates and a default target date. Outputs land in the
organized layout outputs/QA/<date>/<inference_id>/ with all scratch state
in a transient .work/ directory removed when the run ends. Every
run_location.py option remains available, e.g.:

  python examples/run_location_doha.py --search-only
  python examples/run_location_doha.py
  python examples/run_location_doha.py --date 2026-08-09 --tile-size 96
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_location


DOHA_LONGITUDE = 51.5310
DOHA_LATITUDE = 25.2886
DEFAULT_DATE = "2026-08-14"


def has_flag(argv: list[str], flag: str) -> bool:
    return any(item == flag or item.startswith(flag + "=") for item in argv)


def build_argv() -> list[str]:
    argv = sys.argv[1:]
    defaults: list[str] = []
    if not has_flag(argv, "--lon"):
        defaults += ["--lon", str(DOHA_LONGITUDE)]
    if not has_flag(argv, "--lat"):
        defaults += ["--lat", str(DOHA_LATITUDE)]
    if not has_flag(argv, "--date"):
        defaults += ["--date", DEFAULT_DATE]
    return defaults + argv


if __name__ == "__main__":
    run_location.main(build_argv())
