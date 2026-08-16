#!/usr/bin/env python3
"""Run the whole pipeline end to end.

Every number quoted anywhere in this repository is produced by one of these
stages and written to outputs/. Nothing is copied by hand, so re-running this
regenerates the write up's evidence from the source workbook.

    python run_pipeline.py              run every stage
    python run_pipeline.py --from model skip ingestion and cleaning
    python run_pipeline.py --only lift  run a single stage
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pricing.config import load_config  # noqa: E402

STAGES: list[tuple[str, str, str]] = [
    ("verify", "pricing.data.verify", "profile the raw workbook and check its claims"),
    ("clean", "pricing.data.clean", "apply cleaning rules and record what was dropped"),
    ("features", "pricing.features", "build model features"),
    ("model", "pricing.models.elasticity", "fit the hierarchical demand model"),
    ("lift", "pricing.models.lift", "decompose promotional lift and cannibalisation"),
    ("backtest", "pricing.models.backtest", "validate on held out weeks"),
    ("plan", "pricing.report", "optimise the promotional plan"),
    ("figures", "pricing.viz", "render figures"),
]


def run_stage(name: str, module: str) -> float:
    import importlib

    start = time.time()
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    importlib.import_module(module).main()
    elapsed = time.time() - start
    print(f"[{name} finished in {elapsed:.1f}s]")
    return elapsed


def main() -> int:
    names = [s[0] for s in STAGES]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", choices=names, help="start at this stage")
    parser.add_argument("--only", choices=names, help="run only this stage")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args()

    if args.list:
        for name, _, description in STAGES:
            print(f"  {name:<10s} {description}")
        return 0

    load_config().ensure_dirs()

    selected = STAGES
    if args.only:
        selected = [s for s in STAGES if s[0] == args.only]
    elif args.start:
        selected = STAGES[names.index(args.start):]

    total = 0.0
    for name, module, _ in selected:
        total += run_stage(name, module)
    print(f"\nPipeline complete in {total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
