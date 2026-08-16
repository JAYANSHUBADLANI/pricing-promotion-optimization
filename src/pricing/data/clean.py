"""Cleaning rules, applied in a fixed order with a count kept at every step.

Nothing is dropped silently. Each rule records how many rows it removed and why,
and the resulting report is what the write up quotes rather than a remembered
figure. The rules themselves live in config so that changing a threshold does not
mean editing code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pricing import schema
from pricing.config import Config, load_config
from pricing.data.ingest import build_panel


@dataclass
class CleaningReport:
    n_rows_in: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    n_rows_out: int = 0

    def record(self, name: str, removed: int, reason: str, remaining: int) -> None:
        self.steps.append(
            {
                "step": name,
                "rows_removed": int(removed),
                "share_removed": float(removed / self.n_rows_in) if self.n_rows_in else 0.0,
                "rows_remaining": int(remaining),
                "reason": reason,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_rows_removed": self.n_rows_in - self.n_rows_out,
            "share_removed": (
                float((self.n_rows_in - self.n_rows_out) / self.n_rows_in)
                if self.n_rows_in
                else 0.0
            ),
            "steps": self.steps,
        }


def clean(
    cfg: Config | None = None, panel: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, CleaningReport]:
    cfg = cfg or load_config()
    rules = cfg["cleaning"]
    frame = (build_panel(cfg) if panel is None else panel).copy()
    report = CleaningReport(n_rows_in=len(frame))

    if rules.get("drop_missing_price", True):
        mask = frame[schema.PRICE].isna() | frame[schema.BASE_PRICE].isna()
        frame = frame.loc[~mask]
        report.record(
            "drop_missing_price",
            int(mask.sum()),
            "shelf price or base price absent, so no price signal exists for the row",
            len(frame),
        )

    mask = frame[schema.PRICE] < float(rules.get("min_price", 0.01))
    frame = frame.loc[~mask]
    report.record(
        "drop_non_positive_price",
        int(mask.sum()),
        "a price at or below zero cannot enter a log price model",
        len(frame),
    )

    if rules.get("drop_zero_units", True):
        mask = frame[schema.UNITS] <= 0
        frame = frame.loc[~mask]
        report.record(
            "drop_non_positive_units",
            int(mask.sum()),
            "log of units is undefined at zero, and these rows carry zero spend",
            len(frame),
        )

    depth = 1.0 - frame[schema.PRICE] / frame[schema.BASE_PRICE]

    limit = float(rules.get("max_price_above_base", 0.10))
    mask = depth < -limit
    frame = frame.loc[~mask]
    depth = depth.loc[frame.index]
    report.record(
        "drop_price_far_above_base",
        int(mask.sum()),
        (
            f"shelf price more than {limit:.0%} above base price. Small overshoots are "
            "credible as a lagging base price, large ones are not"
        ),
        len(frame),
    )

    max_depth = float(rules.get("max_discount_depth", 0.60))
    mask = depth > max_depth
    frame = frame.loc[~mask]
    report.record(
        "drop_implausible_discount",
        int(mask.sum()),
        (
            f"discount deeper than {max_depth:.0%}, treated as a data error rather "
            "than a promotion"
        ),
        len(frame),
    )

    frame = frame.reset_index(drop=True)
    report.n_rows_out = len(frame)
    return frame, report


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    frame, report = clean(cfg)
    out_path = cfg.path("data", "processed_dir") / "panel_clean.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)

    report_path = cfg.path("reporting", "tables_dir") / "cleaning_report.json"
    report_path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    print(f"{report.n_rows_in:,} rows in")
    for step in report.steps:
        print(f"  {step['step']:<32s} -{step['rows_removed']:>6,} "
              f"({step['share_removed']:.3%})")
    print(f"{report.n_rows_out:,} rows out "
          f"({report.n_rows_out / report.n_rows_in:.2%} retained)")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
