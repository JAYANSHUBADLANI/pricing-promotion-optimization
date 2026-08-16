"""Read the dunnhumby workbook into tidy frames.

The source is a single Excel workbook with three data sheets and a glossary.
Each sheet carries a title banner above the real header row, which is why the
header offset lives in config rather than being guessed at read time.

Reading 500k plus rows out of Excel is slow with the default engine, so the
calamine reader is preferred when installed and the result is cached to Parquet.
Later stages read the cache, which keeps a full pipeline run from paying the
Excel parsing cost more than once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pricing import schema
from pricing.config import Config, load_config

WORKBOOK_HELP = """
Could not find the source workbook.

Download 'Breakfast at the Frat' from https://www.dunnhumby.com/source-files/
and place the .xlsx file in data/raw/.

The pipeline only cares that the file is there. It does not matter whether it
arrived by browser, by script, or by hand.
""".strip()

_CACHE_NAMES = {
    "transactions": "transactions.parquet",
    "stores": "stores.parquet",
    "products": "products.parquet",
}


def _excel_engine() -> str | None:
    """Prefer calamine when available, since it is far faster on large sheets."""
    try:
        import python_calamine  # noqa: F401
    except ImportError:
        return None
    return "calamine"


LOOKUP_CONFLICT_MARKER = "CONFLICTING SOURCE VALUES"


def dedupe_lookup(
    frame: pd.DataFrame, key: str
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Collapse a lookup table to one row per key and report attribute conflicts.

    The store lookup ships with two stores listed twice, each pair agreeing on
    every attribute except segment, where one row says MAINSTREAM and the other
    says UPSCALE. Silently keeping the first row would bury a real contradiction
    in the source and would make the choice depend on sheet ordering. Instead the
    conflicting field is marked as unresolved and the conflict is returned so it
    can be reported. Fields that agree are kept as they are.
    """
    duplicated = frame[key].duplicated(keep=False)
    if not duplicated.any():
        return frame.reset_index(drop=True), []

    conflicts: list[dict[str, object]] = []
    rows: list[pd.Series] = []
    for key_value, group in frame.groupby(key, sort=False):
        row = group.iloc[0].copy()
        if len(group) > 1:
            for column in group.columns:
                if column == key:
                    continue
                values = pd.unique(group[column].dropna())
                if len(values) > 1:
                    conflicts.append(
                        {
                            "key": key,
                            "key_value": key_value,
                            "column": column,
                            "values": [str(v) for v in values],
                        }
                    )
                    row[column] = LOOKUP_CONFLICT_MARKER if group[column].dtype == object else pd.NA
                elif len(values) == 1:
                    row[column] = values[0]
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True), conflicts


def find_workbook(cfg: Config | None = None) -> Path:
    """Locate the source workbook, matching loosely so a renamed file still works."""
    cfg = cfg or load_config()
    raw_dir = cfg.path("data", "raw_dir")
    expected = raw_dir / str(cfg["data"]["workbook_name"])
    if expected.exists():
        return expected

    candidates = sorted(
        p for p in raw_dir.glob("*.xls*") if not p.name.startswith("~$")
    )
    if not candidates:
        raise FileNotFoundError(WORKBOOK_HELP)
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def _read_sheet(
    path: Path,
    sheet: str,
    header_row: int,
    required,
    optional,
) -> tuple[pd.DataFrame, schema.SchemaMapping]:
    frame = pd.read_excel(path, sheet_name=sheet, header=header_row, engine=_excel_engine())
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    mapping = schema.build_mapping(frame.columns, required, optional)
    if not mapping.is_usable:
        raise ValueError(
            f"Sheet {sheet!r} is missing required columns: "
            f"{', '.join(mapping.missing_required)}. "
            f"Columns present: {', '.join(str(c) for c in frame.columns)}"
        )
    return schema.apply_mapping(frame, mapping), mapping


def load_raw(
    cfg: Config | None = None, use_cache: bool = True, refresh: bool = False
) -> dict[str, pd.DataFrame]:
    """Return the three raw tables, reading from the Parquet cache when possible."""
    cfg = cfg or load_config()
    interim = cfg.path("data", "interim_dir")
    interim.mkdir(parents=True, exist_ok=True)
    cache_paths = {k: interim / v for k, v in _CACHE_NAMES.items()}

    if use_cache and not refresh and all(p.exists() for p in cache_paths.values()):
        return {k: pd.read_parquet(p) for k, p in cache_paths.items()}

    path = find_workbook(cfg)
    header_row = int(cfg["data"]["header_row"])
    sheets = cfg["data"]["sheets"]

    transactions, _ = _read_sheet(
        path, sheets["transactions"], header_row,
        schema.TRANSACTION_REQUIRED, schema.TRANSACTION_OPTIONAL,
    )
    stores, _ = _read_sheet(
        path, sheets["stores"], header_row,
        schema.STORE_REQUIRED, schema.STORE_OPTIONAL,
    )
    products, _ = _read_sheet(
        path, sheets["products"], header_row,
        schema.PRODUCT_REQUIRED, schema.PRODUCT_OPTIONAL,
    )

    transactions[schema.WEEK] = pd.to_datetime(transactions[schema.WEEK], errors="coerce")
    for frame in (transactions, products):
        frame[schema.UPC] = frame[schema.UPC].astype("int64")
    for frame in (transactions, stores):
        frame[schema.STORE_ID] = frame[schema.STORE_ID].astype("int64")

    stores, store_conflicts = dedupe_lookup(stores, schema.STORE_ID)
    products, product_conflicts = dedupe_lookup(products, schema.UPC)
    if store_conflicts or product_conflicts:
        interim.joinpath("lookup_conflicts.json").write_text(
            json.dumps(
                {"stores": store_conflicts, "products": product_conflicts}, indent=2
            ),
            encoding="utf-8",
        )

    tables = {"transactions": transactions, "stores": stores, "products": products}
    if use_cache:
        for key, frame in tables.items():
            frame.to_parquet(cache_paths[key], index=False)
    return tables


def build_panel(cfg: Config | None = None, refresh: bool = False) -> pd.DataFrame:
    """Join transactions to the product and store lookups."""
    tables = load_raw(cfg, refresh=refresh)
    panel = tables["transactions"].merge(
        tables["products"], on=schema.UPC, how="left", validate="many_to_one"
    )
    panel = panel.merge(
        tables["stores"], on=schema.STORE_ID, how="left", validate="many_to_one"
    )
    return panel.sort_values([schema.STORE_ID, schema.UPC, schema.WEEK]).reset_index(drop=True)


def main() -> None:
    cfg = load_config()
    cfg.ensure_dirs()
    tables = load_raw(cfg, refresh=True)
    for name, frame in tables.items():
        print(f"{name:14s} {frame.shape[0]:>8,} rows  {frame.shape[1]:>2} cols")


if __name__ == "__main__":
    main()
