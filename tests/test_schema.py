"""Column mapping should tolerate cosmetic differences and refuse real ones."""

from __future__ import annotations

import pandas as pd
import pytest

from pricing import schema

WORKBOOK_COLUMNS = [
    "WEEK_END_DATE", "STORE_NUM", "UPC", "UNITS", "VISITS", "HHS",
    "SPEND", "PRICE", "BASE_PRICE", "FEATURE", "DISPLAY", "TPR_ONLY",
]


class TestNormalise:
    @pytest.mark.parametrize(
        "raw", ["BASE_PRICE", "base price", "Base-Price", "  base_price  ", "BasePrice"]
    )
    def test_cosmetic_variants_collapse(self, raw: str) -> None:
        assert schema.normalise(raw) == "baseprice"

    def test_distinct_names_stay_distinct(self) -> None:
        assert schema.normalise("PRICE") != schema.normalise("BASE_PRICE")


class TestMapping:
    def test_maps_the_real_workbook_header(self) -> None:
        mapping = schema.build_mapping(WORKBOOK_COLUMNS)
        assert mapping.is_usable
        assert not mapping.missing_required
        assert not mapping.unmapped_raw

    def test_price_and_base_price_do_not_collide(self) -> None:
        mapping = schema.build_mapping(WORKBOOK_COLUMNS)
        assert mapping.mapping["PRICE"] == schema.PRICE
        assert mapping.mapping["BASE_PRICE"] == schema.BASE_PRICE

    def test_missing_required_column_is_reported(self) -> None:
        columns = [c for c in WORKBOOK_COLUMNS if c != "BASE_PRICE"]
        mapping = schema.build_mapping(columns)
        assert not mapping.is_usable
        assert schema.BASE_PRICE in mapping.missing_required

    def test_unknown_column_is_reported_not_dropped_silently(self) -> None:
        mapping = schema.build_mapping(WORKBOOK_COLUMNS + ["SOME_NEW_FIELD"])
        assert "SOME_NEW_FIELD" in mapping.unmapped_raw
        assert mapping.is_usable

    def test_product_sheet_maps(self) -> None:
        mapping = schema.build_mapping(
            ["UPC", "DESCRIPTION", "MANUFACTURER", "CATEGORY", "SUB_CATEGORY", "PRODUCT_SIZE"],
            schema.PRODUCT_REQUIRED, schema.PRODUCT_OPTIONAL,
        )
        assert mapping.is_usable and not mapping.unmapped_raw

    def test_store_sheet_maps(self) -> None:
        mapping = schema.build_mapping(
            ["STORE_ID", "STORE_NAME", "ADDRESS_CITY_NAME", "ADDRESS_STATE_PROV_CODE",
             "MSA_CODE", "SEG_VALUE_NAME", "PARKING_SPACE_QTY", "SALES_AREA_SIZE_NUM",
             "AVG_WEEKLY_BASKETS"],
            schema.STORE_REQUIRED, schema.STORE_OPTIONAL,
        )
        assert mapping.is_usable and not mapping.unmapped_raw

    def test_apply_mapping_keeps_only_mapped_columns(self) -> None:
        frame = pd.DataFrame({c: [1] for c in WORKBOOK_COLUMNS + ["JUNK"]})
        mapping = schema.build_mapping(frame.columns)
        out = schema.apply_mapping(frame, mapping)
        assert "JUNK" not in out.columns
        assert schema.UNITS in out.columns
