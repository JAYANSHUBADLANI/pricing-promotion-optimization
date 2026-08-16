"""Canonical column names and tolerant mapping onto them.

The workbook ships with upper case names and a title banner above the header
row. Rather than hard coding those, every raw name is normalised to lower case
alphanumerics and matched against a list of accepted aliases. Anything that does
not match is reported instead of being dropped silently, so a change in the
source file surfaces as a message rather than as a wrong number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

# Transaction grain
WEEK = "week_end_date"
STORE_ID = "store_id"
UPC = "upc"
UNITS = "units"
VISITS = "visits"
HOUSEHOLDS = "households"
SPEND = "spend"
PRICE = "price"
BASE_PRICE = "base_price"
FEATURE = "feature"
DISPLAY = "display"
TPR_ONLY = "tpr_only"

# Product lookup
DESCRIPTION = "description"
MANUFACTURER = "manufacturer"
CATEGORY = "category"
SUB_CATEGORY = "sub_category"
PRODUCT_SIZE = "product_size"

# Store lookup
STORE_NAME = "store_name"
CITY = "city"
STATE = "state"
MSA_CODE = "msa_code"
STORE_SEGMENT = "store_segment"
PARKING_SPACES = "parking_spaces"
SALES_AREA = "sales_area_sqft"
AVG_WEEKLY_BASKETS = "avg_weekly_baskets"

ALIASES: Mapping[str, tuple[str, ...]] = {
    WEEK: ("weekenddate", "week", "weekending", "weekendingdate"),
    STORE_ID: ("storenum", "storeid", "store", "storenumber"),
    UPC: ("upc", "productid", "itemcode"),
    UNITS: ("units", "unitsold", "unitssold", "qty", "quantity"),
    VISITS: ("visits", "baskets"),
    HOUSEHOLDS: ("hhs", "households", "hh"),
    SPEND: ("spend", "sales", "salesrevenue", "revenue", "dollars"),
    PRICE: ("price", "shelfprice", "actualprice", "unitprice"),
    BASE_PRICE: ("baseprice", "regularprice", "listprice"),
    FEATURE: ("feature", "featured", "circular"),
    DISPLAY: ("display", "displayed", "instoredisplay"),
    TPR_ONLY: ("tpronly", "tpr", "temporarypricereduction"),
    DESCRIPTION: ("description", "productdescription", "itemdescription"),
    MANUFACTURER: ("manufacturer", "brand", "vendor"),
    CATEGORY: ("category", "productcategory"),
    SUB_CATEGORY: ("subcategory", "productsubcategory"),
    PRODUCT_SIZE: ("productsize", "size", "packsize"),
    STORE_NAME: ("storename",),
    CITY: ("addresscityname", "city"),
    STATE: ("addressstateprovcode", "state", "stateprovcode"),
    MSA_CODE: ("msacode", "msa"),
    STORE_SEGMENT: ("segvaluename", "segment", "storesegment", "storeappeal"),
    PARKING_SPACES: ("parkingspaceqty", "parkingspaces"),
    SALES_AREA: ("salesareasizenum", "salesarea", "squarefootage", "sqft"),
    AVG_WEEKLY_BASKETS: ("avgweeklybaskets", "averageweeklybaskets"),
}

TRANSACTION_REQUIRED = (WEEK, STORE_ID, UPC, UNITS, SPEND, PRICE, BASE_PRICE)
TRANSACTION_OPTIONAL = (VISITS, HOUSEHOLDS, FEATURE, DISPLAY, TPR_ONLY)
PRODUCT_REQUIRED = (UPC, CATEGORY, SUB_CATEGORY)
PRODUCT_OPTIONAL = (DESCRIPTION, MANUFACTURER, PRODUCT_SIZE)
STORE_REQUIRED = (STORE_ID,)
STORE_OPTIONAL = (
    STORE_NAME, CITY, STATE, MSA_CODE, STORE_SEGMENT,
    PARKING_SPACES, SALES_AREA, AVG_WEEKLY_BASKETS,
)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise(name: str) -> str:
    """Reduce a raw column name to lower case alphanumerics."""
    return _NON_ALNUM.sub("", str(name).strip().lower())


_LOOKUP: dict[str, str] = {}
for canonical, aliases in ALIASES.items():
    _LOOKUP[normalise(canonical)] = canonical
    for alias in aliases:
        _LOOKUP.setdefault(normalise(alias), canonical)


@dataclass(frozen=True)
class SchemaMapping:
    mapping: dict[str, str]
    unmapped_raw: tuple[str, ...]
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return not self.missing_required


def build_mapping(
    columns: Iterable[str],
    required: Iterable[str] = TRANSACTION_REQUIRED,
    optional: Iterable[str] = TRANSACTION_OPTIONAL,
) -> SchemaMapping:
    """Match raw column names onto canonical ones and report what is missing."""
    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    for column in columns:
        canonical = _LOOKUP.get(normalise(column))
        if canonical is None:
            unmapped.append(str(column))
        else:
            mapping[str(column)] = canonical
    found = set(mapping.values())
    return SchemaMapping(
        mapping=mapping,
        unmapped_raw=tuple(unmapped),
        missing_required=tuple(c for c in required if c not in found),
        missing_optional=tuple(c for c in optional if c not in found),
    )


def apply_mapping(frame: pd.DataFrame, mapping: SchemaMapping) -> pd.DataFrame:
    """Rename to canonical names and keep only the columns that mapped."""
    renamed = frame.rename(columns=mapping.mapping)
    keep = [c for c in renamed.columns if c in set(mapping.mapping.values())]
    return renamed.loc[:, keep]
