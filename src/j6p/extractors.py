"""Extractors: source-specific readers that produce a single LazyFrame.

A reader is the `extractor` of a `LeafETL` (see `j6p.blocks`). Readers are
specific to their *source*, not merely the file format, so readers are grouped
by source below.
"""

import json
from pathlib import Path

import polars as pl

from j6p.type_aliases import Reader

# ---- Schwab ----


def schwab_json_reader(path: str | Path, *, key: str | None = None) -> Reader:
    """Return a reader for a Schwab banking/investment transaction JSON file.

    Schwab transaction exports are a JSON object with the records nested under
    a key (e.g. "BrokerageTransactions" for investment, "PostedTransactions"
    for banking).

    Args:
        path: Path to the Schwab JSON file.
        key: The key whose value is the list of transaction records. When
             None, the file must be a bare JSON array.
    """
    p = Path(path)

    def _read() -> pl.LazyFrame:
        if key is None:
            return pl.read_json(p).lazy()
        with open(p) as f:
            data = json.load(f)
        return pl.DataFrame(data[key]).lazy()

    return _read


def schwab_xml_reader(path: str | Path) -> Reader:
    """Return a reader for a Schwab OFX/XML 1099 file.

    Not yet implemented — polars has no native XML scan. Parsing via stdlib xml
    will be added when 1099 ingestion is built out.
    """
    p = Path(path)

    def _read() -> pl.LazyFrame:
        raise NotImplementedError(f"schwab_xml_reader not yet implemented for {p}")

    return _read


# ---- Generic ----


def xls_reader(path: str | Path, **kwargs) -> Reader:
    """Return a reader for an Excel file (.xls / .xlsx).

    Requires an Excel engine (fastexcel or xlsx2csv for .xlsx; xlrd for .xls).
    Install via: uv add fastexcel   or   uv add xlrd
    """
    p = Path(path)

    def _read() -> pl.LazyFrame:
        return pl.read_excel(p, **kwargs).lazy()

    return _read


def parquet_reader(path: str | Path) -> Reader:
    """Return a reader for a previously-written Parquet file."""
    p = Path(path)

    def _read() -> pl.LazyFrame:
        return pl.scan_parquet(p)

    return _read
