import json
from pathlib import Path

import polars as pl

from j6p.type_aliases import Reader


def json_reader(path: str | Path, *, key: str | None = None) -> Reader:
    """Return a reader for a JSON file.

    Args:
        path: Path to the JSON file.
        key: If the JSON is an object (not a bare array), the key whose value is
             the list of records — e.g. "BrokerageTransactions" for Schwab files.
             When None, the file must be a bare JSON array.
    """
    p = Path(path)

    def _read() -> pl.LazyFrame:
        if key is None:
            return pl.read_json(p).lazy()
        with open(p) as f:
            data = json.load(f)
        return pl.DataFrame(data[key]).lazy()

    return _read


def xml_reader(path: str | Path) -> Reader:
    """Return a reader for a Schwab OFX/XML 1099 file.

    Not yet implemented — polars has no native XML scan. Parsing via stdlib xml
    will be added when 1099 ingestion is built out.
    """
    p = Path(path)

    def _read() -> pl.LazyFrame:
        raise NotImplementedError(f"xml_reader not yet implemented for {p}")

    return _read


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
