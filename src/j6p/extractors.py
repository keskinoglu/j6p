"""Extractors: source-specific readers that produce a single AnnotatedLazyFrame.

An `Extractor` is the `extractor` of a `LeafETL` (see `j6p.etl`). Extractors
are specific to their *source*, not merely the file format, so they are grouped
by source below.
"""

import json
from pathlib import Path

import polars as pl

from j6p.datatypes import AnnotatedLazyFrame, Extractor

# ---- Schwab ----


def schwab_json_reader(
    path: str | Path, *, key_containing_records: str | None = None
) -> Extractor:
    """Return a reader for a Schwab banking/investment transaction JSON file.

    Schwab transaction exports are a JSON object with the records nested under
    a key (e.g. "BrokerageTransactions" for investment, "PostedTransactions"
    for banking).

    Args:
        path: Path to the Schwab JSON file.
        key_containing_records: The key whose value is the list of transaction
            records. When None, the file must be a bare JSON array.
    """
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        if key_containing_records is None:
            lazy_frame = pl.read_json(p).lazy()
        else:
            with open(p) as f:
                data = json.load(f)
            lazy_frame = pl.DataFrame(data[key_containing_records]).lazy()

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

    return extractor


def schwab_xml_reader(path: str | Path) -> Extractor:
    """Return a reader for a Schwab OFX/XML 1099 file.

    Not yet implemented — polars has no native XML scan. Parsing via stdlib xml
    will be added when 1099 ingestion is built out.
    """
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        raise NotImplementedError(f"schwab_xml_reader not yet implemented for {p}")

    return extractor


# ---- Generic ----


def xls_reader(path: str | Path, **kwargs) -> Extractor:
    """Return a reader for an Excel file (.xls / .xlsx).

    Requires an Excel engine (fastexcel or xlsx2csv for .xlsx; xlrd for .xls).
    Install via: uv add fastexcel   or   uv add xlrd
    """
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        lazy_frame = pl.read_excel(p, **kwargs).lazy()

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

    return extractor


def parquet_reader(path: str | Path) -> Extractor:
    """Return a reader for a previously-written Parquet file."""
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        lazy_frame = pl.scan_parquet(p)

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

    return extractor
