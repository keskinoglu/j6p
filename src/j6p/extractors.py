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


_SCHWAB_BANKING_JSON_COLUMN_ORDER = [
    "CheckNumber",
    "Description",
    "Date",
    "RunningBalance",
    "Withdrawal",
    "Deposit",
    "Type",
]


def schwab_banking_csv_reader(path: str | Path) -> Extractor:
    """Return a Schwab banking CSV reader shaped like the PostedTransactions JSON."""
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        lazy_frame = (
            pl.read_csv(p, infer_schema=False)
            .lazy()
            .filter(pl.col("Status") == "Posted")
            .with_columns(
                pl.when(pl.col("CheckNumber") != "").then(pl.col("CheckNumber"))
            )
            .select(_SCHWAB_BANKING_JSON_COLUMN_ORDER)
        )

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


# ---- Wise ----


def wise_csv_reader(path: str | Path) -> Extractor:
    """Return a reader for a Wise transaction-history CSV file.

    The Wise export is a single flat CSV whose header row names the columns.
    All values are read as strings (no type inference) so the data stays
    exactly as exported. Empty fields become null.

    Args:
        path: Path to the Wise CSV file.
    """
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        lazy_frame = pl.read_csv(p, infer_schema=False).lazy()

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

    return extractor


# ---- ING DE ----


_ING_DE_HEADER_PREFIX = "Buchung;"


def ing_de_csv_reader(path: str | Path) -> Extractor:
    """Return a reader for an ING Germany (ING-DiBa) Umsatzanzeige CSV file.

    The export is a semicolon-separated, ISO-8859 CSV whose header row follows
    a preamble of varying length.

    Args:
        path: Path to the ING CSV file.
    """
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        lazy_frame = pl.read_csv(
            p,
            separator=";",
            infer_schema=False,
            skip_rows=_count_lines_before_ing_de_header(p),
            encoding="latin-1",
        ).lazy()

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

    return extractor


def _count_lines_before_ing_de_header(path: Path) -> int:
    with open(path, encoding="latin-1") as f:
        for line_index, line in enumerate(f):
            if line.startswith(_ING_DE_HEADER_PREFIX):
                return line_index
    raise ValueError(f"No ING header row starting with {_ING_DE_HEADER_PREFIX!r}")


# ---- Citi ----


def citi_banking_csv_reader(path: str | Path) -> Extractor:
    """Return a reader for a Citi banking (checking) transaction CSV file.

    Every data row ends with a trailing comma that the header lacks, so the
    resulting empty extra field is truncated.

    Args:
        path: Path to the Citi CSV file.
    """
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        lazy_frame = pl.read_csv(
            p, infer_schema=False, truncate_ragged_lines=True
        ).lazy()

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

    return extractor


# ---- Known balances ----


def known_balances_csv_reader(path: str | Path) -> Extractor:
    """Return a reader for a known-balances CSV (see templates/known_balances.csv)."""
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        lazy_frame = pl.scan_csv(p, infer_schema=False, comment_prefix="#")

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

    return extractor


# ---- Internal counterparties ----


def internal_counterparties_txt_reader(path: str | Path) -> Extractor:
    """Return a reader for an internal-counterparties TXT file (see templates/)."""
    p = Path(path)

    def extractor() -> AnnotatedLazyFrame:
        with open(p, encoding="utf-8") as f:
            lines = f.read().splitlines()
        names = [line for line in lines if not line.startswith("#")]
        lazy_frame = pl.LazyFrame(
            {"Counterparty": names}, schema={"Counterparty": pl.String}
        )

        annotations = {"source_path": p}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame, annotations=annotations
        )
        return annotated_lazy_frame

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
