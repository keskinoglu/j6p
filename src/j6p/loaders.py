from pathlib import Path

import polars as pl

from j6p.type_aliases import Writer


def parquet_writer(path: str | Path) -> Writer:
    """Return a writer that persists a frame to a Parquet file."""
    p = Path(path)

    def _write(frame: pl.LazyFrame) -> None:
        frame.collect().write_parquet(p)

    return _write
