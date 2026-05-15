from collections.abc import Callable
from pathlib import Path

import polars as pl

_Writer = Callable[[pl.LazyFrame], None]


def parquet_writer(path: str | Path) -> _Writer:
    """Return a writer that persists a frame to a Parquet file."""
    p = Path(path)

    def _write(frame: pl.LazyFrame) -> None:
        frame.collect().write_parquet(p)

    return _write
