"""Loaders: the final ETL stage — return the frame, optionally persist it.

A loader is the `loader` of a `LeafETL`/`NodeETL` (see `j6p.etl`). Every loader
returns the frame so the pipeline continues; `parquet_writer` also persists it
as a side effect, `return_only` does not.

Each factory returns a closure named `loader` — uniform with `j6p.extractors`
/ `j6p.transformers`.
"""

from pathlib import Path

import polars as pl

from j6p.type_aliases import Loader


def return_only() -> Loader:
    """Return the no-op loader: persist nowhere; only return the frame."""

    def loader(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        return lazy_frame

    return loader


def parquet_writer(path: str | Path) -> Loader:
    """Return a loader that writes the frame to a Parquet file, then returns it."""
    p = Path(path)

    def loader(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        lazy_frame.collect().write_parquet(p)
        return lazy_frame

    return loader
