"""Loaders: the final ETL stage — return the annotated frame, optionally persist it.

A loader is the `loader` of a `LeafETL`/`NodeETL` (see `j6p.etl`). Every loader
returns the annotated frame so the pipeline continues; `parquet_writer` also persists
the inner lazy frame as a side effect, `return_only` does not.

Each factory returns a closure named `loader` — uniform with `j6p.extractors`
/ `j6p.transformers`.
"""

from pathlib import Path

from j6p.datatypes import AnnotatedLazyFrame, Loader


def return_only() -> Loader:
    """Return the no-op loader: persist nowhere; only return the annotated frame."""

    def loader(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        return annotated_lazy_frame

    return loader


def parquet_writer(path: str | Path) -> Loader:
    """Return a loader that writes the frame to a Parquet file, then returns
    the annotated frame.
    """
    p = Path(path)

    def loader(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        annotated_lazy_frame.lazy_frame.collect().write_parquet(p)
        return annotated_lazy_frame

    return loader
