from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class AnnotatedLazyFrame:
    """A polars LazyFrame paired with stage-contributed annotations (metadata).

    This is the currency of the ETL pipeline: every stage receives one, may
    read its annotations, and returns a new one whose `lazy_frame` is the frame
    the next stage operates on.
    """

    lazy_frame: pl.LazyFrame
    annotations: Mapping[str, Any]


@dataclass(frozen=True)
class AnnotatedDataFrame:
    """A polars DataFrame paired with stage-contributed annotations (metadata).

    This is the materialized output of the ETL pipeline: `ETL.collect()` returns
    one, carrying the same annotations alongside an eagerly-evaluated frame.
    """

    data_frame: pl.DataFrame
    annotations: Mapping[str, Any]


Extractor = Callable[[], AnnotatedLazyFrame]
Transformer = Callable[[AnnotatedLazyFrame], AnnotatedLazyFrame]
Fuser = Callable[[list[AnnotatedLazyFrame]], AnnotatedLazyFrame]
Loader = Callable[[AnnotatedLazyFrame], AnnotatedLazyFrame]
