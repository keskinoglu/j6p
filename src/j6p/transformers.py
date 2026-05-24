"""Transformers: factories for the lazy-frame transformations injected into ETLs.

Two kinds live here, split by section:

- **1 -> 1** transformers (`Transformer`): one LazyFrame -> one LazyFrame.
  Used as a `LeafETL.transformer` (see `j6p.etl`).
- **N -> 1** fusion transformers (`Fuser`): a list of LazyFrames -> one
  LazyFrame — i.e. how multiple lazy frames are joined. Used as a
  `NodeETL.transformer`.

A fuser is essentially a transformer; both are kept in this one module, the
section comments making the 1->1 vs N->1 distinction explicit. Each factory
returns a closure named `transformer`.
"""

import polars as pl

from j6p.type_aliases import Fuser, Transformer

# ---- 1 -> 1 transformers ----


def identity() -> Transformer:
    """Return the 1 -> 1 no-op: the lazy frame unchanged."""

    def transformer(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        return lazy_frame

    return transformer


def schwab_brokerage_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for Schwab brokerage exports."""

    def transformer(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        lazy_frame_with_split_dates = _split_date_into_posted_and_as_of(lazy_frame)
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_split_dates,
            columns=["posted_date", "as_of_date"],
            date_format="%m/%d/%Y",
            timezone="America/New_York",
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_localized_dates,
            column_names_left_to_right=["posted_date", "as_of_date"],
        )
        return lazy_frame_with_reordered_columns

    return transformer


def _split_date_into_posted_and_as_of(
    lazy_frame: pl.LazyFrame,
) -> pl.LazyFrame:
    date_parts = pl.col("Date").str.split_exact(" as of ", 1)
    posted_date = date_parts.struct.field("field_0")
    as_of_date = date_parts.struct.field("field_1").fill_null(posted_date)

    lazy_frame_with_split_date_columns = lazy_frame.with_columns(
        posted_date.alias("posted_date"),
        as_of_date.alias("as_of_date"),
    )
    lazy_frame_with_original_date_dropped = lazy_frame_with_split_date_columns.drop(
        "Date"
    )

    return lazy_frame_with_original_date_dropped


# ---- N -> 1 fusion transformers ----


def vertical_concat() -> Fuser:
    """Return an N -> 1 fuser: stack the lazy frames vertically (union); all
    must share one schema."""

    def fuser(lazy_frames: list[pl.LazyFrame]) -> pl.LazyFrame:
        return pl.concat(lazy_frames)

    return fuser


# ---- shared LazyFrame helpers ----


def _localize_dates_in_timezone(
    lazy_frame: pl.LazyFrame,
    columns: list[str],
    date_format: str,
    timezone: str,
) -> pl.LazyFrame:
    parsed_dates = pl.col(*columns).str.to_datetime(date_format)
    localized_dates = parsed_dates.dt.replace_time_zone(timezone)

    lazy_frame_with_localized_dates = lazy_frame.with_columns(localized_dates)

    return lazy_frame_with_localized_dates


def _reorder_columns(
    lazy_frame: pl.LazyFrame,
    column_names_left_to_right: list[str],
) -> pl.LazyFrame:
    leading_columns = [pl.col(name) for name in column_names_left_to_right]
    remaining_columns = pl.all().exclude(*column_names_left_to_right)

    lazy_frame_with_reordered_columns = lazy_frame.select(
        *leading_columns, remaining_columns
    )

    return lazy_frame_with_reordered_columns
