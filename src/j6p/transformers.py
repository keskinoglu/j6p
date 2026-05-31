"""Transformers: factories for annotated-lazy-frame transformations injected into ETLs.

Two kinds live here, split by section:

- **1 -> 1** transformers (`Transformer`): AnnotatedLazyFrame -> AnnotatedLazyFrame.
  Used as a `LeafETL.transformer` (see `j6p.etl`).
- **N -> 1** fusion transformers (`Fuser`): a list of AnnotatedLazyFrames -> one
  AnnotatedLazyFrame — i.e. how multiple annotated lazy frames are joined. Used as a
  `NodeETL.transformer`.

A fuser is essentially a transformer; both are kept in this one module, the
section comments making the 1->1 vs N->1 distinction explicit. Each factory
returns a closure named `transformer`.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

from j6p.datatypes import AnnotatedLazyFrame, Fuser, Transformer

# ---- 1 -> 1 transformers ----


def identity() -> Transformer:
    """Return the 1 -> 1 no-op: the annotated lazy frame unchanged."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        return annotated_lazy_frame

    return transformer


def schwab_brokerage_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for Schwab brokerage exports."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        source_path = annotated_lazy_frame.annotations["source_path"]

        lazy_frame_with_account = _add_new_schwab_account_column_from_filename(
            annotated_lazy_frame.lazy_frame, source_path
        )
        lazy_frame_with_split_dates = _split_date_into_posted_and_as_of(
            lazy_frame_with_account
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_split_dates,
            columns=["posted_date", "as_of_date"],
            date_format="%m/%d/%Y",
            timezone="America/New_York",
        )
        lazy_frame_with_usd_str_to_decimal = _convert_usd_string_to_decimal(
            lazy_frame_with_localized_dates,
            columns=["Price", "Fees & Comm", "Amount"],
            total_digits=18,
            digits_after_decimal_point=4,
        )
        lazy_frame_with_integer_quantity = _convert_string_to_unsigned_integer(
            lazy_frame_with_usd_str_to_decimal,
            columns=["Quantity"],
            integer_type=pl.UInt16,
        )
        lazy_frame_with_integer_acctg_rule_cd = _convert_string_to_unsigned_integer(
            lazy_frame_with_integer_quantity,
            columns=["AcctgRuleCd"],
            integer_type=pl.UInt8,
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_integer_acctg_rule_cd,
            column_names_left_to_right=["posted_date", "as_of_date", "account"],
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_reordered_columns,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def _add_new_schwab_account_column_from_filename(
    lazy_frame: pl.LazyFrame, source_path: Path
) -> pl.LazyFrame:
    account = source_path.name.split("_Transactions_", 1)[0]
    lazy_frame_with_account = lazy_frame.with_columns(pl.lit(account).alias("account"))
    return lazy_frame_with_account


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


def schwab_brokerage_fuser() -> Fuser:
    """Return the N -> 1 fuser for Schwab brokerage exports."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        deduped_lazy_frame = _drop_duplicate_rows(stacked_lazy_frame)
        sorted_lazy_frame = _sort_by_column(
            deduped_lazy_frame, column="posted_date", descending=True
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=sorted_lazy_frame, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def _merge_annotations_from_child_frames(
    annotated_lazy_frames: list[AnnotatedLazyFrame],
) -> Mapping[str, Any]:
    child_annotations = [
        annotated_lazy_frame.annotations
        for annotated_lazy_frame in annotated_lazy_frames
    ]
    return {"children": child_annotations}


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


def _convert_usd_string_to_decimal(
    lazy_frame: pl.LazyFrame,
    columns: list[str],
    total_digits: int,
    digits_after_decimal_point: int,
) -> pl.LazyFrame:
    """NOTE: each converted column is also renamed with a ``' (USD)'`` suffix."""

    expression_to_strip_dollar_sign = pl.col(*columns).str.replace(
        "$", "", literal=True
    )

    expression_to_cast_as_decimal = expression_to_strip_dollar_sign.cast(
        pl.Decimal(precision=total_digits, scale=digits_after_decimal_point),
        strict=False,
    )
    lazy_frame_with_decimal_amounts = lazy_frame.with_columns(
        expression_to_cast_as_decimal
    )

    new_column_names_with_usd_suffix = {name: f"{name} (USD)" for name in columns}
    lazy_frame_with_usd_suffixed_columns = lazy_frame_with_decimal_amounts.rename(
        new_column_names_with_usd_suffix
    )

    return lazy_frame_with_usd_suffixed_columns


def _convert_string_to_unsigned_integer(
    lazy_frame: pl.LazyFrame,
    columns: list[str],
    integer_type: pl.DataType,
) -> pl.LazyFrame:
    expression_to_cast_strings_to_integers = pl.col(*columns).cast(
        integer_type, strict=False
    )
    lazy_frame_with_unsigned_integer_columns = lazy_frame.with_columns(
        expression_to_cast_strings_to_integers
    )

    return lazy_frame_with_unsigned_integer_columns


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


def _stack_vertically(lazy_frames: list[pl.LazyFrame]) -> pl.LazyFrame:
    return pl.concat(lazy_frames)


def _drop_duplicate_rows(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    return lazy_frame.unique()


def _sort_by_column(
    lazy_frame: pl.LazyFrame,
    column: str,
    descending: bool,
) -> pl.LazyFrame:
    return lazy_frame.sort(column, descending=descending)
