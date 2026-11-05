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

import re
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

        lazy_frame_with_spelled_out_column_names = _rename_columns(
            annotated_lazy_frame.lazy_frame,
            old_to_new_column_names={
                "Fees & Comm": "Fees & commissions",
                "AcctgRuleCd": "Accounting rule code",
            },
        )
        lazy_frame_with_account = _add_new_schwab_account_column_from_filename(
            lazy_frame_with_spelled_out_column_names, source_path
        )
        lazy_frame_with_split_dates = _split_date_into_posted_and_as_of(
            lazy_frame_with_account
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_split_dates,
            columns=["Posting date", "As of date"],
            date_format="%m/%d/%Y",
            timezone="America/New_York",
        )
        lazy_frame_with_usd_str_to_decimal = _convert_usd_string_to_decimal(
            lazy_frame_with_localized_dates,
            columns=["Price", "Fees & commissions", "Amount"],
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
            columns=["Accounting rule code"],
            integer_type=pl.UInt8,
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_integer_acctg_rule_cd,
            column_names_left_to_right=["Posting date", "As of date", "Account"],
        )
        lazy_frame_with_occurrences = _number_repeated_rows(
            lazy_frame_with_reordered_columns
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_occurrences,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def _split_date_into_posted_and_as_of(
    lazy_frame: pl.LazyFrame,
) -> pl.LazyFrame:
    date_parts = pl.col("Date").str.split_exact(" as of ", 1)
    posted_date = date_parts.struct.field("field_0")
    as_of_date = date_parts.struct.field("field_1").fill_null(posted_date)

    lazy_frame_with_split_date_columns = lazy_frame.with_columns(
        posted_date.alias("Posting date"),
        as_of_date.alias("As of date"),
    )
    lazy_frame_with_original_date_dropped = lazy_frame_with_split_date_columns.drop(
        "Date"
    )

    return lazy_frame_with_original_date_dropped


def schwab_banking_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for Schwab banking (checking) exports."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        source_path = annotated_lazy_frame.annotations["source_path"]

        lazy_frame_with_consistent_column_names = _rename_columns(
            annotated_lazy_frame.lazy_frame,
            old_to_new_column_names={
                "Date": "Posting date",
                "RunningBalance": "Balance",
                "CheckNumber": "Check number",
            },
        )
        lazy_frame_with_account = _add_new_schwab_account_column_from_filename(
            lazy_frame_with_consistent_column_names, source_path
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_account,
            columns=["Posting date"],
            date_format="%m/%d/%Y",
            timezone="America/New_York",
        )
        lazy_frame_with_usd_str_to_decimal = _convert_usd_string_to_decimal(
            lazy_frame_with_localized_dates,
            columns=["Balance", "Withdrawal", "Deposit"],
            total_digits=18,
            digits_after_decimal_point=2,
        )
        lazy_frame_with_integer_check_number = _convert_string_to_unsigned_integer(
            lazy_frame_with_usd_str_to_decimal,
            columns=["Check number"],
            integer_type=pl.UInt32,
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_integer_check_number,
            column_names_left_to_right=["Posting date", "Account"],
        )
        lazy_frame_with_occurrences = _number_repeated_rows(
            lazy_frame_with_reordered_columns
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_occurrences,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def _add_new_schwab_account_column_from_filename(
    lazy_frame: pl.LazyFrame, source_path: Path
) -> pl.LazyFrame:
    account = re.match(r"(.*?XXX\d+)", source_path.name).group(1)
    lazy_frame_with_account = lazy_frame.with_columns(pl.lit(account).alias("Account"))
    return lazy_frame_with_account


def wise_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for Wise transaction-history exports."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_settled_rows_only = _keep_rows_where_column_is_in(
            annotated_lazy_frame.lazy_frame,
            column="Status",
            values=["COMPLETED", "REFUNDED"],
        )
        lazy_frame_with_dropped_columns = lazy_frame_with_settled_rows_only.drop(
            ["Batch", "Note", "Target fee amount", "Target fee currency"]
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_dropped_columns,
            columns=["Created on", "Finished on"],
            date_format="%Y-%m-%d %H:%M:%S",
            timezone="UTC",
        )
        lazy_frame_with_decimal_amounts = _convert_decimal_string_to_decimal(
            lazy_frame_with_localized_dates,
            columns=[
                "Source fee amount",
                "Source amount (after fees)",
                "Target amount (after fees)",
            ],
            total_digits=18,
            digits_after_decimal_point=2,
        )
        lazy_frame_with_float_exchange_rate = _convert_string_to_float(
            lazy_frame_with_decimal_amounts,
            column="Exchange rate",
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_float_exchange_rate,
            column_names_left_to_right=["Created on", "Finished on", "Direction"],
        )
        lazy_frame_with_occurrences = _number_repeated_rows(
            lazy_frame_with_reordered_columns
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_occurrences,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def ing_de_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for ING Germany (ING-DiBa) transaction exports."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        source_path = annotated_lazy_frame.annotations["source_path"]

        lazy_frame_with_dropped_columns = annotated_lazy_frame.lazy_frame.drop(
            ["Währung", "Währung_duplicated_0"]
        )
        lazy_frame_with_english_column_names = _rename_columns(
            lazy_frame_with_dropped_columns,
            old_to_new_column_names={
                "Buchung": "Posting date",
                "Wertstellungsdatum": "Value date",
                "Auftraggeber/Empfänger": "Payer/Payee",
                "Buchungstext": "Transaction type",
                "Verwendungszweck": "Description",
                "Referenz": "Reference",
                "Saldo": "Balance (EUR)",
                "Betrag": "Amount (EUR)",
            },
        )
        lazy_frame_with_english_transaction_types = _replace_column_values(
            lazy_frame_with_english_column_names,
            column="Transaction type",
            old_to_new_values={
                "Lastschrift": "Direct debit",
                "Gutschrift": "Credit",
                "Gutschrift Echtzeitüberweisung": "Instant transfer credit",
                "Echtzeitüberweisung": "Instant transfer",
                "Dauerauftrag / Terminueberweisung": "Standing/Scheduled transfer",
                "Gehalt/Rente": "Salary/Pension",
            },
        )
        lazy_frame_with_account = _add_new_ing_de_account_column_from_filename(
            lazy_frame_with_english_transaction_types, source_path
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_account,
            columns=["Posting date", "Value date"],
            date_format="%d.%m.%Y",
            timezone="Europe/Berlin",
        )
        lazy_frame_with_decimal_amounts = (
            _convert_european_decimal_string_to_us_decimal(
                lazy_frame_with_localized_dates,
                columns=["Balance (EUR)", "Amount (EUR)"],
                total_digits=18,
                digits_after_decimal_point=2,
            )
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_decimal_amounts,
            column_names_left_to_right=["Posting date", "Value date", "Account"],
        )
        lazy_frame_with_occurrences = _number_repeated_rows(
            lazy_frame_with_reordered_columns
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_occurrences,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def _add_new_ing_de_account_column_from_filename(
    lazy_frame: pl.LazyFrame, source_path: Path
) -> pl.LazyFrame:
    """NOTE: the IBAN is masked to its last 3 digits, matching Schwab's ``XXX784``."""

    iban = re.match(r"Umsatzanzeige_([A-Z]{2}\d+)_", source_path.name).group(1)
    masked_account = f"XXX{iban[-3:]}"
    lazy_frame_with_account = lazy_frame.with_columns(
        pl.lit(masked_account).alias("Account")
    )
    return lazy_frame_with_account


def citi_banking_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for Citi banking (checking) exports."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        source_path = annotated_lazy_frame.annotations["source_path"]

        lazy_frame_with_cleared_rows_only = _keep_rows_where_column_is_in(
            annotated_lazy_frame.lazy_frame, column="Status", values=["Cleared"]
        )
        lazy_frame_with_dropped_columns = lazy_frame_with_cleared_rows_only.drop(
            ["Status"]
        )
        lazy_frame_with_consistent_column_names = _rename_columns(
            lazy_frame_with_dropped_columns,
            old_to_new_column_names={"Date": "Posting date"},
        )
        lazy_frame_with_stripped_descriptions = _strip_surrounding_whitespace(
            lazy_frame_with_consistent_column_names, columns=["Description"]
        )
        lazy_frame_with_account = _add_new_citi_account_column_from_filename(
            lazy_frame_with_stripped_descriptions, source_path
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_account,
            columns=["Posting date"],
            date_format="%m-%d-%Y",
            timezone="America/New_York",
        )
        lazy_frame_with_usd_str_to_decimal = _convert_usd_string_to_decimal(
            lazy_frame_with_localized_dates,
            columns=["Debit", "Credit"],
            total_digits=18,
            digits_after_decimal_point=2,
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_usd_str_to_decimal,
            column_names_left_to_right=["Posting date", "Account"],
        )
        lazy_frame_with_occurrences = _number_repeated_rows(
            lazy_frame_with_reordered_columns
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_occurrences,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def _add_new_citi_account_column_from_filename(
    lazy_frame: pl.LazyFrame, source_path: Path
) -> pl.LazyFrame:
    account_digits = re.match(r"[A-Z]+_(\d+)_", source_path.name).group(1)
    masked_account = f"XXX{account_digits[-3:]}"
    lazy_frame_with_account = lazy_frame.with_columns(
        pl.lit(masked_account).alias("Account")
    )
    return lazy_frame_with_account


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
        deduped_lazy_frame_without_occurrences = deduped_lazy_frame.drop(
            _OCCURRENCE_COLUMN
        )
        sorted_lazy_frame = _sort_by_column(
            deduped_lazy_frame_without_occurrences,
            column="Posting date",
            descending=True,
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


def schwab_banking_fuser() -> Fuser:
    """Return the N -> 1 fuser for Schwab banking (checking) exports."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        deduped_lazy_frame = _drop_duplicate_rows(stacked_lazy_frame)
        deduped_lazy_frame_without_occurrences = deduped_lazy_frame.drop(
            _OCCURRENCE_COLUMN
        )
        sorted_lazy_frame = _sort_by_column(
            deduped_lazy_frame_without_occurrences,
            column="Posting date",
            descending=True,
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=sorted_lazy_frame, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def wise_fuser() -> Fuser:
    """Return the N -> 1 fuser for Wise transaction-history exports.

    Duplicate rows are dropped on *all* columns, not on ID alone: Wise reuses an
    ID across the currency legs of one card transaction.
    """

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        deduped_lazy_frame = _drop_duplicate_rows(stacked_lazy_frame)
        deduped_lazy_frame_without_occurrences = deduped_lazy_frame.drop(
            _OCCURRENCE_COLUMN
        )
        sorted_lazy_frame = _sort_by_column(
            deduped_lazy_frame_without_occurrences,
            column="Created on",
            descending=True,
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=sorted_lazy_frame, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def ing_de_fuser() -> Fuser:
    """Return the N -> 1 fuser for ING Germany (ING-DiBa) transaction exports.

    Duplicate rows are dropped on *all* columns.
    """

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        deduped_lazy_frame = _drop_duplicate_rows(stacked_lazy_frame)
        deduped_lazy_frame_without_occurrences = deduped_lazy_frame.drop(
            _OCCURRENCE_COLUMN
        )
        sorted_lazy_frame = _sort_by_column(
            deduped_lazy_frame_without_occurrences,
            column="Posting date",
            descending=True,
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=sorted_lazy_frame, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def citi_banking_fuser() -> Fuser:
    """Return the N -> 1 fuser for Citi banking (checking) exports."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        deduped_lazy_frame = _drop_duplicate_rows(stacked_lazy_frame)
        deduped_lazy_frame_without_occurrences = deduped_lazy_frame.drop(
            _OCCURRENCE_COLUMN
        )
        sorted_lazy_frame = _sort_by_column(
            deduped_lazy_frame_without_occurrences,
            column="Posting date",
            descending=True,
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=sorted_lazy_frame, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


# ---- shared LazyFrame helpers ----


_OCCURRENCE_COLUMN = "_occurrence"


def _number_repeated_rows(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """NOTE: adds ``_occurrence`` (0, 1, ...) so identical rows stay distinct."""

    column_names = lazy_frame.collect_schema().names()
    occurrence_of_identical_row = pl.int_range(pl.len()).over(column_names)

    lazy_frame_with_occurrences = lazy_frame.with_columns(
        occurrence_of_identical_row.alias(_OCCURRENCE_COLUMN)
    )

    return lazy_frame_with_occurrences


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

    expression_to_strip_usd_formatting = (
        pl.col(*columns)
        .str.replace_all(",", "", literal=True)
        .str.replace("$", "", literal=True)
    )

    expression_to_cast_as_decimal = expression_to_strip_usd_formatting.cast(
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


def _convert_decimal_string_to_decimal(
    lazy_frame: pl.LazyFrame,
    columns: list[str],
    total_digits: int,
    digits_after_decimal_point: int,
) -> pl.LazyFrame:
    """Cast plain decimal strings (no currency symbols or thousands separators)."""

    expression_to_cast_as_decimal = pl.col(*columns).cast(
        pl.Decimal(precision=total_digits, scale=digits_after_decimal_point),
        strict=False,
    )
    lazy_frame_with_decimal_amounts = lazy_frame.with_columns(
        expression_to_cast_as_decimal
    )

    return lazy_frame_with_decimal_amounts


def _convert_european_decimal_string_to_us_decimal(
    lazy_frame: pl.LazyFrame,
    columns: list[str],
    total_digits: int,
    digits_after_decimal_point: int,
) -> pl.LazyFrame:
    """Cast European-formatted strings (``.`` thousands, ``,`` decimal) to Decimal."""

    expression_to_convert_european_to_us_format = (
        pl.col(*columns)
        .str.replace_all(".", "", literal=True)
        .str.replace(",", ".", literal=True)
    )

    expression_to_cast_as_decimal = expression_to_convert_european_to_us_format.cast(
        pl.Decimal(precision=total_digits, scale=digits_after_decimal_point),
        strict=False,
    )
    lazy_frame_with_decimal_amounts = lazy_frame.with_columns(
        expression_to_cast_as_decimal
    )

    return lazy_frame_with_decimal_amounts


def _convert_string_to_float(
    lazy_frame: pl.LazyFrame,
    column: str,
) -> pl.LazyFrame:
    expression_to_cast_string_to_float = pl.col(column).cast(pl.Float64, strict=False)
    lazy_frame_with_float_column = lazy_frame.with_columns(
        expression_to_cast_string_to_float
    )

    return lazy_frame_with_float_column


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


def _keep_rows_where_column_is_in(
    lazy_frame: pl.LazyFrame,
    column: str,
    values: list[str],
) -> pl.LazyFrame:
    return lazy_frame.filter(pl.col(column).is_in(values))


def _strip_surrounding_whitespace(
    lazy_frame: pl.LazyFrame,
    columns: list[str],
) -> pl.LazyFrame:
    expression_to_strip_whitespace = pl.col(*columns).str.strip_chars()
    lazy_frame_with_stripped_columns = lazy_frame.with_columns(
        expression_to_strip_whitespace
    )

    return lazy_frame_with_stripped_columns


def _rename_columns(
    lazy_frame: pl.LazyFrame,
    old_to_new_column_names: Mapping[str, str],
) -> pl.LazyFrame:
    return lazy_frame.rename(old_to_new_column_names)


def _replace_column_values(
    lazy_frame: pl.LazyFrame,
    column: str,
    old_to_new_values: Mapping[str, str],
) -> pl.LazyFrame:
    """NOTE: values missing from ``old_to_new_values`` are kept unchanged."""

    expression_to_replace_values = pl.col(column).replace(old_to_new_values)
    lazy_frame_with_replaced_values = lazy_frame.with_columns(
        expression_to_replace_values
    )

    return lazy_frame_with_replaced_values


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
    return lazy_frame.unique(keep="first", maintain_order=True)


def _sort_by_column(
    lazy_frame: pl.LazyFrame,
    column: str,
    descending: bool,
) -> pl.LazyFrame:
    return lazy_frame.sort(column, descending=descending, maintain_order=True)
