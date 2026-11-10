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
                "Fees & Comm": "Fees & Commissions",
                "AcctgRuleCd": "Accounting Rule Code",
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
            columns=["Posting Date", "As Of Date"],
            date_format="%m/%d/%Y",
            timezone="America/New_York",
        )
        lazy_frame_with_usd_str_to_decimal = _convert_usd_string_to_decimal(
            lazy_frame_with_localized_dates,
            columns=["Price", "Fees & Commissions", "Amount"],
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
            columns=["Accounting Rule Code"],
            integer_type=pl.UInt8,
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_integer_acctg_rule_cd,
            column_names_left_to_right=["Posting Date", "As Of Date", "Account"],
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
        posted_date.alias("Posting Date"),
        as_of_date.alias("As Of Date"),
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
                "Date": "Posting Date",
                "RunningBalance": "Balance",
                "CheckNumber": "Check Number",
            },
        )
        lazy_frame_with_account = _add_new_schwab_account_column_from_filename(
            lazy_frame_with_consistent_column_names, source_path
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_account,
            columns=["Posting Date"],
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
            columns=["Check Number"],
            integer_type=pl.UInt32,
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_integer_check_number,
            column_names_left_to_right=["Posting Date", "Account"],
        )
        lazy_frame_with_positions = _number_entries_within_day(
            lazy_frame_with_reordered_columns, date_column="Posting Date"
        )
        lazy_frame_with_occurrences = _number_repeated_rows(lazy_frame_with_positions)

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
        lazy_frame_with_title_case_column_names = _rename_columns(
            lazy_frame_with_dropped_columns,
            old_to_new_column_names={
                "Created on": "Created On",
                "Finished on": "Finished On",
                "Source fee amount": "Source Fee Amount",
                "Source fee currency": "Source Fee Currency",
                "Source name": "Source Name",
                "Source amount (after fees)": "Source Amount (After Fees)",
                "Source currency": "Source Currency",
                "Target name": "Target Name",
                "Target amount (after fees)": "Target Amount (After Fees)",
                "Target currency": "Target Currency",
                "Exchange rate": "Exchange Rate",
                "Created by": "Created By",
            },
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_title_case_column_names,
            columns=["Created On", "Finished On"],
            date_format="%Y-%m-%d %H:%M:%S",
            timezone="UTC",
        )
        lazy_frame_with_decimal_amounts = _convert_decimal_string_to_decimal(
            lazy_frame_with_localized_dates,
            columns=[
                "Source Fee Amount",
                "Source Amount (After Fees)",
                "Target Amount (After Fees)",
            ],
            total_digits=18,
            digits_after_decimal_point=2,
        )
        lazy_frame_with_float_exchange_rate = _convert_string_to_float(
            lazy_frame_with_decimal_amounts,
            column="Exchange Rate",
        )
        lazy_frame_with_reordered_columns = _reorder_columns(
            lazy_frame_with_float_exchange_rate,
            column_names_left_to_right=["Created On", "Finished On", "Direction"],
        )
        lazy_frame_with_positions = _number_entries_within_day(
            lazy_frame_with_reordered_columns, date_column="Finished On"
        )
        lazy_frame_with_occurrences = _number_repeated_rows(lazy_frame_with_positions)

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_occurrences,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


_ING_DE_OPTIONAL_GERMAN_TO_ENGLISH_COLUMNS = {"Notiz": "Note", "Referenz": "Reference"}
_ING_DE_OPTIONAL_COLUMNS = list(_ING_DE_OPTIONAL_GERMAN_TO_ENGLISH_COLUMNS.values())
_ING_DE_COLUMN_ORDER = [
    "Posting Date",
    "Value Date",
    "Account",
    "Payer/Payee",
    "Transaction Type",
    "Note",
    "Description",
    "Reference",
    "Balance (EUR)",
    "Amount (EUR)",
]


def ing_de_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for ING Germany (ING-DiBa) transaction exports."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        source_path = annotated_lazy_frame.annotations["source_path"]

        lazy_frame_with_dropped_columns = annotated_lazy_frame.lazy_frame.drop(
            ["Währung", "Währung_duplicated_0"]
        )
        lazy_frame_with_all_optional_columns = _add_missing_columns_as_null(
            lazy_frame_with_dropped_columns,
            columns=list(_ING_DE_OPTIONAL_GERMAN_TO_ENGLISH_COLUMNS),
        )
        lazy_frame_with_english_column_names = _rename_columns(
            lazy_frame_with_all_optional_columns,
            old_to_new_column_names={
                "Buchung": "Posting Date",
                "Wertstellungsdatum": "Value Date",
                "Auftraggeber/Empfänger": "Payer/Payee",
                "Buchungstext": "Transaction Type",
                "Verwendungszweck": "Description",
                "Saldo": "Balance (EUR)",
                "Betrag": "Amount (EUR)",
                **_ING_DE_OPTIONAL_GERMAN_TO_ENGLISH_COLUMNS,
            },
        )
        lazy_frame_with_english_transaction_types = _replace_column_values(
            lazy_frame_with_english_column_names,
            column="Transaction Type",
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
            columns=["Posting Date", "Value Date"],
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
            column_names_left_to_right=_ING_DE_COLUMN_ORDER,
        )
        lazy_frame_with_positions = _number_entries_within_day(
            lazy_frame_with_reordered_columns, date_column="Posting Date"
        )
        lazy_frame_with_occurrences = _number_repeated_rows(
            lazy_frame_with_positions,
            columns_to_ignore=_ING_DE_OPTIONAL_COLUMNS,
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
            old_to_new_column_names={"Date": "Posting Date"},
        )
        lazy_frame_with_stripped_descriptions = _strip_surrounding_whitespace(
            lazy_frame_with_consistent_column_names, columns=["Description"]
        )
        lazy_frame_with_account = _add_new_citi_account_column_from_filename(
            lazy_frame_with_stripped_descriptions, source_path
        )
        lazy_frame_with_localized_dates = _localize_dates_in_timezone(
            lazy_frame_with_account,
            columns=["Posting Date"],
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
            column_names_left_to_right=["Posting Date", "Account"],
        )
        lazy_frame_with_positions = _number_entries_within_day(
            lazy_frame_with_reordered_columns, date_column="Posting Date"
        )
        lazy_frame_with_occurrences = _number_repeated_rows(lazy_frame_with_positions)

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


def schwab_banking_known_balances_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for the Schwab rows of a known-balances CSV."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_known_balances = _select_known_balances_of_institution(
            annotated_lazy_frame.lazy_frame,
            institution="Schwab",
            account_columns=["Account"],
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_known_balances,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def ing_de_known_balances_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for the ING rows of a known-balances CSV."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_known_balances = _select_known_balances_of_institution(
            annotated_lazy_frame.lazy_frame,
            institution="ING",
            account_columns=["Account"],
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_known_balances,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def citi_banking_known_balances_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for the Citi rows of a known-balances CSV."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_known_balances = _select_known_balances_of_institution(
            annotated_lazy_frame.lazy_frame,
            institution="Citi",
            account_columns=["Account"],
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_known_balances,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def wise_known_balances_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for the Wise rows of a known-balances CSV."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_known_balances = _select_known_balances_of_institution(
            annotated_lazy_frame.lazy_frame,
            institution="Wise",
            account_columns=["Currency"],
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_known_balances,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def _select_known_balances_of_institution(
    lazy_frame: pl.LazyFrame, institution: str, account_columns: list[str]
) -> pl.LazyFrame:
    lazy_frame_with_institution_rows_only = _keep_rows_where_column_is_in(
        lazy_frame, column="Institution", values=[institution]
    )
    lazy_frame_with_consistent_column_names = _rename_columns(
        lazy_frame_with_institution_rows_only,
        old_to_new_column_names={
            "Date": _KNOWN_BALANCE_DATE_COLUMN,
            "Balance": _KNOWN_BALANCE_COLUMN,
        },
    )
    lazy_frame_with_dates = _parse_dates_in_iso_us_or_german_format(
        lazy_frame_with_consistent_column_names, columns=[_KNOWN_BALANCE_DATE_COLUMN]
    )
    lazy_frame_with_decimal_balances = _convert_decimal_string_to_decimal(
        lazy_frame_with_dates,
        columns=[_KNOWN_BALANCE_COLUMN],
        total_digits=18,
        digits_after_decimal_point=2,
    )
    return lazy_frame_with_decimal_balances.select(
        *account_columns, _KNOWN_BALANCE_DATE_COLUMN, _KNOWN_BALANCE_COLUMN
    )


def internal_counterparties_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for an internal-counterparties TXT file."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_stripped_names = _strip_surrounding_whitespace(
            annotated_lazy_frame.lazy_frame, columns=["Counterparty"]
        )
        lazy_frame_without_blank_names = lazy_frame_with_stripped_names.filter(
            pl.col("Counterparty").is_not_null() & (pl.col("Counterparty") != "")
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_without_blank_names,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


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
            column="Posting Date",
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
            column="Posting Date",
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
            column="Created On",
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

    Duplicate rows are merged on every column except Note and Reference, which
    are export options; each keeps its first non-null value across duplicates.
    """

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        deduped_lazy_frame = _merge_duplicate_rows_ignoring(
            stacked_lazy_frame, columns_to_ignore=_ING_DE_OPTIONAL_COLUMNS
        )
        deduped_lazy_frame_without_occurrences = deduped_lazy_frame.drop(
            _OCCURRENCE_COLUMN
        )
        sorted_lazy_frame = _sort_by_column(
            deduped_lazy_frame_without_occurrences,
            column="Posting Date",
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
            column="Posting Date",
            descending=True,
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=sorted_lazy_frame, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def schwab_banking_balance_fuser() -> Fuser:
    """Return the fuser that balances Schwab checking against its known balances."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (schwab_banking, schwab_banking_known_balances) = annotated_lazy_frames

        lazy_frame_with_balances = _add_balances_with_adjustments(
            schwab_banking.lazy_frame,
            known_balances=schwab_banking_known_balances.lazy_frame,
            amount=pl.col("Deposit (USD)").fill_null(0)
            - pl.col("Withdrawal (USD)").fill_null(0),
            amount_column="Amount (USD)",
            date_column="Posting Date",
            account_columns=["Account"],
            reported_balance_column="Balance (USD)",
            balance_column="Balance (USD)",
            description_column="Description",
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_balances, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def ing_de_balance_fuser() -> Fuser:
    """Return the fuser that balances ING Germany against its known balances."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (ing_de, ing_de_known_balances) = annotated_lazy_frames

        lazy_frame_with_balances = _add_balances_with_adjustments(
            ing_de.lazy_frame,
            known_balances=ing_de_known_balances.lazy_frame,
            amount=pl.col("Amount (EUR)"),
            amount_column="Amount (EUR)",
            date_column="Posting Date",
            account_columns=["Account"],
            reported_balance_column="Balance (EUR)",
            balance_column="Balance (EUR)",
            description_column="Description",
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_balances, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def citi_banking_balance_fuser() -> Fuser:
    """Return the fuser that balances Citi checking against its known balances."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (citi_banking, citi_banking_known_balances) = annotated_lazy_frames

        lazy_frame_with_balances = _add_balances_with_adjustments(
            citi_banking.lazy_frame,
            known_balances=citi_banking_known_balances.lazy_frame,
            amount=pl.coalesce("Debit (USD)", "Credit (USD)"),
            amount_column="Amount (USD)",
            date_column="Posting Date",
            account_columns=["Account"],
            reported_balance_column=None,
            balance_column="Balance (USD)",
            description_column="Description",
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_balances, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def wise_balance_fuser() -> Fuser:
    """Return the fuser that splits Wise into currency legs and balances each one."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (wise, wise_known_balances) = annotated_lazy_frames

        lazy_frame_without_jar_moves = _drop_wise_moves_between_jars(wise.lazy_frame)
        lazy_frame_as_currency_legs = _split_wise_rows_into_currency_legs(
            lazy_frame_without_jar_moves
        )
        lazy_frame_with_balances = _add_balances_with_adjustments(
            lazy_frame_as_currency_legs,
            known_balances=wise_known_balances.lazy_frame,
            amount=pl.col("Amount"),
            amount_column="Amount",
            date_column="Finished On",
            account_columns=["Currency"],
            reported_balance_column=None,
            balance_column="Balance",
            description_column="Reference",
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_balances, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def _drop_wise_moves_between_jars(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    is_move_between_jars = (pl.col("Direction") == "NEUTRAL") & (
        pl.col("Source Currency") == pl.col("Target Currency")
    )
    return lazy_frame.filter(~is_move_between_jars)


def _split_wise_rows_into_currency_legs(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """NOTE: a conversion (NEUTRAL) yields two legs; the source fee leaves with the
    source amount, since Wise deducts it before converting."""

    money_leaving_source_currency = lazy_frame.filter(
        pl.col("Direction").is_in(["OUT", "NEUTRAL"])
    ).with_columns(
        pl.col("Source Currency").alias("Currency"),
        (
            -(
                pl.col("Source Amount (After Fees)")
                + pl.col("Source Fee Amount").fill_null(0)
            )
        ).alias("Amount"),
    )
    money_arriving_in_target_currency = lazy_frame.filter(
        pl.col("Direction").is_in(["IN", "NEUTRAL"])
    ).with_columns(
        pl.col("Target Currency").alias("Currency"),
        pl.col("Target Amount (After Fees)").alias("Amount"),
    )

    lazy_frame_as_currency_legs = _stack_vertically(
        [money_leaving_source_currency, money_arriving_in_target_currency]
    )
    return lazy_frame_as_currency_legs


def schwab_banking_ledger_entries_fuser() -> Fuser:
    """Return the fuser that turns balanced Schwab checking into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (schwab_banking_with_balance,) = annotated_lazy_frames

        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            schwab_banking_with_balance.lazy_frame,
            posting_date=pl.col("Posting Date"),
            institution=pl.lit("Schwab"),
            account=pl.col("Account"),
            currency=pl.lit("USD"),
            amount=pl.col("Amount (USD)"),
            balance=pl.col("Balance (USD)"),
            counterparty=pl.col("Description"),
            description=pl.col("Description"),
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_as_ledger_entries, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def ing_de_ledger_entries_fuser() -> Fuser:
    """Return the fuser that turns balanced ING Germany into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (ing_de_with_balance,) = annotated_lazy_frames

        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            ing_de_with_balance.lazy_frame,
            posting_date=pl.col("Posting Date"),
            institution=pl.lit("ING"),
            account=pl.col("Account"),
            currency=pl.lit("EUR"),
            amount=pl.col("Amount (EUR)"),
            balance=pl.col("Balance (EUR)"),
            counterparty=pl.col("Payer/Payee"),
            description=pl.col("Description"),
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_as_ledger_entries, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def citi_banking_ledger_entries_fuser() -> Fuser:
    """Return the fuser that turns balanced Citi checking into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (citi_banking_with_balance,) = annotated_lazy_frames

        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            citi_banking_with_balance.lazy_frame,
            posting_date=pl.col("Posting Date"),
            institution=pl.lit("Citi"),
            account=pl.col("Account"),
            currency=pl.lit("USD"),
            amount=pl.col("Amount (USD)"),
            balance=pl.col("Balance (USD)"),
            counterparty=pl.col("Description"),
            description=pl.col("Description"),
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_as_ledger_entries, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def wise_ledger_entries_fuser() -> Fuser:
    """Return the fuser that turns balanced Wise currency legs into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (wise_with_balance,) = annotated_lazy_frames

        counterparty_of_the_leg = (
            pl.when(pl.col("Direction") == "IN")
            .then(pl.col("Source Name"))
            .otherwise(pl.col("Target Name"))
        )
        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            wise_with_balance.lazy_frame,
            posting_date=pl.col("Finished On"),
            institution=pl.lit("Wise"),
            account=pl.lit(None),
            currency=pl.col("Currency"),
            amount=pl.col("Amount"),
            balance=pl.col("Balance"),
            counterparty=counterparty_of_the_leg,
            description=pl.col("Reference"),
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_as_ledger_entries, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def _select_ledger_entry_columns(
    lazy_frame: pl.LazyFrame,
    *,
    posting_date: pl.Expr,
    institution: pl.Expr,
    account: pl.Expr,
    currency: pl.Expr,
    amount: pl.Expr,
    balance: pl.Expr,
    counterparty: pl.Expr,
    description: pl.Expr,
) -> pl.LazyFrame:
    """NOTE: every bank is cast to the same dtypes, and dates to UTC, so that
    the ledger can stack them. Adjustments have no counterparty."""

    is_adjustment = pl.col(_ENTRY_TYPE_COLUMN) == _ADJUSTMENT_ENTRY_TYPE
    lazy_frame_as_ledger_entries = lazy_frame.select(
        posting_date.dt.convert_time_zone("UTC").alias("Posting Date"),
        institution.cast(pl.String).alias("Institution"),
        account.cast(pl.String).alias("Account"),
        currency.cast(pl.String).alias("Currency"),
        amount.cast(_MONEY).alias("Amount"),
        balance.cast(_MONEY).alias("Account Balance"),
        pl.when(is_adjustment)
        .then(None)
        .otherwise(counterparty)
        .cast(pl.String)
        .alias("Counterparty"),
        description.cast(pl.String).alias("Description"),
        pl.col(_ENTRY_TYPE_COLUMN),
        pl.col(_ENTRY_NUMBER_COLUMN),
    )
    return lazy_frame_as_ledger_entries


def ledger_entries_fuser() -> Fuser:
    """Return the fuser that stacks every bank's ledger entries, newest first."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        sorted_lazy_frame = _sort_by_columns(
            stacked_lazy_frame,
            columns=[
                "Posting Date",
                "Institution",
                "Account",
                "Currency",
                _ENTRY_NUMBER_COLUMN,
            ],
            descending=True,
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=sorted_lazy_frame, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def ledger_fuser() -> Fuser:
    """Return the fuser that books ledger entries against internal counterparties."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (ledger_entries, internal_counterparties) = annotated_lazy_frames

        internal_counterparty_names = (
            internal_counterparties.lazy_frame.collect()
            .get_column("Counterparty")
            .to_list()
        )
        lazy_frame_with_transfers = _classify_transfers(
            ledger_entries.lazy_frame,
            internal_counterparties=internal_counterparty_names,
        )
        lazy_frame_with_combined_balances = _add_combined_balance_per_currency(
            lazy_frame_with_transfers
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_combined_balances,
            annotations=merged_annotations,
        )
        return annotated_lazy_frame

    return fuser


def _classify_transfers(
    lazy_frame: pl.LazyFrame, internal_counterparties: list[str]
) -> pl.LazyFrame:
    is_adjustment = pl.col(_ENTRY_TYPE_COLUMN) == _ADJUSTMENT_ENTRY_TYPE
    is_internal_counterparty = pl.col("Counterparty").str.contains_any(
        internal_counterparties, ascii_case_insensitive=True
    )
    transfer = (
        pl.when(is_adjustment)
        .then(pl.lit(_ADJUSTMENT_ENTRY_TYPE))
        .when(is_internal_counterparty)
        .then(pl.lit("Internal"))
        .otherwise(pl.lit("External"))
    )
    return lazy_frame.with_columns(transfer.alias("Transfer"))


def _add_combined_balance_per_currency(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """NOTE: expects newest first. Every account opens with an adjustment from 0,
    so the running total of all amounts is the sum of the account balances."""

    running_total_per_currency = pl.col("Amount").cum_sum(reverse=True).over("Currency")
    lazy_frame_with_combined_balances = lazy_frame.with_columns(
        running_total_per_currency.cast(_MONEY).alias("Combined Balance")
    ).drop(_ENTRY_TYPE_COLUMN, _ENTRY_NUMBER_COLUMN)

    lazy_frame_with_reordered_columns = _reorder_columns(
        lazy_frame_with_combined_balances,
        column_names_left_to_right=[
            "Posting Date",
            "Institution",
            "Account",
            "Currency",
            "Amount",
            "Account Balance",
            "Combined Balance",
            "Transfer",
        ],
    )
    return lazy_frame_with_reordered_columns


# ---- shared LazyFrame helpers ----


_OCCURRENCE_COLUMN = "_occurrence"
_POSITION_IN_DAY_COLUMN = "_position_in_day"
_KNOWN_BALANCE_DATE_COLUMN = "Known Balance Date"
_KNOWN_BALANCE_COLUMN = "Known Balance"
_ENTRY_TYPE_COLUMN = "Entry Type"
_ENTRY_NUMBER_COLUMN = "Entry Number"
_TRANSACTION_ENTRY_TYPE = "Transaction"
_ADJUSTMENT_ENTRY_TYPE = "Adjustment"
_ADJUSTMENT_DESCRIPTION = "Automatic adjustment"
_MONEY = pl.Decimal(precision=18, scale=2)


def _number_repeated_rows(
    lazy_frame: pl.LazyFrame, columns_to_ignore: list[str] | None = None
) -> pl.LazyFrame:
    """NOTE: adds ``_occurrence`` (0, 1, ...) so identical rows stay distinct.
    Rows are identical when they match on every column not in columns_to_ignore."""

    ignored = {_POSITION_IN_DAY_COLUMN, *(columns_to_ignore or [])}
    column_names = [
        name for name in lazy_frame.collect_schema().names() if name not in ignored
    ]
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


def _parse_dates_in_iso_us_or_german_format(
    lazy_frame: pl.LazyFrame, columns: list[str]
) -> pl.LazyFrame:
    """NOTE: accepts 2026-11-05, 11/5/2026 and 5.11.2026, so a spreadsheet
    editor's rewrite of the date still parses."""

    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"]
    parsed_dates = [
        pl.coalesce(
            pl.col(column).str.to_date(date_format, strict=False)
            for date_format in date_formats
        ).alias(column)
        for column in columns
    ]
    return lazy_frame.with_columns(parsed_dates)


def _number_entries_within_day(
    lazy_frame: pl.LazyFrame, date_column: str
) -> pl.LazyFrame:
    """NOTE: exports list newest first, so each day's entries are numbered from
    its oldest (0) upward. An export taken mid-day lacks only the day's later
    entries, so it numbers the entries it has the same as a complete export."""

    day = pl.col(date_column).dt.date()
    position_from_oldest = (pl.len() - 1 - pl.int_range(pl.len())).over(day)
    return lazy_frame.with_columns(
        position_from_oldest.cast(pl.UInt32).alias(_POSITION_IN_DAY_COLUMN)
    )


def _add_balances_with_adjustments(
    lazy_frame: pl.LazyFrame,
    *,
    known_balances: pl.LazyFrame,
    amount: pl.Expr,
    amount_column: str,
    date_column: str,
    account_columns: list[str],
    reported_balance_column: str | None,
    balance_column: str,
    description_column: str,
) -> pl.LazyFrame:
    """NOTE: newer known balances are trusted over older ones.

    Known balances are end-of-day: each day's last bank-reported balance, and the
    known-balances CSV (which wins on the same day). Balances are counted forward
    and backward from the next known balance on or after each entry, or from the
    newest one after it. Where an older known balance disagrees, an adjustment is
    booked right after it; the account opens with an adjustment from 0. An
    account with no known balance is taken to end at 0.
    """

    account_and_day = [*account_columns, "_day"]
    date_dtype = lazy_frame.collect_schema()[date_column]

    transactions = lazy_frame.with_columns(
        amount.fill_null(0).cast(_MONEY).alias(amount_column),
        pl.lit(_TRANSACTION_ENTRY_TYPE).alias(_ENTRY_TYPE_COLUMN),
        pl.col(date_column).dt.date().alias("_day"),
        pl.lit(1, dtype=pl.UInt8).alias("_rank_in_day"),
    ).cache()
    end_of_day_known_balances = _collect_end_of_day_known_balances(
        transactions,
        known_balances=known_balances,
        account_columns=account_columns,
        reported_balance_column=reported_balance_column,
    )
    known_balance_markers = end_of_day_known_balances.with_columns(
        pl.lit(0, dtype=_MONEY).alias(amount_column),
        pl.lit(2, dtype=pl.UInt8).alias("_rank_in_day"),
    )
    chronological_order = [
        *account_and_day,
        "_rank_in_day",
        date_column,
        _POSITION_IN_DAY_COLUMN,
    ]
    events = pl.concat(
        [transactions, known_balance_markers], how="diagonal_relaxed"
    ).sort(chronological_order, nulls_last=True)

    is_marker = pl.col("_rank_in_day") == 2
    running_total = pl.col(amount_column).cum_sum().over(account_columns)
    events_with_totals = events.with_columns(
        running_total.alias("_running_total"),
        (~is_marker).cast(pl.UInt32).cum_sum().over(account_columns).alias("_seen"),
    )
    total_at_marker = pl.when(is_marker).then(pl.col("_running_total"))
    anchor_known_balance = (
        pl.col("_known").backward_fill().forward_fill().over(account_columns)
    )
    anchor_total = total_at_marker.backward_fill().forward_fill().over(account_columns)
    account_total = pl.col("_running_total").last().over(account_columns)
    next_known_balance = (
        pl.col("_known").shift(-1).backward_fill().over(account_columns)
    )
    next_total = total_at_marker.shift(-1).backward_fill().over(account_columns)
    events_with_balances = events_with_totals.with_columns(
        (
            anchor_known_balance.fill_null(0)
            + pl.col("_running_total")
            - anchor_total.fill_null(account_total)
        )
        .cast(_MONEY)
        .alias(balance_column),
        (next_known_balance + pl.col("_running_total") - next_total - pl.col("_known"))
        .cast(_MONEY)
        .alias("_adjustment"),
    ).cache()

    balanced_transactions = events_with_balances.filter(~is_marker).cache()
    adjustments_after_known_balances = events_with_balances.filter(
        is_marker & (pl.col("_seen") > 0) & (pl.col("_adjustment") != 0)
    ).select(
        *account_and_day,
        "_rank_in_day",
        pl.col("_day")
        .dt.combine(pl.time(23, 59, 59), time_unit=date_dtype.time_unit)
        .dt.replace_time_zone(date_dtype.time_zone)
        .alias(date_column),
        pl.col("_adjustment").alias(amount_column),
        (pl.col("_known") + pl.col("_adjustment")).cast(_MONEY).alias(balance_column),
    )
    opening_adjustments = (
        balanced_transactions.filter(pl.col("_seen") == 1)
        .group_by(account_columns)
        .agg(pl.all().first())
        .with_columns(
            (pl.col(balance_column) - pl.col(amount_column))
            .cast(_MONEY)
            .alias(amount_column),
            pl.lit(0, dtype=pl.UInt8).alias("_rank_in_day"),
        )
        .filter(pl.col(amount_column) != 0)
        .select(
            *account_and_day,
            "_rank_in_day",
            date_column,
            amount_column,
            pl.col(amount_column).alias(balance_column),
        )
    )
    adjustments = pl.concat(
        [opening_adjustments, adjustments_after_known_balances]
    ).with_columns(
        pl.lit(_ADJUSTMENT_ENTRY_TYPE).alias(_ENTRY_TYPE_COLUMN),
        pl.lit(_ADJUSTMENT_DESCRIPTION).alias(description_column),
    )

    all_entries = pl.concat(
        [balanced_transactions, adjustments], how="diagonal_relaxed"
    ).sort(chronological_order, nulls_last=True)
    entry_number = pl.int_range(1, pl.len() + 1).over(account_columns)
    numbered_entries = all_entries.with_columns(
        entry_number.cast(pl.UInt32).alias(_ENTRY_NUMBER_COLUMN)
    ).drop(
        "_day",
        "_rank_in_day",
        "_known",
        "_running_total",
        "_seen",
        "_adjustment",
        _POSITION_IN_DAY_COLUMN,
    )
    trailing_columns = [
        amount_column,
        balance_column,
        _ENTRY_TYPE_COLUMN,
        _ENTRY_NUMBER_COLUMN,
    ]
    entries_with_trailing_columns = numbered_entries.select(
        pl.all().exclude(trailing_columns), *trailing_columns
    )
    newest_first = _sort_by_columns(
        entries_with_trailing_columns,
        columns=[date_column, *account_columns, _ENTRY_NUMBER_COLUMN],
        descending=True,
    )
    return newest_first


def _collect_end_of_day_known_balances(
    transactions: pl.LazyFrame,
    *,
    known_balances: pl.LazyFrame,
    account_columns: list[str],
    reported_balance_column: str | None,
) -> pl.LazyFrame:
    """NOTE: a known-balances CSV row wins over the bank's balance on the same day.
    Only accounts that have transactions are kept."""

    account_and_day = [*account_columns, "_day"]
    from_csv = known_balances.select(
        *account_columns,
        pl.col(_KNOWN_BALANCE_DATE_COLUMN).alias("_day"),
        pl.col(_KNOWN_BALANCE_COLUMN).alias("_known"),
        pl.lit(0, dtype=pl.UInt8).alias("_source_priority"),
    )
    sources = [from_csv]
    if reported_balance_column is not None:
        last_entry_of_day = pl.col(reported_balance_column).sort_by(
            _POSITION_IN_DAY_COLUMN
        )
        from_bank = transactions.group_by(account_and_day).agg(
            last_entry_of_day.last().alias("_known"),
            pl.lit(1, dtype=pl.UInt8).alias("_source_priority"),
        )
        sources.append(from_bank)

    end_of_day_known_balances = (
        pl.concat(sources, how="vertical_relaxed")
        .group_by(account_and_day)
        .agg(pl.col("_known").sort_by("_source_priority").first().cast(_MONEY))
    )
    accounts_with_transactions = transactions.select(account_columns).unique()
    return end_of_day_known_balances.join(
        accounts_with_transactions, on=account_columns, how="semi"
    )


def _add_missing_columns_as_null(
    lazy_frame: pl.LazyFrame, columns: list[str]
) -> pl.LazyFrame:
    present_columns = lazy_frame.collect_schema().names()
    missing_columns_as_null = [
        pl.lit(None, dtype=pl.String).alias(column)
        for column in columns
        if column not in present_columns
    ]
    return lazy_frame.with_columns(missing_columns_as_null)


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
    """NOTE: rows match on every column except their position within the day."""

    column_names = lazy_frame.collect_schema().names()
    matching_columns = [
        name for name in column_names if name != _POSITION_IN_DAY_COLUMN
    ]
    return lazy_frame.unique(subset=matching_columns, keep="first", maintain_order=True)


def _merge_duplicate_rows_ignoring(
    lazy_frame: pl.LazyFrame, columns_to_ignore: list[str]
) -> pl.LazyFrame:
    column_names = lazy_frame.collect_schema().names()
    ignored_columns = [*columns_to_ignore, _POSITION_IN_DAY_COLUMN]
    key_columns = [name for name in column_names if name not in ignored_columns]
    first_non_null_of_ignored_columns = [
        pl.col(name).drop_nulls().first() for name in ignored_columns
    ]
    merged_lazy_frame = lazy_frame.group_by(key_columns, maintain_order=True).agg(
        first_non_null_of_ignored_columns
    )
    return merged_lazy_frame.select(column_names)


def _sort_by_column(
    lazy_frame: pl.LazyFrame,
    column: str,
    descending: bool,
) -> pl.LazyFrame:
    return lazy_frame.sort(column, descending=descending, maintain_order=True)


def _sort_by_columns(
    lazy_frame: pl.LazyFrame,
    columns: list[str],
    descending: bool,
) -> pl.LazyFrame:
    return lazy_frame.sort(columns, descending=descending, nulls_last=True)
