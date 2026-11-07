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


def citi_banking_known_balance_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for the Citi rows of a known-balances CSV."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_citi_rows_only = _keep_rows_where_column_is_in(
            annotated_lazy_frame.lazy_frame, column="Institution", values=["Citi"]
        )
        lazy_frame_with_known_balances = _convert_known_balances_to_typed_columns(
            lazy_frame_with_citi_rows_only
        )
        lazy_frame_with_citi_columns = lazy_frame_with_known_balances.select(
            pl.col("Account"),
            pl.col("Known balance date"),
            pl.col("Known balance").alias("Known balance (USD)"),
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_citi_columns,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def wise_known_balances_transformer() -> Transformer:
    """Return the 1 -> 1 transformer for the Wise rows of a known-balances CSV."""

    def transformer(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_wise_rows_only = _keep_rows_where_column_is_in(
            annotated_lazy_frame.lazy_frame, column="Institution", values=["Wise"]
        )
        lazy_frame_with_known_balances = _convert_known_balances_to_typed_columns(
            lazy_frame_with_wise_rows_only
        )
        lazy_frame_with_wise_columns = lazy_frame_with_known_balances.select(
            pl.col("Currency"),
            pl.col("Known balance date"),
            pl.col("Known balance"),
        )

        transformed_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_wise_columns,
            annotations=annotated_lazy_frame.annotations,
        )
        return transformed_annotated_lazy_frame

    return transformer


def _convert_known_balances_to_typed_columns(
    lazy_frame: pl.LazyFrame,
) -> pl.LazyFrame:
    lazy_frame_with_consistent_column_names = _rename_columns(
        lazy_frame,
        old_to_new_column_names={
            "Date": "Known balance date",
            "Balance": "Known balance",
        },
    )
    lazy_frame_with_dates = _parse_dates_in_iso_us_or_german_format(
        lazy_frame_with_consistent_column_names, columns=["Known balance date"]
    )
    lazy_frame_with_decimal_balances = _convert_decimal_string_to_decimal(
        lazy_frame_with_dates,
        columns=["Known balance"],
        total_digits=18,
        digits_after_decimal_point=2,
    )
    return lazy_frame_with_decimal_balances


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


def citi_banking_balance_fuser() -> Fuser:
    """Return the fuser that adds a running balance to Citi from its known balance."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (citi_banking, citi_known_balance) = annotated_lazy_frames

        lazy_frame_with_balances = _add_running_balance_from_known_balance(
            citi_banking.lazy_frame,
            known_balances=citi_known_balance.lazy_frame,
            amount=pl.coalesce("Debit (USD)", "Credit (USD)"),
            date_column="Posting date",
            account_columns=["Account"],
            known_balance_column="Known balance (USD)",
            balance_column="Balance (USD)",
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_balances, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def wise_balance_fuser() -> Fuser:
    """Return the fuser that splits Wise into currency legs with running balances."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (wise, wise_known_balances) = annotated_lazy_frames

        lazy_frame_without_jar_moves = _drop_wise_moves_between_jars(wise.lazy_frame)
        lazy_frame_as_currency_legs = _split_wise_rows_into_currency_legs(
            lazy_frame_without_jar_moves
        )
        sorted_lazy_frame = _sort_by_column(
            lazy_frame_as_currency_legs, column="Finished on", descending=True
        )
        lazy_frame_with_balances = _add_running_balance_from_known_balance(
            sorted_lazy_frame,
            known_balances=wise_known_balances.lazy_frame,
            amount=pl.col("Amount"),
            date_column="Finished on",
            account_columns=["Currency"],
            known_balance_column="Known balance",
            balance_column="Balance",
        )

        merged_annotations = _merge_annotations_from_child_frames(annotated_lazy_frames)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_balances, annotations=merged_annotations
        )
        return annotated_lazy_frame

    return fuser


def _drop_wise_moves_between_jars(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    is_move_between_jars = (pl.col("Direction") == "NEUTRAL") & (
        pl.col("Source currency") == pl.col("Target currency")
    )
    return lazy_frame.filter(~is_move_between_jars)


def _split_wise_rows_into_currency_legs(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """NOTE: a conversion (NEUTRAL) yields two legs; the source fee leaves with the
    source amount, since Wise deducts it before converting."""

    money_leaving_source_currency = lazy_frame.filter(
        pl.col("Direction").is_in(["OUT", "NEUTRAL"])
    ).with_columns(
        pl.col("Source currency").alias("Currency"),
        (
            -(
                pl.col("Source amount (after fees)")
                + pl.col("Source fee amount").fill_null(0)
            )
        ).alias("Amount"),
    )
    money_arriving_in_target_currency = lazy_frame.filter(
        pl.col("Direction").is_in(["IN", "NEUTRAL"])
    ).with_columns(
        pl.col("Target currency").alias("Currency"),
        pl.col("Target amount (after fees)").alias("Amount"),
    )

    lazy_frame_as_currency_legs = _stack_vertically(
        [money_leaving_source_currency, money_arriving_in_target_currency]
    )
    return lazy_frame_as_currency_legs


def schwab_banking_ledger_entries_fuser() -> Fuser:
    """Return the fuser that turns fused Schwab checking into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (schwab_banking,) = annotated_lazy_frames

        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            schwab_banking.lazy_frame,
            posting_date=pl.col("Posting date"),
            institution=pl.lit("Schwab"),
            account=pl.col("Account"),
            currency=pl.lit("USD"),
            amount=pl.col("Deposit (USD)").fill_null(0)
            - pl.col("Withdrawal (USD)").fill_null(0),
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
    """Return the fuser that turns fused ING Germany exports into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (ing_de,) = annotated_lazy_frames

        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            ing_de.lazy_frame,
            posting_date=pl.col("Posting date"),
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
    """Return the fuser that turns Citi checking with balances into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (citi_banking_with_balance,) = annotated_lazy_frames

        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            citi_banking_with_balance.lazy_frame,
            posting_date=pl.col("Posting date"),
            institution=pl.lit("Citi"),
            account=pl.col("Account"),
            currency=pl.lit("USD"),
            amount=pl.coalesce("Debit (USD)", "Credit (USD)"),
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
    """Return the fuser that turns Wise currency legs into ledger entries."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        (wise_with_balance,) = annotated_lazy_frames

        counterparty_of_the_leg = (
            pl.when(pl.col("Direction") == "IN")
            .then(pl.col("Source name"))
            .otherwise(pl.col("Target name"))
        )
        lazy_frame_as_ledger_entries = _select_ledger_entry_columns(
            wise_with_balance.lazy_frame,
            posting_date=pl.col("Finished on"),
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
    the ledger can stack them."""

    money = pl.Decimal(precision=18, scale=2)
    lazy_frame_as_ledger_entries = lazy_frame.select(
        posting_date.dt.convert_time_zone("UTC").alias("Posting date"),
        institution.cast(pl.String).alias("Institution"),
        account.cast(pl.String).alias("Account"),
        currency.cast(pl.String).alias("Currency"),
        amount.cast(money).alias("Amount"),
        balance.cast(money).alias("Balance"),
        counterparty.cast(pl.String).alias("Counterparty"),
        description.cast(pl.String).alias("Description"),
    )
    return lazy_frame_as_ledger_entries


def ledger_fuser(internal_counterparties: list[str]) -> Fuser:
    """Return the fuser that books every bank's ledger entries into one ledger."""

    def fuser(annotated_lazy_frames: list[AnnotatedLazyFrame]) -> AnnotatedLazyFrame:
        child_lazy_frames = [
            annotated_lazy_frame.lazy_frame
            for annotated_lazy_frame in annotated_lazy_frames
        ]
        stacked_lazy_frame = _stack_vertically(child_lazy_frames)
        sorted_lazy_frame = _sort_by_column(
            stacked_lazy_frame, column="Posting date", descending=True
        )
        lazy_frame_with_transfers = _classify_transfers_by_counterparty(
            sorted_lazy_frame, internal_counterparties=internal_counterparties
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


def _classify_transfers_by_counterparty(
    lazy_frame: pl.LazyFrame, internal_counterparties: list[str]
) -> pl.LazyFrame:
    is_internal_counterparty = pl.col("Counterparty").str.contains_any(
        internal_counterparties, ascii_case_insensitive=True
    )
    transfer = (
        pl.when(is_internal_counterparty)
        .then(pl.lit("Internal"))
        .otherwise(pl.lit("External"))
    )
    return lazy_frame.with_columns(transfer.alias("Transfer"))


def _add_combined_balance_per_currency(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """NOTE: expects newest first. The combined balance is every account's balance
    before its oldest entry, plus all entries up to this row, per currency."""

    account_columns = ["Institution", "Account", "Currency"]
    balance_before_oldest_entry = (
        lazy_frame.group_by(account_columns)
        .agg((pl.col("Balance") - pl.col("Amount")).last().alias("_starting_balance"))
        .group_by("Currency")
        .agg(pl.col("_starting_balance").sum().alias("_combined_starting_balance"))
    )
    lazy_frame_with_starting_balances = lazy_frame.join(
        balance_before_oldest_entry, on="Currency", how="left", maintain_order="left"
    )

    running_total_per_currency = pl.col("Amount").cum_sum(reverse=True).over("Currency")
    combined_balance = (
        pl.col("_combined_starting_balance") + running_total_per_currency
    ).cast(pl.Decimal(precision=18, scale=2))
    lazy_frame_with_combined_balances = lazy_frame_with_starting_balances.with_columns(
        combined_balance.alias("Combined balance")
    ).drop("_combined_starting_balance")

    lazy_frame_with_reordered_columns = _reorder_columns(
        lazy_frame_with_combined_balances,
        column_names_left_to_right=[
            "Posting date",
            "Institution",
            "Account",
            "Currency",
            "Amount",
            "Balance",
            "Combined balance",
            "Transfer",
        ],
    )
    return lazy_frame_with_reordered_columns


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


def _add_running_balance_from_known_balance(
    lazy_frame: pl.LazyFrame,
    *,
    known_balances: pl.LazyFrame,
    amount: pl.Expr,
    date_column: str,
    account_columns: list[str],
    known_balance_column: str,
    balance_column: str,
) -> pl.LazyFrame:
    """NOTE: expects newest first. The balance after each row is worked out
    backwards and forwards from the known balance on its date; an account with no
    known balance is taken to hold 0 after its newest row."""

    latest_known_balances = known_balances.sort("Known balance date").unique(
        subset=account_columns, keep="last"
    )
    lazy_frame_with_known_balances = lazy_frame.join(
        latest_known_balances,
        on=account_columns,
        how="left",
        maintain_order="left",
    )

    amount_or_zero = amount.fill_null(0)
    is_on_or_before_known_date = pl.col("Known balance date").is_null() | (
        pl.col(date_column).dt.date() <= pl.col("Known balance date")
    )
    total_through_known_date = (
        pl.when(is_on_or_before_known_date)
        .then(amount_or_zero)
        .otherwise(0)
        .sum()
        .over(account_columns)
    )
    running_total = amount_or_zero.cum_sum(reverse=True).over(account_columns)
    balance = (
        pl.col(known_balance_column).fill_null(0)
        - total_through_known_date
        + running_total
    ).cast(pl.Decimal(precision=18, scale=2))

    lazy_frame_with_balances = lazy_frame_with_known_balances.with_columns(
        balance.alias(balance_column)
    ).drop("Known balance date", known_balance_column)
    return lazy_frame_with_balances


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
