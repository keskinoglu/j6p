from pathlib import Path

import polars as pl

from j6p.datatypes import AnnotatedLazyFrame
from j6p.transformers import (
    _add_new_schwab_account_column_from_filename,
    schwab_brokerage_fuser,
    schwab_brokerage_transformer,
)


def _make_minimal_schwab_brokerage_annotated_lazy_frame(
    filename: str,
) -> AnnotatedLazyFrame:
    lazy_frame = pl.DataFrame(
        {
            "Date": ["01/15/2026"],
            "Price": ["$150.00"],
            "Fees & Comm": ["$0.00"],
            "Amount": ["$150.00"],
            "Quantity": ["1"],
            "AcctgRuleCd": ["1"],
        }
    ).lazy()
    source_path = Path(filename)
    annotations = {"source_path": source_path}
    annotated_lazy_frame = AnnotatedLazyFrame(
        lazy_frame=lazy_frame, annotations=annotations
    )
    return annotated_lazy_frame


# ---- _add_new_schwab_account_column_from_filename ----


def test_account_parsed_from_simple_filename():
    lazy_frame = pl.DataFrame({"v": [1]}).lazy()
    source_path = Path("IRA_XXX020_Transactions_20260510-083544.json")

    result = _add_new_schwab_account_column_from_filename(lazy_frame, source_path)

    assert result.collect()["account"].to_list() == ["IRA_XXX020"]


def test_account_parsed_from_multi_segment_filename():
    lazy_frame = pl.DataFrame({"v": [1]}).lazy()
    source_path = Path("Margin_Money_Market_XXX777_Transactions_20260510-083614.json")

    result = _add_new_schwab_account_column_from_filename(lazy_frame, source_path)

    assert result.collect()["account"].to_list() == ["Margin_Money_Market_XXX777"]


# ---- schwab_brokerage_transformer ----


def test_schwab_brokerage_transformer_output_has_account_column():
    annotated_lazy_frame = _make_minimal_schwab_brokerage_annotated_lazy_frame(
        "IRA_XXX020_Transactions_20260510-083544.json"
    )

    result = schwab_brokerage_transformer()(annotated_lazy_frame)

    data_frame = result.lazy_frame.collect()
    assert "account" in data_frame.columns
    assert all(value == "IRA_XXX020" for value in data_frame["account"].to_list())


def test_schwab_brokerage_transformer_leading_columns_are_dates_then_account():
    annotated_lazy_frame = _make_minimal_schwab_brokerage_annotated_lazy_frame(
        "IRA_XXX020_Transactions_20260510-083544.json"
    )

    result = schwab_brokerage_transformer()(annotated_lazy_frame)

    leading_columns = result.lazy_frame.collect_schema().names()[:3]
    assert leading_columns == ["posted_date", "as_of_date", "account"]


# ---- schwab_brokerage_fuser ----


def test_schwab_brokerage_fuser_does_not_deduplicate_across_accounts():
    """Regression: _drop_duplicate_rows once collapsed cross-account duplicates."""
    ira_annotated_lazy_frame = _make_minimal_schwab_brokerage_annotated_lazy_frame(
        "IRA_XXX020_Transactions_20260510-083544.json"
    )
    etf_annotated_lazy_frame = _make_minimal_schwab_brokerage_annotated_lazy_frame(
        "ETFs_XXX309_Transactions_20260510-083521.json"
    )

    transformer = schwab_brokerage_transformer()
    ira_transformed = transformer(ira_annotated_lazy_frame)
    etf_transformed = transformer(etf_annotated_lazy_frame)

    fused = schwab_brokerage_fuser()([ira_transformed, etf_transformed])

    assert len(fused.lazy_frame.collect()) == 2


def test_schwab_brokerage_fuser_output_annotations_contain_children():
    ira_annotated_lazy_frame = _make_minimal_schwab_brokerage_annotated_lazy_frame(
        "IRA_XXX020_Transactions_20260510-083544.json"
    )
    etf_annotated_lazy_frame = _make_minimal_schwab_brokerage_annotated_lazy_frame(
        "ETFs_XXX309_Transactions_20260510-083521.json"
    )

    transformer = schwab_brokerage_transformer()
    ira_transformed = transformer(ira_annotated_lazy_frame)
    etf_transformed = transformer(etf_annotated_lazy_frame)

    fused = schwab_brokerage_fuser()([ira_transformed, etf_transformed])

    assert "children" in fused.annotations
    assert len(fused.annotations["children"]) == 2
