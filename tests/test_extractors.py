import json
from unittest.mock import patch

import polars as pl
import pytest

from j6p.datatypes import AnnotatedLazyFrame
from j6p.extractors import (
    parquet_reader,
    schwab_json_reader,
    schwab_xml_reader,
    xls_reader,
)


def test_schwab_json_reader_returns_annotated_lazy_frame_with_source_path(tmp_path):
    source_file = tmp_path / "IRA_XXX020_Transactions_20260510-083544.json"
    source_file.write_text(json.dumps([{"Date": "01/01/2026", "Amount": "$100.00"}]))

    annotated_lazy_frame = schwab_json_reader(path=source_file)()

    assert isinstance(annotated_lazy_frame, AnnotatedLazyFrame)
    assert annotated_lazy_frame.annotations["source_path"] == source_file


def test_schwab_json_reader_with_key_annotates_source_path(tmp_path):
    source_file = tmp_path / "IRA_XXX020_Transactions_20260510-083544.json"
    records = [{"Date": "01/01/2026", "Amount": "$100.00"}]
    source_file.write_text(json.dumps({"BrokerageTransactions": records}))

    annotated_lazy_frame = schwab_json_reader(
        path=source_file, key_containing_records="BrokerageTransactions"
    )()

    assert isinstance(annotated_lazy_frame, AnnotatedLazyFrame)
    assert annotated_lazy_frame.annotations["source_path"] == source_file


def test_schwab_xml_reader_is_constructible_but_raises_on_call(tmp_path):
    source_file = tmp_path / "account.xml"
    extractor = schwab_xml_reader(path=source_file)

    with pytest.raises(NotImplementedError):
        extractor()


def test_xls_reader_returns_annotated_lazy_frame_with_source_path(tmp_path):
    source_file = tmp_path / "transactions.xlsx"

    with patch("polars.read_excel", return_value=pl.DataFrame({"a": [1]})):
        annotated_lazy_frame = xls_reader(path=source_file)()

    assert isinstance(annotated_lazy_frame, AnnotatedLazyFrame)
    assert annotated_lazy_frame.annotations["source_path"] == source_file


def test_parquet_reader_returns_annotated_lazy_frame_with_source_path(tmp_path):
    source_file = tmp_path / "transactions.parquet"
    pl.DataFrame({"a": [1, 2], "b": ["x", "y"]}).write_parquet(source_file)

    annotated_lazy_frame = parquet_reader(path=source_file)()

    assert isinstance(annotated_lazy_frame, AnnotatedLazyFrame)
    assert annotated_lazy_frame.annotations["source_path"] == source_file
