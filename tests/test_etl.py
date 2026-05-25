import polars as pl
import pytest

from j6p.etl import ETL, LeafETL, NodeETL
from j6p.extractors import parquet_reader
from j6p.loaders import parquet_writer, return_only
from j6p.transformers import identity


def _make_reader(data: dict | None = None):
    df = pl.DataFrame(data or {"x": [1, 2], "y": ["a", "b"]})

    def _read() -> pl.LazyFrame:
        return df.lazy()

    return _read


def _concat_fuser(lazy_frames: list[pl.LazyFrame]) -> pl.LazyFrame:
    return pl.concat(lazy_frames)


def _capture_writer():
    received: list[pl.DataFrame] = []

    def _write(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        received.append(lazy_frame.collect())
        return lazy_frame

    return _write, received


# --- LeafETL ---


def test_leaf_etl_run_returns_lazy_frame():
    etl = LeafETL(
        extractor=_make_reader(), transformer=identity(), loader=return_only()
    )
    assert isinstance(etl.run(), pl.LazyFrame)


def test_leaf_etl_extractor_is_invoked():
    calls = []

    def counting_reader() -> pl.LazyFrame:
        calls.append(1)
        return pl.DataFrame({"v": [1]}).lazy()

    LeafETL(
        extractor=counting_reader, transformer=identity(), loader=return_only()
    ).run()
    assert len(calls) == 1


def test_leaf_etl_transformer_is_applied():
    def add_col(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        return lazy_frame.with_columns(pl.lit(99).alias("added"))

    result = LeafETL(
        extractor=_make_reader(), transformer=add_col, loader=return_only()
    ).collect()
    assert "added" in result.columns


def test_leaf_etl_invokes_loader():
    loader, received = _capture_writer()
    LeafETL(extractor=_make_reader(), transformer=identity(), loader=loader).run()
    assert len(received) == 1


def test_leaf_etl_requires_loader():
    with pytest.raises(TypeError):
        LeafETL(extractor=_make_reader(), transformer=identity())


# --- collect() ---


def test_collect_returns_dataframe():
    etl = LeafETL(
        extractor=_make_reader(), transformer=identity(), loader=return_only()
    )
    assert isinstance(etl.collect(), pl.DataFrame)


def test_collect_invokes_loader():
    loader, received = _capture_writer()
    LeafETL(extractor=_make_reader(), transformer=identity(), loader=loader).collect()
    assert len(received) == 1


# --- transformer is required ---


def test_leaf_etl_requires_transformer():
    with pytest.raises(TypeError):
        LeafETL(extractor=_make_reader(), loader=return_only())


def test_node_etl_requires_transformer():
    with pytest.raises(TypeError):
        NodeETL(extractor=[], loader=return_only())


# --- Parquet round-trip (parquet is just a reader) ---


def test_parquet_roundtrip_via_parquet_reader(tmp_path):
    outfile = tmp_path / "tier0.parquet"
    source = LeafETL(
        extractor=_make_reader({"a": [10, 20], "b": ["x", "y"]}),
        transformer=identity(),
        loader=parquet_writer(outfile),
    )
    source.run()

    cached = LeafETL(
        extractor=parquet_reader(outfile),
        transformer=identity(),
        loader=return_only(),
    )
    result = cached.collect()
    assert result["a"].to_list() == [10, 20]
    assert result["b"].to_list() == ["x", "y"]


# --- NodeETL ---


def _leaf(data: dict) -> LeafETL:
    return LeafETL(
        extractor=_make_reader(data), transformer=identity(), loader=return_only()
    )


def test_node_etl_run_returns_lazy_frame():
    node = NodeETL(
        extractor=[_leaf({"v": [1]}), _leaf({"v": [2]})],
        transformer=_concat_fuser,
        loader=return_only(),
    )
    assert isinstance(node.run(), pl.LazyFrame)


def test_node_etl_fuser_receives_all_frames():
    received: list[list[pl.LazyFrame]] = []

    def capturing_fuser(lazy_frames: list[pl.LazyFrame]) -> pl.LazyFrame:
        received.append(lazy_frames)
        return pl.concat(lazy_frames)

    NodeETL(
        extractor=[_leaf({"v": [1]}), _leaf({"v": [2]})],
        transformer=capturing_fuser,
        loader=return_only(),
    ).collect()

    assert len(received) == 1
    assert len(received[0]) == 2


def test_node_etl_chaining_three_tiers():
    tier1 = NodeETL(
        extractor=[_leaf({"v": [1]}), _leaf({"v": [2]})],
        transformer=_concat_fuser,
        loader=return_only(),
    )
    tier2 = NodeETL(
        extractor=[tier1, _leaf({"v": [3]})],
        transformer=_concat_fuser,
        loader=return_only(),
    )
    assert tier2.collect()["v"].to_list() == [1, 2, 3]


def test_node_etl_is_substitutable_for_etl():
    def accepts_any_etl(e: ETL) -> pl.DataFrame:
        return e.collect()

    node = NodeETL(
        extractor=[_leaf({"v": [1]})],
        transformer=_concat_fuser,
        loader=return_only(),
    )
    assert isinstance(accepts_any_etl(node), pl.DataFrame)


def test_return_only_is_passthrough():
    lf = pl.DataFrame({"v": [1]}).lazy()
    assert return_only()(lf) is lf
