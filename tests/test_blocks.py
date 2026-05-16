import polars as pl
import pytest

from j6p.blocks import Block, LeafBlock, LeafETL, NodeBlock, NodeETL
from j6p.extractors import parquet_reader
from j6p.loaders import parquet_writer
from j6p.transformers import identity, vertical_concat


def _make_reader(data: dict | None = None):
    df = pl.DataFrame(data or {"x": [1, 2], "y": ["a", "b"]})

    def _read() -> pl.LazyFrame:
        return df.lazy()

    return _read


def _capture_writer():
    received: list[pl.DataFrame] = []

    def _write(frame: pl.LazyFrame) -> None:
        received.append(frame.collect())

    return _write, received


# --- LeafBlock ---


def test_leaf_block_run_returns_lazy_frame():
    block = LeafBlock(LeafETL(extractor=_make_reader(), transformer=identity))
    assert isinstance(block.run(), pl.LazyFrame)


def test_leaf_block_extractor_is_invoked():
    calls = []

    def counting_reader() -> pl.LazyFrame:
        calls.append(1)
        return pl.DataFrame({"v": [1]}).lazy()

    LeafBlock(LeafETL(extractor=counting_reader, transformer=identity)).run()
    assert len(calls) == 1


def test_leaf_block_transformer_is_applied():
    def add_col(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(pl.lit(99).alias("added"))

    result = LeafBlock(LeafETL(extractor=_make_reader(), transformer=add_col)).collect()
    assert "added" in result.columns


def test_leaf_block_write_invokes_loader():
    loader, received = _capture_writer()
    LeafBlock(
        LeafETL(extractor=_make_reader(), transformer=identity, loader=loader)
    ).run(write=True)
    assert len(received) == 1


def test_leaf_block_write_without_loader_raises():
    block = LeafBlock(LeafETL(extractor=_make_reader(), transformer=identity))
    with pytest.raises(ValueError, match="a loader is required"):
        block.run(write=True)


# --- collect() ---


def test_collect_returns_dataframe():
    block = LeafBlock(LeafETL(extractor=_make_reader(), transformer=identity))
    assert isinstance(block.collect(), pl.DataFrame)


def test_collect_write_true_invokes_loader():
    loader, received = _capture_writer()
    LeafBlock(
        LeafETL(extractor=_make_reader(), transformer=identity, loader=loader)
    ).collect(write=True)
    assert len(received) == 1


# --- ETL value objects: transformer is required ---


def test_leaf_etl_requires_transformer():
    with pytest.raises(TypeError):
        LeafETL(extractor=_make_reader())


def test_node_etl_requires_transformer():
    with pytest.raises(TypeError):
        NodeETL(extractor=[])


# --- Parquet round-trip (no special block — parquet is just a reader) ---


def test_parquet_roundtrip_via_parquet_reader(tmp_path):
    outfile = tmp_path / "tier0.parquet"
    source = LeafBlock(
        LeafETL(
            extractor=_make_reader({"a": [10, 20], "b": ["x", "y"]}),
            transformer=identity,
            loader=parquet_writer(outfile),
        )
    )
    source.run(write=True)

    cached = LeafBlock(LeafETL(extractor=parquet_reader(outfile), transformer=identity))
    result = cached.collect()
    assert result["a"].to_list() == [10, 20]
    assert result["b"].to_list() == ["x", "y"]


# --- NodeBlock ---


def _leaf(data: dict) -> LeafBlock:
    return LeafBlock(LeafETL(extractor=_make_reader(data), transformer=identity))


def test_node_block_run_returns_lazy_frame():
    node = NodeBlock(
        NodeETL(
            extractor=[_leaf({"v": [1]}), _leaf({"v": [2]})],
            transformer=vertical_concat,
        )
    )
    assert isinstance(node.run(), pl.LazyFrame)


def test_node_block_fuser_receives_all_frames():
    received: list[list[pl.LazyFrame]] = []

    def capturing_fuser(frames: list[pl.LazyFrame]) -> pl.LazyFrame:
        received.append(frames)
        return pl.concat(frames)

    NodeBlock(
        NodeETL(
            extractor=[_leaf({"v": [1]}), _leaf({"v": [2]})],
            transformer=capturing_fuser,
        )
    ).collect()

    assert len(received) == 1
    assert len(received[0]) == 2


def test_node_block_chaining_three_tiers():
    tier1 = NodeBlock(
        NodeETL(
            extractor=[_leaf({"v": [1]}), _leaf({"v": [2]})],
            transformer=vertical_concat,
        )
    )
    tier2 = NodeBlock(
        NodeETL(extractor=[tier1, _leaf({"v": [3]})], transformer=vertical_concat)
    )
    assert tier2.collect()["v"].to_list() == [1, 2, 3]


def test_node_block_is_substitutable_for_block():
    def accepts_any_block(b: Block) -> pl.DataFrame:
        return b.collect()

    node = NodeBlock(
        NodeETL(extractor=[_leaf({"v": [1]})], transformer=vertical_concat)
    )
    assert isinstance(accepts_any_block(node), pl.DataFrame)
