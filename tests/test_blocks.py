import polars as pl
import pytest

from j6p.blocks import Block, FusionBlock, SourceBlock
from j6p.fusers import vertical_concat
from j6p.readers import parquet_reader
from j6p.writers import parquet_writer


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


# --- SourceBlock ---


def test_source_block_run_returns_lazy_frame():
    block = SourceBlock(reader=_make_reader())
    assert isinstance(block.run(), pl.LazyFrame)


def test_source_block_reader_is_invoked():
    calls = []

    def counting_reader() -> pl.LazyFrame:
        calls.append(1)
        return pl.DataFrame({"v": [1]}).lazy()

    SourceBlock(reader=counting_reader).run()
    assert len(calls) == 1


def test_source_block_transformer_is_applied():
    def add_col(frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(pl.lit(99).alias("added"))

    result = SourceBlock(reader=_make_reader(), transformer=add_col).collect()
    assert "added" in result.columns


def test_source_block_write_invokes_writer():
    writer, received = _capture_writer()
    SourceBlock(reader=_make_reader(), writer=writer).run(write=True)
    assert len(received) == 1


def test_source_block_write_without_writer_raises():
    with pytest.raises(ValueError, match="writer is required"):
        SourceBlock(reader=_make_reader()).run(write=True)


# --- collect() ---


def test_collect_returns_dataframe():
    result = SourceBlock(reader=_make_reader()).collect()
    assert isinstance(result, pl.DataFrame)


def test_collect_write_true_invokes_writer():
    writer, received = _capture_writer()
    SourceBlock(reader=_make_reader(), writer=writer).collect(write=True)
    assert len(received) == 1


# --- Parquet round-trip (no special block — parquet is just a reader) ---


def test_parquet_roundtrip_via_parquet_reader(tmp_path):
    outfile = tmp_path / "tier0.parquet"
    source = SourceBlock(
        reader=_make_reader({"a": [10, 20], "b": ["x", "y"]}),
        writer=parquet_writer(outfile),
    )
    source.run(write=True)

    cached = SourceBlock(reader=parquet_reader(outfile))
    result = cached.collect()
    assert result["a"].to_list() == [10, 20]
    assert result["b"].to_list() == ["x", "y"]


# --- FusionBlock ---


def test_fusion_block_run_returns_lazy_frame():
    a = SourceBlock(reader=_make_reader({"v": [1]}))
    b = SourceBlock(reader=_make_reader({"v": [2]}))
    fusion = FusionBlock([a, b], fuser=vertical_concat)
    assert isinstance(fusion.run(), pl.LazyFrame)


def test_fusion_block_fuser_receives_all_frames():
    received: list[list[pl.LazyFrame]] = []

    def capturing_fuser(frames: list[pl.LazyFrame]) -> pl.LazyFrame:
        received.append(frames)
        return pl.concat(frames)

    a = SourceBlock(reader=_make_reader({"v": [1]}))
    b = SourceBlock(reader=_make_reader({"v": [2]}))
    FusionBlock([a, b], fuser=capturing_fuser).collect()

    assert len(received) == 1
    assert len(received[0]) == 2


def test_fusion_block_chaining_three_tiers():
    a = SourceBlock(reader=_make_reader({"v": [1]}))
    b = SourceBlock(reader=_make_reader({"v": [2]}))
    c = SourceBlock(reader=_make_reader({"v": [3]}))

    tier1 = FusionBlock([a, b], fuser=vertical_concat)
    tier2 = FusionBlock([tier1, c], fuser=vertical_concat)

    assert tier2.collect()["v"].to_list() == [1, 2, 3]


def test_fusion_block_is_substitutable_for_block():
    def accepts_any_block(b: Block) -> pl.DataFrame:
        return b.collect()

    a = SourceBlock(reader=_make_reader({"v": [1]}))
    fusion = FusionBlock([a], fuser=vertical_concat)
    assert isinstance(accepts_any_block(fusion), pl.DataFrame)
