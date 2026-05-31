import polars as pl
import pytest

from j6p.datatypes import AnnotatedDataFrame, AnnotatedLazyFrame
from j6p.etl import ETL, LeafETL, NodeETL
from j6p.extractors import parquet_reader
from j6p.loaders import parquet_writer, return_only
from j6p.transformers import identity


def _make_reader(data: dict | None = None):
    df = pl.DataFrame(data or {"x": [1, 2], "y": ["a", "b"]})

    def _read() -> AnnotatedLazyFrame:
        annotations = {}
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=df.lazy(), annotations=annotations
        )
        return annotated_lazy_frame

    return _read


def _concat_fuser(
    annotated_lazy_frames: list[AnnotatedLazyFrame],
) -> AnnotatedLazyFrame:
    child_lazy_frames = [
        annotated_lazy_frame.lazy_frame
        for annotated_lazy_frame in annotated_lazy_frames
    ]
    stacked_lazy_frame = pl.concat(child_lazy_frames)
    child_annotations = [
        annotated_lazy_frame.annotations
        for annotated_lazy_frame in annotated_lazy_frames
    ]
    merged_annotations = {"children": child_annotations}
    annotated_lazy_frame = AnnotatedLazyFrame(
        lazy_frame=stacked_lazy_frame, annotations=merged_annotations
    )
    return annotated_lazy_frame


def _capture_writer():
    received: list[pl.DataFrame] = []

    def _write(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        received.append(annotated_lazy_frame.lazy_frame.collect())
        return annotated_lazy_frame

    return _write, received


# --- LeafETL ---


def test_leaf_etl_run_returns_annotated_lazy_frame():
    etl = LeafETL(
        extractor=_make_reader(), transformer=identity(), loader=return_only()
    )
    assert isinstance(etl.run(), AnnotatedLazyFrame)


def test_leaf_etl_extractor_is_invoked():
    calls = []

    def counting_reader() -> AnnotatedLazyFrame:
        calls.append(1)
        annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=pl.DataFrame({"v": [1]}).lazy(),
            annotations={},
        )
        return annotated_lazy_frame

    LeafETL(
        extractor=counting_reader, transformer=identity(), loader=return_only()
    ).run()
    assert len(calls) == 1


def test_leaf_etl_transformer_is_applied():
    def add_col(annotated_lazy_frame: AnnotatedLazyFrame) -> AnnotatedLazyFrame:
        lazy_frame_with_added_column = annotated_lazy_frame.lazy_frame.with_columns(
            pl.lit(99).alias("added")
        )
        updated_annotated_lazy_frame = AnnotatedLazyFrame(
            lazy_frame=lazy_frame_with_added_column,
            annotations=annotated_lazy_frame.annotations,
        )
        return updated_annotated_lazy_frame

    result = LeafETL(
        extractor=_make_reader(), transformer=add_col, loader=return_only()
    ).collect()
    assert "added" in result.data_frame.columns


def test_leaf_etl_invokes_loader():
    loader, received = _capture_writer()
    LeafETL(extractor=_make_reader(), transformer=identity(), loader=loader).run()
    assert len(received) == 1


def test_leaf_etl_requires_loader():
    with pytest.raises(TypeError):
        LeafETL(extractor=_make_reader(), transformer=identity())


# --- collect() ---


def test_collect_returns_annotated_dataframe():
    etl = LeafETL(
        extractor=_make_reader(), transformer=identity(), loader=return_only()
    )
    assert isinstance(etl.collect(), AnnotatedDataFrame)


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
    assert result.data_frame["a"].to_list() == [10, 20]
    assert result.data_frame["b"].to_list() == ["x", "y"]


# --- NodeETL ---


def _leaf(data: dict) -> LeafETL:
    return LeafETL(
        extractor=_make_reader(data), transformer=identity(), loader=return_only()
    )


def test_node_etl_run_returns_annotated_lazy_frame():
    node = NodeETL(
        extractor=[_leaf({"v": [1]}), _leaf({"v": [2]})],
        transformer=_concat_fuser,
        loader=return_only(),
    )
    assert isinstance(node.run(), AnnotatedLazyFrame)


def test_node_etl_fuser_receives_all_frames():
    received: list[list[AnnotatedLazyFrame]] = []

    def capturing_fuser(
        annotated_lazy_frames: list[AnnotatedLazyFrame],
    ) -> AnnotatedLazyFrame:
        received.append(annotated_lazy_frames)
        return _concat_fuser(annotated_lazy_frames)

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
    assert tier2.collect().data_frame["v"].to_list() == [1, 2, 3]


def test_node_etl_is_substitutable_for_etl():
    def accepts_any_etl(e: ETL) -> AnnotatedDataFrame:
        return e.collect()

    node = NodeETL(
        extractor=[_leaf({"v": [1]})],
        transformer=_concat_fuser,
        loader=return_only(),
    )
    assert isinstance(accepts_any_etl(node), AnnotatedDataFrame)


def test_return_only_is_passthrough():
    annotated_lazy_frame = AnnotatedLazyFrame(
        lazy_frame=pl.DataFrame({"v": [1]}).lazy(),
        annotations={},
    )
    assert return_only()(annotated_lazy_frame) is annotated_lazy_frame
