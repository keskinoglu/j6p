"""The block tree: a data pipeline shaped as a tree of ETL blocks.

Each `Block` is one node in the tree and runs an ETL — extract, transform, then
optionally save — always returning a lazy frame.

- `LeafBlock` is a **leaf node**: it extracts from a single source (a `Reader`)
  and applies a 1 -> 1 `Transformer`.
- `NodeBlock` is a **non-leaf (internal) node**: its children are other
  `Block`s; it extracts their lazy frames and fuses them with an N -> 1 `Fuser`.

A `NodeBlock`'s children are themselves `Block`s, so nodes nest to any depth:
the leaves read the raw sources and each internal tier fuses the tier below it,
up to a single root.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from j6p.type_aliases import Fuser, LazyFrame, Reader, Transformer, Writer


class Block[ExtractedPayload](ABC):
    """One node in the block tree: runs an ETL and returns a lazy frame.

    `run()` extracts, transforms, and always returns the lazy frame; with
    `write=True` it also loads. Subclasses implement `_extract` / `_transform`.
    """

    def __init__(self, loader: Writer | None) -> None:
        self._loader = loader

    def run(self, *, write: bool = False) -> LazyFrame:
        lazy_frame = self._transform(self._extract())
        if write:
            if self._loader is None:
                raise ValueError("a loader is required when write=True")
            self._loader(lazy_frame)
        return lazy_frame

    def collect(self, *, write: bool = False) -> pl.DataFrame:
        return self.run(write=write).collect()

    @abstractmethod
    def _extract(self) -> ExtractedPayload: ...

    @abstractmethod
    def _transform(self, extracted: ExtractedPayload) -> LazyFrame: ...


class LeafBlock(Block[LazyFrame]):
    """A leaf node: extracts from one source and applies a 1 -> 1 transformer."""

    def __init__(self, etl: LeafETL) -> None:
        super().__init__(etl.loader)
        self._etl = etl

    def _extract(self) -> LazyFrame:
        return self._etl.extractor()

    def _transform(self, extracted: LazyFrame) -> LazyFrame:
        return self._etl.transformer(extracted)


class NodeBlock(Block[list[LazyFrame]]):
    """A non-leaf (internal) node: fuses its child blocks with an N -> 1 fuser."""

    def __init__(self, etl: NodeETL) -> None:
        super().__init__(etl.loader)
        self._etl = etl

    def _extract(self) -> list[LazyFrame]:
        return [block.run() for block in self._etl.extractor]

    def _transform(self, extracted: list[LazyFrame]) -> LazyFrame:
        return self._etl.transformer(extracted)


@dataclass(frozen=True, slots=True)
class LeafETL:
    extractor: Reader
    transformer: Transformer
    loader: Writer | None = None


@dataclass(frozen=True, slots=True)
class NodeETL:
    extractor: Sequence[Block]
    transformer: Fuser
    loader: Writer | None = None
