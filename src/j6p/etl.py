"""The ETL tree: a data pipeline shaped as a tree of ETL nodes.

Each `ETL` is one node in the tree and runs an ETL — extract, transform, load
— always returning a lazy frame (the load step may be a no-op; see
`j6p.loaders.return_only`).

- `LeafETL` is a **leaf node**: it extracts from a single source (an
  `Extractor`) and applies a 1 -> 1 `Transformer`.
- `NodeETL` is a **non-leaf (internal) node**: its children are other `ETL`s;
  it extracts their lazy frames and fuses them with an N -> 1 `Fuser`.

A `NodeETL`'s children are themselves `ETL`s, so nodes nest to any depth: the
leaves read the raw sources and each internal tier fuses the tier below it, up
to a single root.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import polars as pl

from j6p.type_aliases import Extractor, Fuser, LazyFrame, Loader, Transformer


class ETL[ExtractedPayload, TransformedPayload](ABC):
    """One node in the ETL tree: runs an ETL and returns a lazy frame.

    `run()` extracts, transforms, then loads, returning the lazy frame; the
    injected loader decides whether anything is persisted. Subclasses implement
    `_extract` / `_transform` / `_load`.
    """

    def run(self) -> LazyFrame:
        return self._load(self._transform(self._extract()))

    def collect(self) -> pl.DataFrame:
        return self.run().collect()

    @abstractmethod
    def _extract(self) -> ExtractedPayload: ...

    @abstractmethod
    def _transform(self, extracted: ExtractedPayload) -> TransformedPayload: ...

    @abstractmethod
    def _load(self, transformed: TransformedPayload) -> LazyFrame: ...


class LeafETL(ETL[LazyFrame, LazyFrame]):
    """A leaf node: extracts from one source and applies a 1 -> 1 transformer."""

    def __init__(
        self,
        *,
        extractor: Extractor,
        transformer: Transformer,
        loader: Loader,
    ) -> None:
        self._extractor = extractor
        self._transformer = transformer
        self._loader = loader

    def _extract(self) -> LazyFrame:
        return self._extractor()

    def _transform(self, extracted: LazyFrame) -> LazyFrame:
        return self._transformer(extracted)

    def _load(self, transformed: LazyFrame) -> LazyFrame:
        return self._loader(transformed)


class NodeETL(ETL[list[LazyFrame], LazyFrame]):
    """A non-leaf (internal) node: fuses its child ETLs with an N -> 1 fuser."""

    def __init__(
        self,
        *,
        extractor: Sequence[ETL],
        transformer: Fuser,
        loader: Loader,
    ) -> None:
        self._extractor = extractor
        self._transformer = transformer
        self._loader = loader

    def _extract(self) -> list[LazyFrame]:
        return [child.run() for child in self._extractor]

    def _transform(self, extracted: list[LazyFrame]) -> LazyFrame:
        return self._transformer(extracted)

    def _load(self, transformed: LazyFrame) -> LazyFrame:
        return self._loader(transformed)
