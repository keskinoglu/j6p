"""Transformers: factories for the lazy-frame transformations injected into ETLs.

Two kinds live here, split by section:

- **1 -> 1** transformers (`Transformer`): one LazyFrame -> one LazyFrame.
  Used as a `LeafETL.transformer` (see `j6p.etl`).
- **N -> 1** fusion transformers (`Fuser`): a list of LazyFrames -> one
  LazyFrame — i.e. how multiple lazy frames are joined. Used as a
  `NodeETL.transformer`.

A fuser is essentially a transformer; both are kept in this one module, the
section comments making the 1->1 vs N->1 distinction explicit. Each factory
returns a closure named `transformer`.
"""

import polars as pl

from j6p.type_aliases import Fuser, Transformer

# ---- 1 -> 1 transformers ----


def identity() -> Transformer:
    """Return the 1 -> 1 no-op: the lazy frame unchanged."""

    def transformer(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        return lazy_frame

    return transformer


# ---- N -> 1 fusion transformers ----


def vertical_concat() -> Fuser:
    """Return an N -> 1 fuser: stack the lazy frames vertically (union); all
    must share one schema."""

    def transformer(lazy_frames: list[pl.LazyFrame]) -> pl.LazyFrame:
        return pl.concat(lazy_frames)

    return transformer
