"""Transformers: lazy-frame transformations injected into blocks.

Two kinds live here, split by section:

- **1 -> 1** transformers (`Transformer`): one LazyFrame -> one LazyFrame.
  Used as a `LeafETL.transformer` (see `j6p.blocks`).
- **N -> 1** fusion transformers (`Fuser`): a list of LazyFrames -> one
  LazyFrame — i.e. how multiple lazy frames are joined. Used as a
  `NodeETL.transformer`.

A fuser is essentially a transformer; both are kept in this one module, the
section comments making the 1->1 vs N->1 distinction explicit.
"""

import polars as pl

# ---- 1 -> 1 transformers ----


def identity(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
    """The explicit 1 -> 1 no-op: return the lazy frame unchanged."""
    return lazy_frame


# ---- N -> 1 fusion transformers ----


def vertical_concat(lazy_frames: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Stack the lazy frames vertically (union); all must share one schema."""
    return pl.concat(lazy_frames)
