"""Transformers: frame transformations injected into blocks.

Two kinds live here, split by section:

- **1 -> 1** transformers (`Transformer`): one LazyFrame -> one LazyFrame.
  Used as a `LeafETL.transformer` (see `j6p.blocks`).
- **N -> 1** fusion transformers (`Fuser`): a list of LazyFrames -> one
  LazyFrame — i.e. how multiple frames are joined. Used as a
  `NodeETL.transformer`.

A fuser is essentially a transformer; both are kept in this one module, the
section comments making the 1->1 vs N->1 distinction explicit.
"""

import polars as pl

# ---- 1 -> 1 transformers ----


def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
    """The explicit 1 -> 1 no-op: return the frame unchanged."""
    return frame


# ---- N -> 1 fusion transformers ----


def vertical_concat(frames: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Stack frames vertically (union); all frames must share the same schema."""
    return pl.concat(frames)
