"""Transformers: one-to-one frame transformations.

A Transformer is a **1 -> 1** mapping — it takes a single LazyFrame and returns
a single LazyFrame with some normalization applied. It is the `transformer` of
a `LeafETL` (see `j6p.blocks`).

For the **N -> 1** case (combining several frames into one) see the Fusers in
`j6p.fusers`: a Fuser is the same idea, kept in a separate module to make the
many-to-one nature explicit.
"""

import polars as pl


def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
    """The explicit 1 -> 1 no-op: return the frame unchanged."""
    return frame
