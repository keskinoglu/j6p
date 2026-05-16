"""Fusers: many-to-one frame transformations — how multiple frames are joined.

A Fuser is essentially a Transformer, but **N -> 1**: it takes a list of
LazyFrames (the outputs of a node's child blocks) and returns a single
LazyFrame. It lives in its own module — rather than beside the 1 -> 1
Transformers in `j6p.transformers` — to make explicit that this is the
many-to-one case, and that a Fuser *is how multiple frames are joined* into one.

It is the `transformer` of a `NodeETL` (see `j6p.blocks`).
"""

import polars as pl


def vertical_concat(frames: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Stack frames vertically (union); all frames must share the same schema."""
    return pl.concat(frames)
