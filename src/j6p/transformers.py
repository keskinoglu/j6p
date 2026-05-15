"""Per-dataset frame-to-frame normalization strategies.

Each function here is a Transformer: it receives a LazyFrame and returns a
LazyFrame with the desired normalization applied. Inject one into a SourceBlock
or FusionBlock via the `transformer=` argument.
"""

import polars as pl


def identity(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Pass the frame through unchanged (useful as an explicit no-op)."""
    return frame
