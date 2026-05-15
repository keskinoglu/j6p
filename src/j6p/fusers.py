"""Multi-frame join and merge strategies.

Each function here is a Fuser: it receives a list of LazyFrames (the outputs of
upstream blocks) and returns a single LazyFrame. Inject one into a FusionBlock
via the `fuser=` argument.
"""

import polars as pl


def vertical_concat(frames: list[pl.LazyFrame]) -> pl.LazyFrame:
    """Stack frames vertically (union); all frames must share the same schema."""
    return pl.concat(frames)
