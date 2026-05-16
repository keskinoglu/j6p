from collections.abc import Callable

import polars as pl

Frame = pl.LazyFrame
Reader = Callable[[], Frame]  # leaf extractor: reads ONE source
Transformer = Callable[[Frame], Frame]  # leaf transformer: 1 -> 1
Fuser = Callable[[list[Frame]], Frame]  # node transformer: N -> 1
Writer = Callable[[Frame], None]  # loader: side-effect, returns nothing
