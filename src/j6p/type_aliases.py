from collections.abc import Callable

from polars import LazyFrame

Reader = Callable[[], LazyFrame]  # leaf extractor: reads ONE source
Transformer = Callable[[LazyFrame], LazyFrame]  # leaf transformer: 1 -> 1
Fuser = Callable[[list[LazyFrame]], LazyFrame]  # node transformer: N -> 1
Writer = Callable[[LazyFrame], None]  # loader: side-effect, returns nothing
