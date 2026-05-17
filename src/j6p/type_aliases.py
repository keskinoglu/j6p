from collections.abc import Callable

from polars import LazyFrame

Extractor = Callable[[], LazyFrame]  # leaf extractor: reads ONE source
Transformer = Callable[[LazyFrame], LazyFrame]  # leaf transformer: 1 -> 1
Fuser = Callable[[list[LazyFrame]], LazyFrame]  # node transformer: N -> 1
Loader = Callable[[LazyFrame], LazyFrame]  # final stage: returns frame; may persist
