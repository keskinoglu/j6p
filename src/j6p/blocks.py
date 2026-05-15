from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import polars as pl

Frame = pl.LazyFrame
Reader = Callable[[], Frame]
Transformer = Callable[[Frame], Frame]
Writer = Callable[[Frame], None]
Fuser = Callable[[list[Frame]], Frame]


def _identity(frame: Frame) -> Frame:
    return frame


class Block(ABC):
    def __init__(
        self,
        *,
        transformer: Transformer | None = None,
        writer: Writer | None = None,
    ) -> None:
        self._transformer = transformer or _identity
        self._writer = writer

    # ---- public API ----

    def run(self, *, write: bool = False) -> Frame:
        frame = self._transform(self._read())
        if write:
            self._write(frame)
        return frame

    def collect(self, *, write: bool = False) -> pl.DataFrame:
        return self.run(write=write).collect()

    # ---- template steps (private) ----

    @abstractmethod
    def _read(self) -> Frame: ...

    def _transform(self, frame: Frame) -> Frame:
        return self._transformer(frame)

    def _write(self, frame: Frame) -> None:
        if self._writer is None:
            raise ValueError("writer is required when write=True")
        self._writer(frame)


class SourceBlock(Block):
    def __init__(
        self,
        *,
        reader: Reader,
        transformer: Transformer | None = None,
        writer: Writer | None = None,
    ) -> None:
        super().__init__(transformer=transformer, writer=writer)
        self._reader = reader

    def _read(self) -> Frame:
        return self._reader()


class FusionBlock(Block):
    def __init__(
        self,
        blocks: Sequence[Block],
        *,
        fuser: Fuser,
        transformer: Transformer | None = None,
        writer: Writer | None = None,
    ) -> None:
        super().__init__(transformer=transformer, writer=writer)
        self._blocks = list(blocks)
        self._fuser = fuser

    def _read(self) -> Frame:
        return self._fuser([b.run() for b in self._blocks])
