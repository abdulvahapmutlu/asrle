from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TypeVar

from asrle.utils.time import Timer

T = TypeVar("T")


@dataclass
class Profiler:
    stages_s: dict[str, float] = field(default_factory=dict)

    def time(self, stage: str, fn: Callable[[], T]) -> T:
        with Timer(stage) as t:
            out = fn()
        self.stages_s[stage] = self.stages_s.get(stage, 0.0) + t.elapsed_s
        return out

    def merge(self, other: "Profiler") -> None:
        for k, v in other.stages_s.items():
            self.stages_s[k] = self.stages_s.get(k, 0.0) + v
