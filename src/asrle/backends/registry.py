from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

from asrle.backends.base import ASRBackend


@dataclass(frozen=True)
class BackendSpec:
    name: str
    loader: Callable[[], type[ASRBackend]]


class BackendRegistry:
    """
    Loads backends from:
    - installed entry points group: asrle.backends
    - local builtins (fallback)
    """

    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}
        self._loaded = False

    def _load_entrypoints(self) -> None:
        try:
            from importlib.metadata import entry_points  # py3.10+
        except Exception:
            return

        try:
            eps = entry_points(group="asrle.backends")
        except TypeError:
            eps = entry_points().get("asrle.backends", [])
        for ep in eps:
            name = ep.name

            def make_loader(e=ep) -> type[ASRBackend]:
                obj = e.load()
                return obj

            self._specs[name] = BackendSpec(name=name, loader=make_loader)

    def _load_builtins(self) -> None:
        # Always available dummy backend for tests/demos
        self._specs.setdefault(
            "dummy",
            BackendSpec(
                name="dummy",
                loader=lambda: getattr(importlib.import_module("asrle.backends.dummy"), "DummyBackend"),
            ),
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_entrypoints()
        self._load_builtins()
        self._loaded = True

    def list(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._specs.keys())

    def create(self, name: str, **kwargs: Any) -> ASRBackend:
        self._ensure_loaded()
        if name not in self._specs:
            raise KeyError(f"Unknown backend '{name}'. Available: {', '.join(self.list())}")
        cls = self._specs[name].loader()
        return cls(**kwargs)  # type: ignore[arg-type]
