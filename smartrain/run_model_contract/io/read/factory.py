from __future__ import annotations

from smartrain.unified.io.read.model_adapter import ModelAdapter
from smartrain.unified.io.read.resolvers import infer_source_kind
from smartrain.unified.io.read.run_adapter import RunAdapter


class ReadAdapterFactory:
    def resolve(self, source_kind: str | None, source_ref: str):
        kind = (source_kind or "").strip().lower() or infer_source_kind(source_ref)
        if kind == "run":
            return RunAdapter()
        if kind == "model":
            return ModelAdapter()
        raise ValueError(f"Unsupported source_kind: {source_kind!r}")

