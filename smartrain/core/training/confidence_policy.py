"""Production confidence threshold policy for inference.

Default contract: objective **A** (F1), **macro** aggregation, fallback **0.25**.
Override with ``--confidence-objective B|C`` (and optional aggregation) when a
recommendation JSON is supplied.
"""

from __future__ import annotations

from typing import Any, Mapping

from smartrain.core.training.confidence_recommendation import DEFAULT_FALLBACK_CONFIDENCE


DEFAULT_OBJECTIVE = "A"
DEFAULT_AGGREGATION = "macro"


def resolve_inference_confidence(
    rec_json: Mapping[str, Any] | None,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    aggregation: str = DEFAULT_AGGREGATION,
    fallback: float = DEFAULT_FALLBACK_CONFIDENCE,
) -> float:
    """Pick a single confidence threshold for production inference.

    Prefers ``objectives[objective].aggregations[aggregation].global.threshold``,
    then legacy ``objectives[objective].global.threshold``, else ``fallback``.
    """
    fb = float(fallback)
    if not isinstance(rec_json, Mapping):
        return fb
    objectives = rec_json.get("objectives")
    if not isinstance(objectives, Mapping):
        return fb
    key = str(objective or DEFAULT_OBJECTIVE).strip().upper() or DEFAULT_OBJECTIVE
    item = objectives.get(key)
    if not isinstance(item, Mapping):
        return fb
    agg = str(aggregation or DEFAULT_AGGREGATION).strip().lower() or DEFAULT_AGGREGATION
    aggregations = item.get("aggregations")
    if isinstance(aggregations, Mapping):
        bucket = aggregations.get(agg)
        if isinstance(bucket, Mapping):
            g = bucket.get("global")
            if isinstance(g, Mapping) and g.get("threshold") is not None:
                try:
                    return float(g["threshold"])
                except (TypeError, ValueError):
                    pass
    # Backward compatible: primary global is macro.
    if agg == "macro":
        g = item.get("global")
        if isinstance(g, Mapping) and g.get("threshold") is not None:
            try:
                return float(g["threshold"])
            except (TypeError, ValueError):
                return fb
    return fb
