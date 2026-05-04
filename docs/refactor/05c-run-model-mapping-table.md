# Legacy to Canonical Mapping

## Legacy Run -> Canonical

| Legacy Field | Canonical Field | Rule |
|---|---|---|
| run directory name | `CanonicalRunRef.run_id` | folder basename |
| canonical/legacy model path | `CanonicalModelRef.weights_path` | resolve canonical first, fallback legacy |
| test metrics csv/json | `CanonicalMetricsRef` | normalize and namespace |
| inference json report | `CanonicalPredictionRef` | extract path/count/schema |

## Legacy Model -> Canonical

| Legacy Field | Canonical Field | Rule |
|---|---|---|
| model directory name | `CanonicalModelRef.model_id` | folder basename |
| model manifest | `CanonicalArtifactRef` | map source and metadata |
| weights file extension | `CanonicalModelRef.format` | infer from extension |

## Unmapped Fields

- Source-specific temporary diagnostics that have no stable consumer.

## Conflict Resolution

- Canonical metadata has priority if present.
- Otherwise prefer explicit CLI-provided references over inferred values.
