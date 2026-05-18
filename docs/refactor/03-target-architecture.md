# Target Architecture

## Layers

1. CLI Router Layer
2. Input Contracts Layer
3. Orchestrator Layer
4. Backend Registry + Adapters
5. Task Adapters + Metrics Adapters
6. Artifact/Canonical Storage Layer
7. Analyze Layer

## Dependency Rules

- CLI layer must not call backend implementations directly.
- Orchestrators depend on interfaces, not concrete backends.
- Task-specific logic must stay in task adapters.
- Analyze reads unified/canonical artifacts only.

## Invariants

- Every command yields a normalized response object.
- Every persisted artifact includes schema version and provenance.
- New backends/tasks are added by adapters, not by patching CLI branching.
