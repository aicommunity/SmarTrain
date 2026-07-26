> Russian version: [../ru/audit/2026-07-26-project-audit.md](../ru/audit/2026-07-26-project-audit.md)

# SmarTrain project audit (2026-07-26)

**Package version:** 0.0.5 · **Scope:** ~69.8k LOC in `smartrain/` + ~27.5k LOC tests

## Verdict

SmarTrain is a mature workspace-first YOLO/MLOps CLI with layered architecture (`cli → workflows → services → backends/tasks/run_model_contract`), file contracts, and strong regression coverage. Main code debt: oversized modules, remaining interactive/MFEL duplication, legacy metadata reads outside the gateway in places historically noted, narrow lint/coverage scope, and optional science/MLOps upgrades.

## Architecture

| Decision | Assessment |
|----------|------------|
| Workspace-as-database | Strong fit for local CV pipelines |
| Layer guardrails (`services` ↛ `workflows`) | Strong; enforced in CI |
| Backend + task adapters | Strong extensibility path |
| Unified run/model contract | Strong; dual-write supported |
| Dual Typer/argparse CLI | Pragmatic compromise |

Primary risks: god modules (`dataset_augment`, `model_convert_cli`, `dataset_balance`), logical `services ↔ workflows` cycle via adapters, `core → services` inversion in a few runtime helpers, queue without auto-retry, no container image.

See also: [development/architecture.md](../development/architecture.md), [refactor/13-project-current-state.md](../refactor/13-project-current-state.md).

## Algorithms vs literature

| Area | Status | References |
|------|--------|------------|
| Repeat Factor Sampling | Aligned (offline expansion) | [arXiv:1908.03195](https://arxiv.org/abs/1908.03195) (LVIS) |
| Class-Balanced effective number weights | Formula aligned; used as sampling weights | [arXiv:1901.05555](https://arxiv.org/abs/1901.05555) |
| F-β confidence thresholds (A/B/C) | Good practice | [arXiv:2210.10221](https://arxiv.org/abs/2210.10221) |
| Class-aware offline augment | Docs-motivated heuristics | DODA (ICLR 2024), [arXiv:2302.05499](https://arxiv.org/abs/2302.05499) |
| SAHI sliced inference | Aligned defaults | [arXiv:2202.06934](https://arxiv.org/abs/2202.06934) |
| Leaderboard composite score | Ops heuristic (not a published metric) | — |

Gaps (backlog P2): object-level/IRFS resampling, Optimal LRP, micro-averaged F-β policy, SAHI fine-tune recipe.

## Code quality snapshot

- ~73% of package LOC in `services/`; several modules &gt;1.5k LOC
- Duplication hotspots: interactive wizards, MFEL launchers, dual prompt helpers, `_resolve_run_ref`
- Dead compat shims under `services/datasets/augment_*.py` (removal in P0)
- ~400 broad `except Exception`; print-centric CLI I/O
- Strong test volume; coverage gate and full-package ruff/mypy historically limited

## Remediation waves

**Execution flag:** `1` = in current delivery wave; `0` = backlog only.

### P0 (execution=1)

1. Registry CLI → gateway-first / contract metadata (no `metrics_reader.load_metadata`)
2. Remove dead `augment_{cli,pipeline,donors,yolo_io}` shims; drop stale `smartrain/domain` guardrail
3. Shared interactive preamble helper
4. Shared MFEL shim module
5. Refresh [refactor/02-duplication-matrix.md](../refactor/02-duplication-matrix.md)

### P1 (execution=1)

1. Split `model_convert_cli` into services + thin facade
2. Split `dataset_augment`
3. Split `dataset_balance`
4. Unify prompt APIs
5. Shared `resolve_run_ref` in `core.runtime`
6. Logging baseline + coverage/ruff/mypy on `core` / `run_model_contract` / `backends`
7. Algorithm paper citations in code and CLI docs

### P2 (execution=0) — backlog

1. IRFS / object-aware RFS
2. Optimal LRP operating point
3. Micro-averaged F-β + default threshold policy
4. SAHI slicing-aided fine-tune recipe
5. Queue retry/backoff
6. Dockerfile + CUDA lockfile
7. Balance-preset eval harness
8. Break `core → services` inversions

Tracker: [refactor/09-tech-debt.md](../refactor/09-tech-debt.md).
