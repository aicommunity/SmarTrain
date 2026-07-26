> English version: [../../audit/2026-07-26-project-audit.md](../../audit/2026-07-26-project-audit.md)

# Аудит проекта SmarTrain (2026-07-26)

**Версия пакета:** 0.0.5 · **Объём:** ~69.8k LOC в `smartrain/` + ~27.5k LOC тестов

## Вердикт

SmarTrain — зрелый workspace-first YOLO/MLOps CLI со слоистой архитектурой (`cli → workflows → services → backends/tasks/run_model_contract`), файловыми контрактами и сильной регрессией. P0–P2 по аудиту закрыты; остаточный долг — oversized-модули и дальнейшее усиление lint/coverage.

## Архитектура

| Решение | Оценка |
|---------|--------|
| Workspace-as-database | Удачно для локальных CV-пайплайнов |
| Guardrails слоёв (`services` ↛ `workflows`) | Сильно; в CI |
| Backend + task adapters | Хорошая расширяемость |
| Unified run/model contract | Сильно |
| Dual Typer/argparse | Прагматичный компромисс |

Риски (исторические; P2 закрыл retry/Docker/`core→services`): god-модули, логический цикл через adapters.

См. также: [development/architecture.md](../development/architecture.md), [refactor/13-project-current-state.md](../../refactor/13-project-current-state.md).

## Алгоритмы и литература

| Область | Статус | Ссылки |
|---------|--------|--------|
| Repeat Factor Sampling | Соответствует (offline) | [arXiv:1908.03195](https://arxiv.org/abs/1908.03195) |
| Class-Balanced effective number | Формула верна; веса сэмплинга | [arXiv:1901.05555](https://arxiv.org/abs/1901.05555) |
| Пороги F-β (A/B/C) | Хорошая практика | [arXiv:2210.10221](https://arxiv.org/abs/2210.10221) |
| Class-aware offline augment | Эвристики по мотивации papers | DODA, [arXiv:2302.05499](https://arxiv.org/abs/2302.05499) |
| SAHI | Defaults согласованы | [arXiv:2202.06934](https://arxiv.org/abs/2202.06934) |
| Leaderboard composite | Операционная эвристика | — |

## Волны доработок

**Поле execution:** `1` — текущая волна; `0` — только бэклог.

- **P0 (execution=1):** registry gateway-first; удаление мёртвых augment shims; interactive helper; MFEL shim; обновление duplication matrix
- **P1 (execution=1):** распил convert/augment/balance; prompts; `resolve_run_ref`; logging + coverage/ruff/mypy; цитаты arXiv
- **P2 (execution=1, closed):** IRFS, LRP, micro-Fβ, SAHI-FT, queue retry, Docker, balance harness, `core→services`

Трекер: [refactor/09-tech-debt.md](../../refactor/09-tech-debt.md).
