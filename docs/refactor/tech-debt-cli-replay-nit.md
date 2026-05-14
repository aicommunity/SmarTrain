# Tech debt: CLI replay and `--nit`

## Register

| id | status | context | deferred_decision | resolution |
|----|--------|---------|-------------------|------------|
| TD-R01 | OPEN | `cli.py` / `python -m` | Typer-only `--nit`; `python -m` entry without Typer does not strip `--nit`. | Document WONTFIX or env `SMART_TRAIN_FORCE_NON_INTERACTIVE` after implementation. |
| TD-R02 | OPEN | `train` argparse | Unify `--nit` with `--yes` / `-y` dest or keep Typer-only `--nit` for train. | Prefer Typer-only; document in architecture. |

## Burn-down checklist

- [ ] TD-R01
- [ ] TD-R02
