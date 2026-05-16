# Duplication Matrix

| Area | Files | Duplication Type | Refactor Target |
|---|---|---|---|
| CLI prompts | `train_cli_callbacks.py`, `dataset_augment.py`, `dataset_balance.py` | Repeated prompt/validation | `cli/core/interactive.py` |
| Replay output | train/test/inference/dataset commands | Slightly different replay builders | `cli/core/replay.py` |
| Dataset catalog/splits | `dataset_augment.py`, `dataset_balance.py`, `dataset_orient.py`, `dataset_prune.py` | Repeated dataset lookup and split logic | `dataset_cli_common.py` (`load_dataset_catalog`, …) + `dataset_cli_catalog.py` (interactive dataset prompt helper) |
| Run/model resolution | `model_test_cli.py`, `inference_cli.py`, `results_analyzer.py` | Similar target resolution branches | canonical gateway layer |
| Backend dispatch | `inference_backends.py`, `model_test_backends.py`, train path | Backend-specific branching in app layer | capability registry |
| Detection assumptions | `model_test_backends.py`, `results_analyzer.py`, dataset modules | Box-specific fields and metrics | task adapters |

## Priority

1. CLI request/interactive/replay
2. Dataset helper extraction
3. Run/model canonical contract
4. Backend and task abstraction
