> English version: [../../development/testing.md](../../development/testing.md)

# Тестирование

Настройки pytest заданы в `pyproject.toml` (`[tool.pytest.ini_options]`): `testpaths = ["tests"]`, `pythonpath = ["."]` — команды выполнять из корня репозитория.

Опционально локально (не enforced в CI):

```bash
pip install pytest-cov
pytest --cov=smartrain.cli --cov=smartrain.cli_entrypoints --cov-report=term-missing -q
```

## Запуск

```bash
pip install -e ".[dev]"
pytest
```

## Рекомендованные срезы

```bash
pytest tests/test_train_*.py tests/test_training_metadata*.py
pytest tests/test_results_analyzer.py
pytest tests/test_training_queue.py
pytest tests/test_inference_cli.py -q
pytest tests/regression/test_train_service_guardrails.py -q
pytest -k "replay" tests/test_cli_replay.py -q
```

Дополнительно:

```bash
pytest tests/test_cli_replay.py -q
pytest -k "analyze" tests/test_results_analyzer_workflows.py -q
```

Эталонный паттерн CLI: `tests/test_cli_replay.py`.

Интеграция canonical: `tests/integration/test_canonical_consumers.py`.

## Что проверять после изменений документации

- команды и флаги в примерах соответствуют текущему `--help`;
- коды выхода и резервные пути описаны корректно;
- ссылки между разделами не битые.
