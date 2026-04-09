> English version: [../../development/testing.md](../../development/testing.md)

# Тестирование

## Запуск

```bash
pip install -e ".[dev]"
pytest
```

## Рекомендованные срезы

```bash
pytest tests/test_model_training_module.py
pytest tests/test_results_analyzer.py
pytest tests/test_training_queue.py
```

## Что проверять после изменений документации

- команды и флаги в примерах соответствуют текущему `--help`;
- коды выхода и резервные пути описаны корректно;
- ссылки между разделами не битые.
