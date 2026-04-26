> English version: [../index.md](../index.md)

# Документация Smart Train

Этот раздел отражает актуальное состояние кода и разбит по сценариям: старт, ежедневная работа с CLI, справочные форматы и архитектура.

## 1) Начало работы

- [Установка](getting-started/installation.md)
- [Рабочий каталог и каталоги проекта](getting-started/workspace.md)
- [Быстрый сценарий `scan -> fusion -> train`](getting-started/quickstart.md)

## 2) CLI-руководство

- [Обзор команд](cli/overview.md)
- [Датасеты: `scan`, `fusion`, `augment`, `balance`, `orient`, `roi`, `hash`, `stats`](cli/datasets.md)
- [Обучение и тест: `train`, `clearml-upload`](cli/training.md)
- [Внешние провайдеры: `providers`, алиасы моделей, интеграция train/inference](cli/providers.md)
- [Очередь: `queue`, `queue-run`](cli/queue.md)
- [Аналитика: `analyze` (включая `pr-curves`, `inference-benchmark`, `inference-plot`, `test-metrics-plot`)](cli/analyze.md)
- [Реестр моделей: `registry`](cli/registry.md)
- [CVAT for images 1.1: `cvat`](cli/cvat.md)
- [Инференс-инструменты: `sahi`, `heatmap`](cli/inference-tools.md)

## 2.1) Профили провайдеров

- [Обзор провайдеров](providers/overview.md)
- [Провайдер DR-YOLO](providers/dr-yolo.md)
- [Провайдер LEAF-YOLO](providers/leaf-yolo.md)
- [Провайдер MFEL-YOLO](providers/mfel-yolo.md)
- [Провайдер MP-YOLO](providers/mp-yolo.md)
- [Провайдер SSDM-YOLO](providers/ssdm-yolo.md)
- [Провайдер Enhanced YOLOv8](providers/enhanced-yolov8.md)

## 3) Reference

- [API и модули](reference/api.md)
- [Форматы данных и контракты файлов](reference/data-formats.md)
- [Формат `training_metadata.json`](reference/training-metadata.md)

## 4) Development

- [Архитектура и диаграммы](development/architecture.md)
- [Тестирование](development/testing.md)
- [Расширение проекта](development/extension-guide.md)
- [Разработка новых внешних провайдеров](development/provider-development.md)

## 5) Migration

- [Устаревшие команды и совместимость](migration/legacy-commands.md)

## Примечания

- Корневой обзор проекта: [README.md](../README.md).
- Старые плоские документы в `docs/*.md` сохранены для обратной совместимости и ссылаются на новые разделы.

## Как читать документацию

- Для быстрого старта: `getting-started -> cli`.
- Для поддержки прод-пайплайна: `cli -> reference`.
- Для изменения кода: `development -> reference -> migration`.

