# Project-Based Practice: посткалибровка вероятностей эмоций

Небольшой учебный проект на датасете RAVDESS. На первом этапе скрипт читает
имена WAV-файлов и создаёт CSV с готовой разметкой эмоций. Он работает как с
распакованной папкой, так и непосредственно с ZIP-архивом.

## Быстрый запуск

```bash
git clone https://github.com/brattsew/project-based-practice.git
cd project-based-practice
uv sync
```

После установки положите `Audio_Speech_Actors_01-24.zip` в `data/raw/` и
выполните:

```bash
uv run ravdess-convert \
  data/raw/Audio_Speech_Actors_01-24.zip \
  --output data/processed/ravdess_labels.csv
```

## Важно о скачанном архиве

Для проекта используется речевой архив `Audio_Speech_Actors_01-24.zip`.
Положите его в `data/raw/`. Архив распаковывать не нужно: конвертер умеет
читать WAV-файлы прямо из ZIP. Скачать архив можно с официальной страницы RAVDESS:
<https://zenodo.org/records/1188976>.

## Требования

- Python 3.11 или новее;
- [uv](https://docs.astral.sh/uv/).

Создание виртуального окружения `.venv` и установка проекта:

```bash
uv sync
```

`uv sync` автоматически создаёт `.venv`, поэтому отдельно запускать
`python -m venv` и активировать окружение не требуется.

## Создание CSV-разметки

Основной запуск из корня проекта:

```bash
uv run ravdess-convert \
  data/raw/Audio_Speech_Actors_01-24.zip \
  --output data/processed/ravdess_labels.csv
```

Из распакованной папки:

```bash
uv run ravdess-convert data/raw/Audio_Speech_Actors_01-24 \
  --output data/processed/ravdess_labels.csv
```

Чтобы оставить только четыре простых класса:

```bash
uv run ravdess-convert data/raw/Audio_Speech_Actors_01-24.zip \
  --output data/processed/ravdess_labels.csv \
  --emotions neutral happy sad angry
```

Получаемая таблица содержит путь внутри источника, тип записи, эмоцию,
интенсивность, номер фразы, повтор, номер актёра и пол актёра. В RAVDESS эти
поля закодированы в имени файла вида `03-01-05-02-02-01-12.wav`.

## Проверка

```bash
uv run python -m unittest discover -s tests -v
```

Эти же тесты автоматически запускаются в GitHub Actions при каждом push и
для каждого pull request. Сам датасет для тестов не нужен.

## Следующий небольшой этап

После подготовки CSV проект можно завершить в минимальном объёме:

1. Извлечь простые аудиопризнаки MFCC.
2. Обучить один классификатор, например логистическую регрессию.
3. Получить вероятности классов до калибровки.
4. Применить sigmoid-калибровку.
5. Сравнить Brier score и один график калибровки до и после.

Данные нужно делить по актёрам, чтобы записи одного человека не попадали
одновременно в обучающую и тестовую выборки.
