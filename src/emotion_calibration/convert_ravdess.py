"""Создание CSV-разметки из речевого архива RAVDESS."""

from pathlib import Path
from zipfile import ZipFile

import pandas as pd


ARCHIVE = Path("data/raw/Audio_Speech_Actors_01-24.zip")
OUTPUT = Path("data/processed/ravdess_labels.csv")

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

INTENSITIES = {"01": "normal", "02": "strong"}
STATEMENTS = {
    "01": "kids_are_talking_by_the_door",
    "02": "dogs_are_sitting_by_the_door",
}


def parse_filename(path: str) -> dict:
    """Расшифровывает метаданные, записанные в имени файла RAVDESS."""
    filename = Path(path).name
    codes = filename.removesuffix(".wav").split("-")
    if len(codes) != 7 or codes[:2] != ["03", "01"]:
        raise ValueError(f"Ожидался файл речевого аудио, получен: {filename}")

    _, _, emotion, intensity, statement, repetition, actor = codes
    actor_number = int(actor)
    return {
        "path": path,
        "filename": filename,
        "emotion": EMOTIONS[emotion],
        "intensity": INTENSITIES[intensity],
        "statement": STATEMENTS[statement],
        "repetition": int(repetition),
        "actor": actor_number,
        "actor_gender": "male" if actor_number % 2 else "female",
    }


def convert(archive_path: Path = ARCHIVE, output_path: Path = OUTPUT) -> pd.DataFrame:
    """Читает имена WAV-файлов из ZIP и сохраняет разметку в CSV."""
    with ZipFile(archive_path) as archive:
        wav_paths = sorted(name for name in archive.namelist() if name.endswith(".wav"))

    labels = pd.DataFrame(parse_filename(path) for path in wav_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)
    return labels


def main() -> None:
    labels = convert()
    print(f"Создан файл {OUTPUT}: {len(labels)} строк")
