"""Извлечение аудиопризнаков из речевых записей RAVDESS."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import librosa
import numpy as np
import pandas as pd


ARCHIVE = Path("data/raw/Audio_Speech_Actors_01-24.zip")
LABELS = Path("data/processed/ravdess_labels.csv")
OUTPUT = Path("data/processed/features.csv")


def extract_features(
    archive_path: Path = ARCHIVE,
    labels_path: Path = LABELS,
    output_path: Path = OUTPUT,
) -> pd.DataFrame:
    """Вычисляет 62 компактных аудиопризнака для каждой записи."""
    labels = pd.read_csv(labels_path)
    rows = []

    with ZipFile(archive_path) as archive:
        for row_number, label in labels.iterrows():
            audio = BytesIO(archive.read(label["path"]))
            signal, sample_rate = librosa.load(audio, sr=16_000, mono=True)
            mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
            delta = librosa.feature.delta(mfcc)
            delta_delta = librosa.feature.delta(mfcc, order=2)

            features = {}
            for index in range(13):
                coefficient = index + 1
                features[f"mfcc_{coefficient:02d}_mean"] = np.mean(mfcc[index])
                features[f"mfcc_{coefficient:02d}_std"] = np.std(mfcc[index])
            for name, values in {"delta": delta, "delta2": delta_delta}.items():
                for index in range(13):
                    features[f"{name}_{index + 1:02d}_std"] = np.std(values[index])

            spectral_features = {
                "rms": librosa.feature.rms(y=signal),
                "zero_crossing_rate": librosa.feature.zero_crossing_rate(signal),
                "spectral_centroid": librosa.feature.spectral_centroid(
                    y=signal, sr=sample_rate
                ),
                "spectral_bandwidth": librosa.feature.spectral_bandwidth(
                    y=signal, sr=sample_rate
                ),
                "spectral_rolloff": librosa.feature.spectral_rolloff(
                    y=signal, sr=sample_rate
                ),
            }
            for name, values in spectral_features.items():
                features[f"{name}_mean"] = np.mean(values)
                features[f"{name}_std"] = np.std(values)
            rows.append(features)

            if (row_number + 1) % 200 == 0:
                print(f"Обработано файлов: {row_number + 1}/{len(labels)}")

    result = pd.concat([labels, pd.DataFrame(rows)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    features = extract_features()
    print(f"Создан файл {OUTPUT}: {len(features)} строк")
