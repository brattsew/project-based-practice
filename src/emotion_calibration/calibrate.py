"""
Посткалибровка вероятностей эмоционального классификатора (Задача 5).

Логика ровно та, что описана в README проекта:
1. Актёры 1-16  -> обучение базовой модели (как в classify.py)
2. Актёры 21-24 -> обучение калибратора поверх уже обученной модели
3. Актёры 17-20 -> финальный тест "до" и "после" калибровки (запускается один раз)

Положи этот файл в src/emotion_calibration/calibrate.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import label_binarize

DATA_PATH = Path("data/processed/features.csv")
REPORT_PATH = Path("reports/calibration_metrics.csv")

# Колонки, которые НЕ являются аудиопризнаками.
# Открой features.csv и сверь с реальными названиями колонок —
# если что-то называется иначе, поправь список здесь.
NON_FEATURE_COLUMNS = [
    "path",
    "filename",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "actor",
    "actor_gender",
]

TRAIN_ACTORS = list(range(1, 17))        # 1-16
CALIBRATION_ACTORS = list(range(21, 25))  # 21-24
TEST_ACTORS = list(range(17, 21))         # 17-20


def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA_PATH)
    feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    return df, feature_columns


def split_by_actors(df: pd.DataFrame, actors: list[int]) -> pd.DataFrame:
    return df[df["actor"].isin(actors)]


def multiclass_brier_score(y_true_labels, y_proba, classes) -> float:
    """Brier score для многоклассовой задачи (one-vs-rest, усреднённый)."""
    y_true_bin = label_binarize(y_true_labels, classes=classes)
    return float(np.mean(np.sum((y_proba - y_true_bin) ** 2, axis=1)))


def evaluate(model, X, y_true, classes, stage_name: str) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return {
        "stage": stage_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "log_loss": log_loss(y_true, y_proba, labels=classes),
        "brier_score": multiclass_brier_score(y_true, y_proba, classes),
    }


def main() -> None:
    df, feature_columns = load_data()

    train_df = split_by_actors(df, TRAIN_ACTORS)
    calib_df = split_by_actors(df, CALIBRATION_ACTORS)
    test_df = split_by_actors(df, TEST_ACTORS)

    X_train, y_train = train_df[feature_columns], train_df["emotion"]
    X_calib, y_calib = calib_df[feature_columns], calib_df["emotion"]
    X_test, y_test = test_df[feature_columns], test_df["emotion"]

    classes = sorted(df["emotion"].unique())

    # 1. Базовая модель (Random Forest — лучшая по текущим результатам в README)
    base_model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    base_model.fit(X_train, y_train)

    results = [evaluate(base_model, X_test, y_test, classes, "before_calibration")]

    # 2. Калибровка на актёрах 21-24.
    # cv="prefit" означает: модель уже обучена, CalibratedClassifierCV
    # только "перекалибровывает" её вероятности на новых данных.
    #
    # Если твоя версия sklearn (>=1.6) ругается на cv="prefit" как deprecated,
    # замени две строки ниже на:
    #   from sklearn.frozen import FrozenEstimator
    #   calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    from sklearn.frozen import FrozenEstimator

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated_model.fit(X_calib, y_calib)

    results.append(evaluate(calibrated_model, X_test, y_test, classes, "after_calibration"))

    result_df = pd.DataFrame(results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(REPORT_PATH, index=False)

    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()