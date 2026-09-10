"""Обучение базовых классификаторов эмоций на аудиопризнаках."""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = Path("data/processed/features.csv")
OUTPUT = Path("reports/model_metrics.csv")
SEED = 42


def feature_columns(data: pd.DataFrame) -> list[str]:
    """Return numeric audio features, excluding metadata columns."""
    metadata = {
        "path", "filename", "emotion", "intensity", "statement",
        "repetition", "actor", "actor_gender",
    }
    return [column for column in data.columns if column not in metadata]


def multiclass_brier_score(y_true, probabilities, classes) -> float:
    """Multiclass Brier score: mean squared error against one-hot targets."""
    class_to_index = {label: index for index, label in enumerate(classes)}
    target = pd.get_dummies(pd.Series(y_true), dtype=float).reindex(
        columns=classes, fill_value=0.0
    ).to_numpy()
    # pandas may sort columns differently when classes are numpy strings.
    if target.shape[1] != len(classes):
        target = pd.DataFrame(
            [[float(class_to_index[label] == index) for index in range(len(classes))]
             for label in y_true]
        ).to_numpy()
    return float(((probabilities - target) ** 2).sum(axis=1).mean())


def expected_calibration_error(y_true, probabilities, classes, bins: int = 10) -> float:
    """ECE based on confidence of the top prediction."""
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (predictions == pd.Series(y_true).to_numpy()).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence >= left) & (confidence <= right if right == 1 else confidence < right)
        if mask.any():
            error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def calibration_errors(y_true, probabilities, classes, bins: int = 10) -> dict[str, float]:
    """Return fixed-bin ECE/MCE and quantile-bin (adaptive) ECE."""
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = (predictions == pd.Series(y_true).to_numpy()).astype(float)

    def bin_error(groups):
        errors = []
        for group in groups:
            if len(group):
                errors.append(abs(correct[group].mean() - confidence[group].mean()))
        return errors

    fixed_groups = [np.flatnonzero((confidence >= left) & (confidence <= right if right == 1 else confidence < right))
                    for left, right in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:])]
    fixed_errors = bin_error(fixed_groups)
    adaptive_edges = np.quantile(confidence, np.linspace(0, 1, bins + 1))
    adaptive_groups = [np.flatnonzero((confidence >= left) & (confidence <= right if i == bins - 1 else confidence < right))
                       for i, (left, right) in enumerate(zip(adaptive_edges[:-1], adaptive_edges[1:]))]
    adaptive_errors = bin_error(adaptive_groups)
    return {
        "ece": float(sum(len(group) * error for group, error in zip(fixed_groups, fixed_errors)) / len(confidence)),
        "mce": float(max(fixed_errors, default=0.0)),
        "adaptive_ece": float(sum(len(group) * error for group, error in zip(adaptive_groups, adaptive_errors)) / len(confidence)),
    }


def main() -> None:
    data = pd.read_csv(FEATURES)
    columns = feature_columns(data)

    train = data[data["actor"] <= 16]
    calibration = data[data["actor"] >= 21]
    test = data[data["actor"].between(17, 20)]
    x_train, y_train = train[columns], train["emotion"]
    x_calibration = calibration[columns]
    y_calibration = calibration["emotion"]

    group_cv = GroupKFold(n_splits=4)
    forest_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2],
        },
        scoring="f1_macro",
        cv=group_cv,
        n_jobs=-1,
    )
    forest_search.fit(x_train, y_train, groups=train["actor"])

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
        "Random Forest": forest_search.best_estimator_,
        "Gaussian Naive Bayes": GaussianNB(),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_calibration)
        probabilities = model.predict_proba(x_calibration)
        results.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_calibration, prediction),
                "macro_f1": f1_score(y_calibration, prediction, average="macro"),
                "log_loss": log_loss(
                    y_calibration, probabilities, labels=model.classes_
                ),
                "brier_score": multiclass_brier_score(
                    y_calibration, probabilities, model.classes_
                ),
                "ece": expected_calibration_error(
                    y_calibration, probabilities, model.classes_
                ),
            }
        )

    metrics = pd.DataFrame(results).round(4)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT, index=False)

    print(
        f"Обучение: {len(train)}, калибровка: {len(calibration)}, "
        f"финальный тест: {len(test)} файлов"
    )
    print("Финальный тест на актёрах 17–20 не запускался")
    print(f"Лучшие параметры Random Forest: {forest_search.best_params_}")
    print(metrics.to_string(index=False))
    print(f"\nСоздан файл: {OUTPUT}")


if __name__ == "__main__":
    main()
