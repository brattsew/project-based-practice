"""Обучение трёх простых классификаторов эмоций на аудиопризнаках."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = Path("data/processed/features.csv")
OUTPUT = Path("reports/model_metrics.csv")


def main() -> None:
    data = pd.read_csv(FEATURES)
    feature_columns = [
        column
        for column in data
        if column
        not in {
            "path",
            "filename",
            "emotion",
            "intensity",
            "statement",
            "repetition",
            "actor",
            "actor_gender",
        }
    ]

    train = data[data["actor"] <= 16]
    calibration = data[data["actor"] >= 21]
    test = data[data["actor"].between(17, 20)]
    x_train, y_train = train[feature_columns], train["emotion"]
    x_calibration = calibration[feature_columns]
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
