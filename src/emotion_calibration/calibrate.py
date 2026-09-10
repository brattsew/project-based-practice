"""Post-calibrate emotion probabilities and evaluate only on held-out actors."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .classify import calibration_errors, feature_columns, multiclass_brier_score


FEATURES = Path("data/processed/features.csv")
METRICS = Path("reports/calibration_metrics.csv")
CONFUSION = Path("reports/confusion_matrices.csv")
ARTIFACTS = Path("reports/models")
SEED = 42


def train_models(train: pd.DataFrame, columns: list[str]) -> dict[str, object]:
    """Fit base models on actors 1-16 only."""
    x_train, y_train = train[columns], train["emotion"]
    search = GridSearchCV(
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"n_estimators": [200, 400], "max_depth": [None, 10, 20],
         "min_samples_leaf": [1, 2]},
        scoring="f1_macro", cv=GroupKFold(n_splits=4), n_jobs=-1,
    )
    search.fit(x_train, y_train, groups=train["actor"])
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        ),
        "Random Forest": search.best_estimator_,
        "Gaussian Naive Bayes": GaussianNB(),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def split_actor_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, calibration and final-test groups without overlap."""
    return (
        data[data["actor"].between(1, 16)].copy(),
        data[data["actor"].between(21, 24)].copy(),
        data[data["actor"].between(17, 20)].copy(),
    )


class TemperatureScaler:
    """Multiclass temperature scaling fitted by minimizing calibration NLL."""

    def fit(self, probabilities, y_true, classes):
        self.classes_ = np.asarray(classes)
        targets = np.array([np.flatnonzero(self.classes_ == value)[0] for value in y_true])
        probabilities = np.clip(np.asarray(probabilities), 1e-7, 1.0)
        logits = np.log(probabilities)

        def objective(log_temperature):
            scaled = logits / np.exp(log_temperature)
            scaled -= scaled.max(axis=1, keepdims=True)
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return float(-log_probs[np.arange(len(targets)), targets].mean())

        result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
        self.temperature_ = float(np.exp(result.x))
        return self

    def predict_proba(self, probabilities):
        logits = np.log(np.clip(np.asarray(probabilities), 1e-7, 1.0)) / self.temperature_
        logits -= logits.max(axis=1, keepdims=True)
        result = np.exp(logits)
        return result / result.sum(axis=1, keepdims=True)


def evaluate(name: str, method: str, y_true, prediction, probabilities, classes) -> dict:
    errors = calibration_errors(y_true, probabilities, classes)
    return {
        "model": name, "method": method,
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(y_true, prediction, average="macro"),
        "log_loss": log_loss(y_true, probabilities, labels=classes),
        "brier_score": multiclass_brier_score(y_true, probabilities, classes),
        **errors,
    }


def calibrate_model(model, x_calibration, y_calibration, method: str = "sigmoid"):
    """Fit a sigmoid calibrator on a separately held-out actor group."""
    # scikit-learn 1.7+ replaced cv="prefit" with FrozenEstimator. Keep the
    # old branch so the project remains usable with the versions from the
    # original README as well.
    try:
        from sklearn.frozen import FrozenEstimator
    except ImportError:
        try:
            calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
        except TypeError:
            calibrated = CalibratedClassifierCV(
                estimator=model, method=method, cv="prefit"
            )
    else:
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method=method
        )
    calibrated.fit(x_calibration, y_calibration)
    return calibrated


def run(features_path: Path = FEATURES, metrics_path: Path = METRICS) -> pd.DataFrame:
    data = pd.read_csv(features_path)
    columns = feature_columns(data)
    train, calibration, test = split_actor_data(data)
    if train.empty or calibration.empty or test.empty:
        raise ValueError("Ожидались непустые группы актёров 1–16, 21–24 и 17–20")
    models = train_models(train, columns)
    rows, matrix_rows, emotion_rows = [], [], []
    for name, model in models.items():
        x_cal, y_cal = calibration[columns], calibration["emotion"]
        x_test, y_test = test[columns], test["emotion"]
        before = model.predict_proba(x_test)
        before_pred = model.classes_[before.argmax(axis=1)]
        rows.append(evaluate(name, "none", y_test, before_pred, before, model.classes_))
        calibrators = {
            "sigmoid": calibrate_model(model, x_cal, y_cal, method="sigmoid"),
            "isotonic": calibrate_model(model, x_cal, y_cal, method="isotonic"),
        }
        temperature = TemperatureScaler().fit(model.predict_proba(x_cal), y_cal, model.classes_)
        for method, calibrator in [("sigmoid", calibrators["sigmoid"]), ("isotonic", calibrators["isotonic"]), ("temperature", temperature)]:
            if method == "temperature":
                after = calibrator.predict_proba(before)
                classes = model.classes_
            else:
                after = calibrator.predict_proba(x_test)
                classes = calibrator.classes_
            after_pred = classes[after.argmax(axis=1)]
            metrics_row = evaluate(name, method, y_test, after_pred, after, classes)
            rows.append(metrics_row)
            print(
                f"{name} | {method}: Log Loss={metrics_row['log_loss']:.4f}, "
                f"Brier={metrics_row['brier_score']:.4f}, ECE={metrics_row['ece']:.4f}"
            )
            for emotion in classes:
                index = list(classes).index(emotion)
                target = (y_test.to_numpy() == emotion).astype(float)
                emotion_rows.append({"model": name, "method": method, "emotion": emotion,
                                     "brier_one_vs_rest": float(np.mean((after[:, index] - target) ** 2))})
        labels = list(model.classes_)
        for phase, predictions in (("none", before_pred), ("sigmoid", calibrators["sigmoid"].predict(x_test)),
                                   ("isotonic", calibrators["isotonic"].predict(x_test)),
                                   ("temperature", model.classes_[TemperatureScaler().fit(model.predict_proba(x_cal), y_cal, model.classes_).predict_proba(before).argmax(axis=1)])):
            matrix = confusion_matrix(y_test, predictions, labels=labels)
            for actual, row in zip(labels, matrix):
                for predicted, value in zip(labels, row):
                    matrix_rows.append({"model": name, "phase": phase,
                                        "actual": actual, "predicted": predicted,
                                        "count": int(value)})
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ARTIFACTS / f"{name.lower().replace(' ', '_')}_none.joblib")
        for method, calibrator in calibrators.items():
            joblib.dump(calibrator, ARTIFACTS / f"{name.lower().replace(' ', '_')}_{method}.joblib")
        joblib.dump(temperature, ARTIFACTS / f"{name.lower().replace(' ', '_')}_temperature.joblib")

    result = pd.DataFrame(rows)
    baseline = result[result.method == "none"].set_index("model")
    for metric in ("accuracy", "macro_f1", "log_loss", "brier_score", "ece", "mce", "adaptive_ece"):
        result[f"delta_{metric}"] = result.apply(
            lambda row: row[metric] - baseline.loc[row.model, metric], axis=1
        )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    result.round(6).to_csv(metrics_path, index=False)
    pd.DataFrame(matrix_rows).to_csv(CONFUSION, index=False)
    pd.DataFrame(emotion_rows).to_csv(metrics_path.parent / "calibration_by_emotion.csv", index=False)
    return result


def main() -> None:
    result = run()
    print("Калибраторы обучены на актёрах 21–24; финальная оценка выполнена только на 17–20.")
    print(result.round(4).to_string(index=False))
    print(f"Создан файл: {METRICS}")


if __name__ == "__main__":
    main()
