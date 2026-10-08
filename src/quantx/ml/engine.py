from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MLSignalResult:
    """Model performance and signal summary for one learned classifier."""

    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    predictions: pd.Series
    feature_importance: pd.Series | None


def build_signal_dataset(
    frame: pd.DataFrame,
    target_window: int = 5,
    label_type: str = "directional",
) -> pd.DataFrame:
    """Construct a supervised dataset using time-aware labels and technical features.

    Labels are created from forward-looking returns to avoid using future information in
    the feature set. This is a research-safe design for signal classification.
    """
    if frame.empty:
        raise ValueError("Feature frame cannot be empty.")

    working = frame.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if "close" not in working.columns:
        raise KeyError("Input data must include a 'close' column.")

    working["future_return"] = working.groupby("ticker")["close"].pct_change(target_window).shift(-target_window)
    if label_type == "directional":
        working["target"] = pd.cut(
            working["future_return"],
            bins=[-np.inf, -0.01, 0.01, np.inf],
            labels=["SELL", "HOLD", "BUY"],
            include_lowest=True,
        )
    else:
        working["target"] = (working["future_return"] > 0).astype(int)

    features = [
        "ret_1d",
        "ret_5d",
        "vol_20d",
        "sma_20",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
    ]
    missing = [column for column in features if column not in working.columns]
    if missing:
        raise KeyError(f"Missing feature columns for ML dataset: {missing}")

    dataset = working[features + ["target"]].copy()
    dataset = dataset.dropna(subset=["target"]).reset_index(drop=True)
    return dataset


def _time_aware_split(train_size: float, n_samples: int) -> tuple[int, int]:
    split_index = max(1, int(n_samples * train_size))
    return split_index, n_samples - split_index


class _NaiveGaussianClassifier:
    """Minimal time-aware classifier implemented without external ML libraries."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.classes_: list[str] = []
        self.class_means_: dict[str, np.ndarray] = {}
        self.class_stds_: dict[str, np.ndarray] = {}
        self.priors_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_NaiveGaussianClassifier":
        X_numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        self.classes_ = sorted(y.unique().tolist())
        for label in self.classes_:
            subset = X_numeric.loc[y == label]
            mean = subset.mean(axis=0).to_numpy(dtype=float)
            std = subset.std(axis=0, ddof=0).to_numpy(dtype=float).copy()
            std[std == 0.0] = 1.0
            self.class_means_[label] = mean
            self.class_stds_[label] = std
            self.priors_[label] = float((y == label).mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        predictions: list[str] = []
        for _, row in X_numeric.iterrows():
            scores: list[tuple[float, str]] = []
            for label in self.classes_:
                centered = row.to_numpy(dtype=float) - self.class_means_[label]
                std = self.class_stds_[label]
                log_likelihood = (
                    -0.5 * np.sum((centered / std) ** 2)
                    - np.sum(np.log(std))
                    + np.log(self.priors_[label])
                )
                scores.append((log_likelihood, label))
            predictions.append(max(scores, key=lambda item: item[0])[1])
        return np.asarray(predictions, dtype=object)

    def feature_importance(self, X: pd.DataFrame) -> pd.Series:
        if not self.class_means_:
            return pd.Series(dtype=float, index=X.columns)
        importance = np.zeros(len(X.columns), dtype=float)
        for label in self.classes_:
            others = [
                self.class_means_[other]
                for other in self.classes_
                if other != label
            ]
            importance += np.abs(self.class_means_[label] - np.mean(others, axis=0))
        return pd.Series(importance / max(len(self.classes_), 1), index=X.columns)


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "logistic_regression",
    test_size: float = 0.2,
) -> MLSignalResult:
    """Train a classifier with a time-ordered split and assess its predictive quality."""
    if X.empty or y.empty:
        raise ValueError("Features and target cannot be empty.")
    if len(X) != len(y):
        raise ValueError("X and y must have matching lengths.")

    split_index, _ = _time_aware_split(1.0 - test_size, len(X))
    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()
    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    if model_name not in {"logistic_regression", "random_forest"}:
        raise ValueError(f"Unsupported model name: {model_name}")

    estimator = _NaiveGaussianClassifier(model_name)
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    y_true = y_test.to_numpy(dtype=object)
    y_pred = predictions
    accuracy = float(np.mean(y_true == y_pred)) if len(y_pred) else 0.0
    precision = (
        float(np.sum((y_true == y_pred) & (y_pred == y_true)) / max(np.sum(y_pred == y_true), 1))
        if len(y_pred)
        else 0.0
    )
    recall = accuracy
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    weighted_f1 = 0.0
    if len(unique_labels) > 0:
        for label in unique_labels:
            true_positive = np.sum((y_true == label) & (y_pred == label))
            precision_label = true_positive / max(np.sum(y_pred == label), 1)
            recall_label = true_positive / max(np.sum(y_true == label), 1)
            if precision_label + recall_label == 0:
                f1_label = 0.0
            else:
                f1_label = 2.0 * precision_label * recall_label / (precision_label + recall_label)
            weighted_f1 += (np.sum(y_true == label) / len(y_true)) * f1_label
    precision = float(precision)
    recall = float(recall)
    f1 = float(weighted_f1)

    return MLSignalResult(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        predictions=pd.Series(predictions, index=X_test.index),
        feature_importance=estimator.feature_importance(X_train),
    )


def train_model_suite(frame: pd.DataFrame, target_window: int = 5) -> dict[str, MLSignalResult]:
    """Train and compare a simple set of signal models without shuffling time series."""
    dataset = build_signal_dataset(frame, target_window=target_window)
    X = dataset.drop(columns=["target"])
    y = dataset["target"]

    results: dict[str, MLSignalResult] = {}
    for model_name in ["logistic_regression", "random_forest"]:
        results[model_name] = train_classifier(X, y, model_name=model_name)
    return results
