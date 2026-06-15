"""Train and evaluate ATP match prediction models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_END_YEAR = 2022
TEST_START_YEAR = 2023
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_SPLITS = 5

DROP_COLUMNS = [
    "tourney_date",
    "player_a_id",
    "player_b_id",
    "player_a_name",
    "player_b_name",
    "round",
    "target",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path, parse_dates=["tourney_date"])
    logging.info("Loaded features: %s rows, %s columns", *features.shape)
    return features


def split_time_aware(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_year = features["tourney_date"].dt.year
    train = features[match_year <= TRAIN_END_YEAR].copy()
    test = features[match_year >= TEST_START_YEAR].copy()

    if train.empty or test.empty:
        raise ValueError("Time split produced empty train or test data.")

    logging.info("Train rows: %s | Test rows: %s", len(train), len(test))
    return train, test


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in DROP_COLUMNS]


def calibrate_model(model: object) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=model,
        method=CALIBRATION_METHOD,
        cv=TimeSeriesSplit(n_splits=CALIBRATION_SPLITS),
        ensemble=True,
    )


def build_models(best_xgb_params: dict | None = None) -> dict[str, object]:
    models: dict[str, object] = {
        "calibrated_logistic_regression": calibrate_model(
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            n_jobs=None,
                        ),
                    ),
                ]
            )
        ),
        "calibrated_random_forest": calibrate_model(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        ),
    }

    try:
        from xgboost import XGBClassifier

        if best_xgb_params:
            xgb_classifier = XGBClassifier(
                **best_xgb_params,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        else:
            xgb_classifier = XGBClassifier(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )

        models["calibrated_xgboost"] = calibrate_model(xgb_classifier)
    except ImportError:
        logging.warning("xgboost is not installed; skipping XGBoost.")

    try:
        from lightgbm import LGBMClassifier

        models["calibrated_lightgbm"] = calibrate_model(
            LGBMClassifier(
                n_estimators=700,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbosity=-1,
            )
        )
    except ImportError:
        logging.warning("lightgbm is not installed; skipping LightGBM.")

    return models


def tune_xgboost_optuna(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
) -> dict:
    import optuna
    from xgboost import XGBClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }

        model = XGBClassifier(**params)

        # TimeSeriesSplit inside Optuna follows the same principle as calibration:
        # tune on training data only, never touching the test set.
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_idx, val_idx in tscv.split(x_train):
            x_tr = x_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            x_val = x_train.iloc[val_idx]
            y_val = y_train.iloc[val_idx]
            model.fit(x_tr, y_tr)
            probs = model.predict_proba(x_val)[:, 1]
            scores.append(roc_auc_score(y_val, probs))

        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logging.info("Best Optuna ROC-AUC: %.4f", study.best_value)
    logging.info("Best Optuna params: %s", study.best_params)
    return study.best_params


def evaluate_model(model: object, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "f1": f1_score(y_test, predictions),
        "brier": brier_score_loss(y_test, probabilities),
        "log_loss": log_loss(y_test, probabilities),
    }


def train_all(features: pd.DataFrame) -> tuple[str, object, dict[str, dict[str, float]], list[str]]:
    train, test = split_time_aware(features)
    columns = feature_columns(features)

    x_train = train[columns]
    y_train = train["target"]
    x_test = test[columns]
    y_test = test["target"]

    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, object] = {}

    # Attempt to tune XGBoost with Optuna; if packages missing, skip tuning
    best_xgb_params = None
    try:
        logging.info("Tuning XGBoost with Optuna (50 trials)...")
        best_xgb_params = tune_xgboost_optuna(x_train, y_train, n_trials=50)
    except Exception as exc:  # ImportError, RuntimeError, etc.
        logging.warning("Optuna/XGBoost tuning skipped: %s", exc)

    for name, model in build_models(best_xgb_params=best_xgb_params).items():
        logging.info("Training %s", name)
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        results[name] = metrics
        fitted_models[name] = model
        logging.info("%s metrics: %s", name, metrics)

    best_name = max(results, key=lambda model_name: results[model_name]["roc_auc"])
    return best_name, fitted_models[best_name], results, columns


def save_artifacts(
    model_name: str,
    model: object,
    results: dict[str, dict[str, float]],
    columns: list[str],
    output_dir: Path = MODELS_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_model.joblib"
    metadata_path = output_dir / "model_metadata.json"

    joblib.dump(model, model_path)
    metadata = {
        "best_model": model_name,
        "train_end_year": TRAIN_END_YEAR,
        "test_start_year": TEST_START_YEAR,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_cv": f"TimeSeriesSplit(n_splits={CALIBRATION_SPLITS})",
        "feature_columns": columns,
        "metrics": results,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logging.info("Saved best model to %s", model_path)
    logging.info("Saved metadata to %s", metadata_path)


def save_feature_importance(model_name: str, model: object, columns: list[str]) -> None:
    """Save a simple feature-importance table for the best model."""
    importances = None

    if isinstance(model, CalibratedClassifierCV):
        fitted_estimators = [
            calibrated_classifier.estimator
            for calibrated_classifier in model.calibrated_classifiers_
        ]
    else:
        fitted_estimators = [model]

    if "logistic_regression" in model_name:
        coefficients = []
        for estimator in fitted_estimators:
            if hasattr(estimator, "named_steps"):
                coefficients.append(abs(estimator.named_steps["model"].coef_[0]))
        if coefficients:
            importances = sum(coefficients) / len(coefficients)
    else:
        tree_importances = [
            estimator.feature_importances_
            for estimator in fitted_estimators
            if hasattr(estimator, "feature_importances_")
        ]
        if tree_importances:
            importances = sum(tree_importances) / len(tree_importances)

    if importances is None:
        logging.warning("Feature importance is not available for %s", model_name)
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    importance = (
        pd.DataFrame({"feature": columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    output_path = REPORTS_DIR / "feature_importance.csv"
    importance.to_csv(output_path, index=False)
    logging.info("Saved feature importance to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ATP match prediction models.")
    parser.add_argument(
        "--features",
        type=Path,
        default=PROCESSED_DIR / "features.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    features = load_features(args.features)
    best_name, best_model, results, columns = train_all(features)
    save_artifacts(best_name, best_model, results, columns, args.output_dir)
    save_feature_importance(best_name, best_model, columns)
    logging.info("Best model by ROC-AUC: %s", best_name)


if __name__ == "__main__":
    main()
