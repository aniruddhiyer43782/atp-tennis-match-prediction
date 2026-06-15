"""Prediction helpers for the Streamlit app."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

ROUND_ORDER = {
    "Round Robin": 1,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
}

SURFACE_COLUMN = {
    "Clay": "surface_Clay",
    "Grass": "surface_Grass",
    "Hard": "surface_Hard",
}

LEVEL_COLUMN = {
    "ATP 250 / 500": "level_A",
    "Tour Finals": "level_F",
    "Grand Slam": "level_G",
    "Masters 1000": "level_M",
}

DEFAULT_DRAW_SIZE = {
    "ATP 250 / 500": 32,
    "Tour Finals": 8,
    "Grand Slam": 128,
    "Masters 1000": 96,
}

SURFACE_ELO_COLUMN = {
    "Clay": "clay_elo",
    "Grass": "grass_elo",
    "Hard": "hard_elo",
}

SURFACE_FORM_COLUMN = {
    "Clay": "clay_form",
    "Grass": "grass_form",
    "Hard": "hard_form",
}


def load_artifacts() -> tuple[object, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    players = pd.read_csv(PROCESSED_DIR / "player_state_snapshot.csv", parse_dates=["last_match_date"])
    h2h = pd.read_csv(PROCESSED_DIR / "h2h_snapshot.csv")
    competitions = pd.read_csv(PROCESSED_DIR / "competition_snapshot.csv")
    return model, metadata, players, h2h, competitions


def active_players(players: pd.DataFrame, since_year: int = 2024) -> pd.DataFrame:
    active = players[players["last_match_date"].dt.year >= since_year].copy()
    active = active.dropna(subset=["player_name"])
    return active.sort_values(["latest_rank", "player_name"]).reset_index(drop=True)


def h2h_win_rate(h2h: pd.DataFrame, player_a_id: int, player_b_id: int) -> tuple[float, str]:
    low = min(player_a_id, player_b_id)
    high = max(player_a_id, player_b_id)
    record = h2h[(h2h["player_a_id"] == low) & (h2h["player_b_id"] == high)]

    if record.empty:
        return 0.5, "No previous tour-level meetings in this dataset."

    row = record.iloc[0]
    low_wins = int(row["player_a_wins"])
    high_wins = int(row["player_b_wins"])

    wins_a = low_wins if player_a_id == low else high_wins
    wins_b = high_wins if player_a_id == low else low_wins
    total = wins_a + wins_b

    if total == 0:
        return 0.5, "No previous tour-level meetings in this dataset."
    return wins_a / total, f"Historical H2H: {wins_a}-{wins_b} for selected Player A."


def competition_options(competitions: pd.DataFrame, min_matches: int = 20) -> list[str]:
    options = (
        competitions[competitions["competition_matches"] >= min_matches]["competition_name"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    return options


def competition_stats(
    competitions: pd.DataFrame,
    player_a_id: int,
    player_b_id: int,
    competition_name: str,
) -> tuple[float, int, str]:
    records = competitions[competitions["competition_name"] == competition_name]
    record_a = records[records["player_id"] == player_a_id]
    record_b = records[records["player_id"] == player_b_id]

    rate_a = 0.5 if record_a.empty else float(record_a.iloc[0]["competition_win_rate"])
    rate_b = 0.5 if record_b.empty else float(record_b.iloc[0]["competition_win_rate"])
    matches_a = 0 if record_a.empty else int(record_a.iloc[0]["competition_matches"])
    matches_b = 0 if record_b.empty else int(record_b.iloc[0]["competition_matches"])
    wins_a = 0 if record_a.empty else int(record_a.iloc[0]["competition_wins"])
    wins_b = 0 if record_b.empty else int(record_b.iloc[0]["competition_wins"])

    text = (
        f"{competition_name} record: selected Player A {wins_a}-{matches_a - wins_a}, "
        f"Player B {wins_b}-{matches_b - wins_b}."
    )
    return rate_a - rate_b, matches_a - matches_b, text


def compute_fatigue(player: pd.Series, reference_date: pd.Timestamp) -> int:
    dates_str = player.get("recent_match_dates", "")
    if not dates_str or pd.isna(dates_str):
        return 0
    dates = [pd.Timestamp(d) for d in str(dates_str).split(",") if d.strip()]
    cutoff = reference_date - pd.Timedelta(days=14)
    return sum(1 for d in dates if cutoff <= d < reference_date)


def build_feature_row(
    player_a: pd.Series,
    player_b: pd.Series,
    h2h: pd.DataFrame,
    competitions: pd.DataFrame,
    metadata: dict,
    surface: str,
    competition_name: str,
    tournament_level: str,
    round_name: str,
    best_of: int,
    reference_date: pd.Timestamp,
    fatigue_diff: int,
) -> pd.DataFrame:
    h2h_rate, _ = h2h_win_rate(h2h, int(player_a["player_id"]), int(player_b["player_id"]))
    competition_win_rate_diff, competition_experience_diff, _ = competition_stats(
        competitions,
        int(player_a["player_id"]),
        int(player_b["player_id"]),
        competition_name,
    )
    surface_elo_column = SURFACE_ELO_COLUMN[surface]
    surface_form_column = SURFACE_FORM_COLUMN[surface]

    row = {
        "draw_size": DEFAULT_DRAW_SIZE[tournament_level],
        "round_order": ROUND_ORDER[round_name],
        "is_best_of_5": int(best_of == 5),
        "rank_diff": player_a["latest_rank"] - player_b["latest_rank"],
        "rank_points_diff": player_a["latest_rank_points"] - player_b["latest_rank_points"],
        "elo_diff": player_a["elo"] - player_b["elo"],
        "surface_elo_diff": player_a[surface_elo_column] - player_b[surface_elo_column],
        "h2h_win_rate": h2h_rate,
        "recent_form_diff": player_a["recent_form"] - player_b["recent_form"],
        "surface_form_diff": player_a[surface_form_column] - player_b[surface_form_column],
        "fatigue_diff": fatigue_diff,
        "serve_dom_diff": player_a["serve_dominance"] - player_b["serve_dominance"],
        "competition_win_rate_diff": competition_win_rate_diff,
        "competition_experience_diff": competition_experience_diff,
        "surface_Clay": 0,
        "surface_Grass": 0,
        "surface_Hard": 0,
        "level_A": 0,
        "level_F": 0,
        "level_G": 0,
        "level_M": 0,
    }
    row[SURFACE_COLUMN[surface]] = 1
    row[LEVEL_COLUMN[tournament_level]] = 1

    return pd.DataFrame([row], columns=metadata["feature_columns"])


def predict_probability(
    model: object,
    metadata: dict,
    players: pd.DataFrame,
    h2h: pd.DataFrame,
    competitions: pd.DataFrame,
    player_a_name: str,
    player_b_name: str,
    surface: str,
    competition_name: str,
    tournament_level: str,
    round_name: str,
    best_of: int,
    reference_date: pd.Timestamp,
    fatigue_diff: int,
) -> tuple[float, str, str, pd.DataFrame]:
    if player_a_name == player_b_name:
        raise ValueError("Choose two different players.")

    player_a = players[players["player_name"] == player_a_name].iloc[0]
    player_b = players[players["player_name"] == player_b_name].iloc[0]
    row = build_feature_row(
        player_a,
        player_b,
        h2h,
        competitions,
        metadata,
        surface,
        competition_name,
        tournament_level,
        round_name,
        best_of,
        reference_date,
        fatigue_diff,
    )
    probability = float(model.predict_proba(row)[0, 1])
    _, h2h_text = h2h_win_rate(h2h, int(player_a["player_id"]), int(player_b["player_id"]))
    _, _, competition_text = competition_stats(
        competitions,
        int(player_a["player_id"]),
        int(player_b["player_id"]),
        competition_name,
    )
    return probability, h2h_text, competition_text, row


def base_estimator(model: object) -> object:
    """Return an interpretable fitted estimator behind a calibrated wrapper."""
    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        return calibrated_classifiers[0].estimator
    return model


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def shap_explanation(model: object, feature_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Explain a single prediction using the base tree estimator behind calibration."""
    import shap

    estimator = base_estimator(model)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(feature_row)

    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    shap_values = values[0]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[-1]
    base_probability = sigmoid(float(expected_value))

    explanation = pd.DataFrame(
        {
            "feature": feature_row.columns,
            "value": feature_row.iloc[0].to_numpy(),
            "shap_value": shap_values,
        }
    )
    explanation["direction"] = np.where(explanation["shap_value"] >= 0, "increases", "decreases")
    explanation["approx_probability_impact"] = explanation["shap_value"].apply(
        lambda value: sigmoid(float(expected_value) + float(value)) - base_probability
    )
    explanation["abs_shap_value"] = explanation["shap_value"].abs()
    return (
        explanation.sort_values("abs_shap_value", ascending=False)
        .head(top_n)
        .drop(columns=["abs_shap_value"])
        .reset_index(drop=True)
    )
