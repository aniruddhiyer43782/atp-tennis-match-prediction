"""Create leakage-free tennis match features.

The key design rule is point-in-time correctness:
features are read before a match is processed, and player state is updated
only after that match result is known.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

BASE_ELO = 1500.0
ELO_K = 32.0
RECENT_MATCHES = 10
FATIGUE_DAYS = 14


@dataclass
class PlayerState:
    elo: float = BASE_ELO
    surface_elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: BASE_ELO))
    recent_results: deque[int] = field(default_factory=lambda: deque(maxlen=RECENT_MATCHES))
    surface_results: dict[str, deque[int]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=RECENT_MATCHES))
    )
    match_dates: deque[pd.Timestamp] = field(default_factory=deque)
    serve_points_won: deque[float] = field(default_factory=lambda: deque(maxlen=RECENT_MATCHES))
    latest_name: str = ""
    latest_rank: float = 500.0
    latest_rank_points: float = 0.0


@dataclass
class CompetitionRecord:
    competition_name: str = ""
    wins: int = 0
    matches: int = 0

    @property
    def win_rate(self) -> float:
        return 0.5 if self.matches == 0 else self.wins / self.matches


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(winner_rating: float, loser_rating: float, k: float = ELO_K) -> tuple[float, float]:
    expected_winner = expected_score(winner_rating, loser_rating)
    expected_loser = 1.0 - expected_winner
    return (
        winner_rating + k * (1.0 - expected_winner),
        loser_rating + k * (0.0 - expected_loser),
    )


def mean_or_default(values: deque[float] | deque[int], default: float = 0.5) -> float:
    if not values:
        return default
    return float(np.mean(values))


def serve_win_rate(row: pd.Series, prefix: str) -> float | None:
    first_in = row.get(f"{prefix}_1stIn")
    first_won = row.get(f"{prefix}_1stWon")
    second_won = row.get(f"{prefix}_2ndWon")
    serve_points = row.get(f"{prefix}_svpt")

    if pd.isna(serve_points) or serve_points == 0:
        return None

    won = 0.0
    for value in [first_won, second_won]:
        if not pd.isna(value):
            won += float(value)
    return won / float(serve_points)


def fatigue_count(state: PlayerState, current_date: pd.Timestamp) -> int:
    cutoff = current_date - pd.Timedelta(days=FATIGUE_DAYS)
    while state.match_dates and state.match_dates[0] < cutoff:
        state.match_dates.popleft()
    return len(state.match_dates)


def h2h_key(player_a: int, player_b: int) -> tuple[int, int]:
    return tuple(sorted((int(player_a), int(player_b))))


def competition_key(row: pd.Series) -> str:
    tourney_id = str(row.get("tourney_id", ""))
    if "-" in tourney_id:
        return tourney_id.split("-", 1)[1]
    return str(row.get("tourney_name", "")).strip().lower().replace(" ", "_")


def h2h_rate(
    h2h: dict[tuple[int, int], dict[int, int]],
    player_a: int,
    player_b: int,
) -> float:
    record = h2h.get(h2h_key(player_a, player_b))
    if not record:
        return 0.5
    wins_a = record.get(int(player_a), 0)
    wins_b = record.get(int(player_b), 0)
    total = wins_a + wins_b
    return 0.5 if total == 0 else wins_a / total


def pre_match_features(
    row: pd.Series,
    player_a_id: int,
    player_b_id: int,
    player_a_rank: float,
    player_b_rank: float,
    player_a_points: float,
    player_b_points: float,
    states: dict[int, PlayerState],
    h2h: dict[tuple[int, int], dict[int, int]],
    competition_records: dict[tuple[int, str], CompetitionRecord],
) -> dict[str, float | int | str | pd.Timestamp]:
    state_a = states[int(player_a_id)]
    state_b = states[int(player_b_id)]
    surface = row["surface"]
    match_date = row["tourney_date"]
    competition = competition_key(row)
    competition_record_a = competition_records[(int(player_a_id), competition)]
    competition_record_b = competition_records[(int(player_b_id), competition)]

    return {
        "tourney_date": match_date,
        "player_a_id": int(player_a_id),
        "player_b_id": int(player_b_id),
        "player_a_name": row["winner_name"] if player_a_id == row["winner_id"] else row["loser_name"],
        "player_b_name": row["loser_name"] if player_a_id == row["winner_id"] else row["winner_name"],
        "surface": surface,
        "tourney_level": row["tourney_level"],
        "round": row["round"],
        "draw_size": int(row["draw_size"]) if not pd.isna(row["draw_size"]) else 0,
        "round_order": int(row["round_order"]),
        "is_best_of_5": int(row["is_best_of_5"]),
        "rank_diff": float(player_a_rank) - float(player_b_rank),
        "rank_points_diff": float(player_a_points) - float(player_b_points),
        "elo_diff": state_a.elo - state_b.elo,
        "surface_elo_diff": state_a.surface_elo[surface] - state_b.surface_elo[surface],
        "h2h_win_rate": h2h_rate(h2h, player_a_id, player_b_id),
        "recent_form_diff": mean_or_default(state_a.recent_results) - mean_or_default(state_b.recent_results),
        "surface_form_diff": mean_or_default(state_a.surface_results[surface]) - mean_or_default(state_b.surface_results[surface]),
        "fatigue_diff": fatigue_count(state_a, match_date) - fatigue_count(state_b, match_date),
        "serve_dom_diff": mean_or_default(state_a.serve_points_won) - mean_or_default(state_b.serve_points_won),
        "competition_win_rate_diff": competition_record_a.win_rate - competition_record_b.win_rate,
        "competition_experience_diff": competition_record_a.matches - competition_record_b.matches,
    }


def update_states(
    row: pd.Series,
    states: dict[int, PlayerState],
    h2h: dict[tuple[int, int], dict[int, int]],
    competition_records: dict[tuple[int, str], CompetitionRecord],
) -> None:
    winner_id = int(row["winner_id"])
    loser_id = int(row["loser_id"])
    surface = row["surface"]
    match_date = row["tourney_date"]

    winner_state = states[winner_id]
    loser_state = states[loser_id]

    winner_state.latest_name = str(row["winner_name"])
    winner_state.latest_rank = float(row["winner_rank"])
    winner_state.latest_rank_points = float(row["winner_rank_points"])
    loser_state.latest_name = str(row["loser_name"])
    loser_state.latest_rank = float(row["loser_rank"])
    loser_state.latest_rank_points = float(row["loser_rank_points"])

    winner_state.elo, loser_state.elo = update_elo(winner_state.elo, loser_state.elo)
    winner_state.surface_elo[surface], loser_state.surface_elo[surface] = update_elo(
        winner_state.surface_elo[surface],
        loser_state.surface_elo[surface],
    )

    winner_state.recent_results.append(1)
    loser_state.recent_results.append(0)
    winner_state.surface_results[surface].append(1)
    loser_state.surface_results[surface].append(0)
    winner_state.match_dates.append(match_date)
    loser_state.match_dates.append(match_date)

    winner_serve = serve_win_rate(row, "w")
    loser_serve = serve_win_rate(row, "l")
    if winner_serve is not None:
        winner_state.serve_points_won.append(winner_serve)
    if loser_serve is not None:
        loser_state.serve_points_won.append(loser_serve)

    key = h2h_key(winner_id, loser_id)
    h2h[key][winner_id] += 1

    competition = competition_key(row)
    competition_name = str(row.get("tourney_name", competition))
    winner_record = competition_records[(winner_id, competition)]
    loser_record = competition_records[(loser_id, competition)]
    winner_record.competition_name = competition_name
    loser_record.competition_name = competition_name
    winner_record.wins += 1
    winner_record.matches += 1
    loser_record.matches += 1


def build_features(
    clean_matches: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[int, PlayerState],
    dict[tuple[int, int], dict[int, int]],
    dict[tuple[int, str], CompetitionRecord],
]:
    """Build symmetric, leakage-free modelling rows."""
    matches = clean_matches.sort_values(
        ["tourney_date", "tourney_id", "match_num"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    states: dict[int, PlayerState] = defaultdict(PlayerState)
    h2h: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    competition_records: dict[tuple[int, str], CompetitionRecord] = defaultdict(CompetitionRecord)
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []

    for index, row in matches.iterrows():
        winner_id = int(row["winner_id"])
        loser_id = int(row["loser_id"])

        winner_features = pre_match_features(
            row=row,
            player_a_id=winner_id,
            player_b_id=loser_id,
            player_a_rank=row["winner_rank"],
            player_b_rank=row["loser_rank"],
            player_a_points=row["winner_rank_points"],
            player_b_points=row["loser_rank_points"],
            states=states,
            h2h=h2h,
            competition_records=competition_records,
        )
        winner_features["target"] = 1
        rows.append(winner_features)

        loser_features = pre_match_features(
            row=row,
            player_a_id=loser_id,
            player_b_id=winner_id,
            player_a_rank=row["loser_rank"],
            player_b_rank=row["winner_rank"],
            player_a_points=row["loser_rank_points"],
            player_b_points=row["winner_rank_points"],
            states=states,
            h2h=h2h,
            competition_records=competition_records,
        )
        loser_features["target"] = 0
        rows.append(loser_features)

        update_states(row, states, h2h, competition_records)

        if (index + 1) % 5000 == 0:
            logging.info("Processed %s/%s matches", index + 1, len(matches))

    features = pd.DataFrame(rows)
    features = pd.get_dummies(
        features,
        columns=["surface", "tourney_level"],
        prefix=["surface", "level"],
        dtype=int,
    )

    logging.info("Feature dataset: %s rows, %s columns", *features.shape)
    return features, dict(states), dict(h2h), dict(competition_records)


def state_snapshot(states: dict[int, PlayerState]) -> pd.DataFrame:
    rows = []
    for player_id, state in states.items():
        rows.append(
            {
                "player_id": player_id,
                "player_name": state.latest_name,
                "latest_rank": state.latest_rank,
                "latest_rank_points": state.latest_rank_points,
                "elo": state.elo,
                "hard_elo": state.surface_elo["Hard"],
                "clay_elo": state.surface_elo["Clay"],
                "grass_elo": state.surface_elo["Grass"],
                "recent_form": mean_or_default(state.recent_results),
                "hard_form": mean_or_default(state.surface_results["Hard"]),
                "clay_form": mean_or_default(state.surface_results["Clay"]),
                "grass_form": mean_or_default(state.surface_results["Grass"]),
                "serve_dominance": mean_or_default(state.serve_points_won),
                "last_match_date": max(state.match_dates) if state.match_dates else pd.NaT,
                "matches_tracked_for_fatigue": len(state.match_dates),
            }
        )
    return pd.DataFrame(rows).sort_values(["latest_rank", "player_name"]).reset_index(drop=True)


def h2h_snapshot(h2h: dict[tuple[int, int], dict[int, int]]) -> pd.DataFrame:
    rows = []
    for (player_a, player_b), record in h2h.items():
        wins_a = record.get(player_a, 0)
        wins_b = record.get(player_b, 0)
        rows.append(
            {
                "player_a_id": player_a,
                "player_b_id": player_b,
                "player_a_wins": wins_a,
                "player_b_wins": wins_b,
                "total_matches": wins_a + wins_b,
            }
        )
    return pd.DataFrame(rows)


def competition_snapshot(
    competition_records: dict[tuple[int, str], CompetitionRecord],
) -> pd.DataFrame:
    rows = []
    for (player_id, key), record in competition_records.items():
        if record.matches == 0:
            continue
        rows.append(
            {
                "player_id": player_id,
                "competition_key": key,
                "competition_name": record.competition_name,
                "competition_wins": record.wins,
                "competition_matches": record.matches,
                "competition_win_rate": record.win_rate,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["competition_name", "competition_matches", "player_id"],
        ascending=[True, False, True],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-free ATP features.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DIR / "matches_clean.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "features.csv",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    clean_matches = pd.read_csv(args.input, parse_dates=["tourney_date"])
    features, states, h2h, competition_records = build_features(clean_matches)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    state_snapshot(states).to_csv(PROCESSED_DIR / "player_state_snapshot.csv", index=False)
    h2h_snapshot(h2h).to_csv(PROCESSED_DIR / "h2h_snapshot.csv", index=False)
    competition_snapshot(competition_records).to_csv(
        PROCESSED_DIR / "competition_snapshot.csv",
        index=False,
    )
    logging.info("Saved features to %s", args.output)


if __name__ == "__main__":
    main()
