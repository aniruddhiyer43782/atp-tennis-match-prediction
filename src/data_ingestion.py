"""Load and clean ATP singles match data.

This module turns the raw yearly Jeff Sackmann CSV files into one clean,
chronologically sorted dataset for downstream feature engineering.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR = 2026
TOUR_LEVELS = {"G", "M", "A", "F"}

KEEP_COLUMNS = [
    "tourney_id",
    "tourney_name",
    "surface",
    "draw_size",
    "tourney_level",
    "tourney_date",
    "match_num",
    "winner_id",
    "winner_name",
    "winner_hand",
    "winner_ht",
    "winner_age",
    "loser_id",
    "loser_name",
    "loser_hand",
    "loser_ht",
    "loser_age",
    "score",
    "best_of",
    "round",
    "minutes",
    "w_ace",
    "w_df",
    "w_svpt",
    "w_1stIn",
    "w_1stWon",
    "w_2ndWon",
    "w_SvGms",
    "w_bpSaved",
    "w_bpFaced",
    "l_ace",
    "l_df",
    "l_svpt",
    "l_1stIn",
    "l_1stWon",
    "l_2ndWon",
    "l_SvGms",
    "l_bpSaved",
    "l_bpFaced",
    "winner_rank",
    "winner_rank_points",
    "loser_rank",
    "loser_rank_points",
]

ROUND_ORDER = {
    "RR": 1,
    "R128": 1,
    "R96": 1,
    "R64": 2,
    "R48": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "F": 7,
    "BR": 7,
}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_raw_matches(
    raw_dir: Path = RAW_DIR,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> pd.DataFrame:
    """Load yearly ATP singles match CSVs."""
    frames: list[pd.DataFrame] = []

    for year in range(start_year, end_year + 1):
        file_path = raw_dir / f"atp_matches_{year}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing expected file: {file_path}")

        frame = pd.read_csv(file_path)
        frames.append(frame)
        logging.info("Loaded %s with %s rows", file_path.name, len(frame))

    matches = pd.concat(frames, ignore_index=True)
    logging.info("Combined raw dataset: %s rows, %s columns", *matches.shape)
    return matches


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Clean raw match rows and keep only model-ready fields."""
    cleaned = matches.copy()
    start_rows = len(cleaned)

    cleaned = cleaned[cleaned["tourney_level"].isin(TOUR_LEVELS)].copy()
    logging.info("Tour-level filter: %s -> %s rows", start_rows, len(cleaned))

    cleaned["tourney_date"] = pd.to_datetime(
        cleaned["tourney_date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )

    existing_columns = [column for column in KEEP_COLUMNS if column in cleaned.columns]
    cleaned = cleaned[existing_columns].copy()

    required_columns = ["tourney_date", "winner_id", "loser_id", "surface", "round"]
    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=required_columns)
    logging.info("Required-value drop: %s -> %s rows", before_drop, len(cleaned))

    for column in ["winner_rank", "loser_rank"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(500)

    for column in ["winner_rank_points", "loser_rank_points"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0)

    numeric_columns = [
        "draw_size",
        "winner_id",
        "loser_id",
        "best_of",
        "minutes",
        "winner_ht",
        "loser_ht",
        "winner_age",
        "loser_age",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["surface"] = cleaned["surface"].str.strip().str.title()
    cleaned["round_order"] = cleaned["round"].map(ROUND_ORDER).fillna(0).astype(int)
    cleaned["is_best_of_5"] = (cleaned["best_of"] == 5).astype(int)

    cleaned = cleaned.sort_values(
        ["tourney_date", "tourney_id", "match_num"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    logging.info("Cleaned dataset: %s rows, %s columns", *cleaned.shape)
    return cleaned


def save_clean_matches(matches: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output_path, index=False)
    logging.info("Saved cleaned data to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean ATP match CSV files.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DIR / "matches_clean.csv",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    raw_matches = load_raw_matches(args.raw_dir, args.start_year, args.end_year)
    clean = clean_matches(raw_matches)
    save_clean_matches(clean, args.output)


if __name__ == "__main__":
    main()
