# Project Plan

## Objective

Build an ATP tennis match prediction system that can estimate player A's win probability against player B using only information available before the match.

The project should demonstrate:

- Real-world data cleaning
- Leakage-free feature engineering
- Time-aware model validation
- Calibrated probability prediction
- Model explainability
- A usable Streamlit demo

## Phase 1: Data Foundation

- Load yearly ATP singles match files from `data/raw`.
- Focus on modern ATP tour-level singles from 2010-2026.
- Filter tournament levels to Grand Slams, Masters, ATP events, and Tour Finals.
- Clean dates, ranks, surfaces, rounds, and missing values.
- Save `data/processed/matches_clean.csv`.

## Phase 2: Leakage-Free Features

For each match in chronological order:

1. Read pre-match state for both players.
2. Create model features.
3. Update state after the result.

Feature groups:

- Ranking difference
- Ranking points difference
- Overall Elo difference
- Surface-specific Elo difference
- Head-to-head win rate
- Competition-specific historical win rate and match experience
- Recent form over last 10 matches
- Surface form over last 10 matches on that surface
- Fatigue based on matches in previous 14 days
- Recent serve dominance
- Ranking trend over recent match history
- Round, surface, tournament level, best-of-five context

## Phase 3: Modelling

- Use train years up to 2022.
- Use test years 2023-2026.
- Compare Logistic Regression, Random Forest, XGBoost, and LightGBM where installed.
- Tune XGBoost with Optuna using time-series validation on training data only.
- Select best model by ROC-AUC, while also reporting accuracy and F1.
- Save model and metadata.

## Phase 4: Placement Polish

- Add probability calibration. Done with sigmoid calibration and time-series cross-validation.
- Add SHAP or permutation importance. Done with SHAP explanations from the base estimator behind the calibrated model.
- Build Streamlit app.
- Add visual evaluation report.
- Write final README with results, architecture, and interview explanation.

## What Makes This Defensible

The strongest point of the project is not the specific algorithm. It is the time-aware design. The system never uses future matches when creating a feature for an earlier match, which is one of the most common hidden mistakes in sports prediction projects.
