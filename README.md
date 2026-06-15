# ATP Tennis Match Prediction

Placement-grade machine learning project that predicts the winner of ATP tennis matches using real historical data from Jeff Sackmann's `tennis_atp` dataset.

The main technical goal is not just high accuracy. The project is designed to demonstrate production-style ML thinking: clean ingestion, point-in-time feature engineering, time-aware evaluation, calibrated probabilities, and an interactive Streamlit demo.

## Why This Project Is Strong

- Uses real-world sports data instead of toy datasets.
- Avoids data leakage by computing Elo, head-to-head, form, fatigue, and serve features chronologically.
- Uses time-based train/test splits, so the model is evaluated like a real future predictor.
- Produces probabilities, not only winner labels.
- Includes explainability so the app can show why a prediction was made.

## Current Scope

Version 1 focuses on ATP tour-level singles matches from 2010 to 2026:

- `G`: Grand Slams
- `M`: Masters 1000
- `A`: Other ATP tour events
- `F`: Tour Finals / season-ending events

Challengers, Futures, doubles, Davis Cup, and amateur files are intentionally excluded from the first model to keep the target problem clean and interview-friendly.

## Current Results

The current calibrated models were trained on matches from 2010-2022 and tested on future matches from 2023-2026. Calibration uses sigmoid scaling with time-series cross-validation inside the training period only.

| Model | Accuracy | ROC-AUC | F1 | Brier | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibrated Logistic Regression | 0.650 | 0.708 | 0.650 | 0.217 | 0.622 |
| Calibrated Random Forest | 0.648 | 0.710 | 0.648 | 0.217 | 0.623 |
| Calibrated XGBoost | 0.651 | 0.711 | 0.651 | 0.217 | 0.622 |
| Calibrated LightGBM | 0.646 | 0.707 | 0.645 | 0.218 | 0.624 |

Current best model by ROC-AUC: Calibrated XGBoost.

These are honest time-split results, not random-split results. That matters because a random split would mix past and future tennis seasons and make performance look better than it really is.

The strongest baseline signals by calibrated XGBoost feature importance are:

1. Overall Elo difference
2. Ranking points difference
3. Surface-specific Elo difference
4. Ranking difference
5. Recent serve dominance difference
6. Grand Slam context

## Project Structure

```text
data/
  raw/              # Original CSV files from JeffSackmann/tennis_atp
  processed/        # Cleaned and feature-engineered datasets
models/             # Saved trained models and metadata
notebooks/          # Optional exploration notebooks
reports/
  figures/          # Evaluation plots and feature importance charts
src/
  data_ingestion.py
  feature_engineering.py
  model_trainer.py
tests/
app.py              # Streamlit app, added after model training
requirements.txt
```

## Planned Pipeline

```bash
python -m src.data_ingestion
python -m src.feature_engineering
python -m src.model_trainer
streamlit run app.py
```

## Interview Story

This project predicts player A's probability of beating player B on a given surface and tournament context. The key engineering idea is that every feature must be known before the match starts. That means the code processes matches in chronological order, reads the current player state, creates features, and only then updates Elo, H2H, form, and fatigue after the match.

That is the difference between a model that only looks good on paper and a model that behaves like a real forecasting system.

## Current Feature Set

- Draw size and round stage
- Best-of-five indicator
- Ranking and ranking-points difference
- Overall Elo difference
- Surface-specific Elo difference
- Historical head-to-head win rate
- Competition-specific historical win rate and match experience
- Recent form over last 10 matches
- Surface form over last 10 matches on the same surface
- Date-aware fatigue difference based on matches in the last 14 days
- Recent serve dominance difference
- Ranking trend difference
- Surface and tournament-level one-hot features
- Per-prediction SHAP explanations for the selected matchup
