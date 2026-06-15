from __future__ import annotations

import pandas as pd
import streamlit as st

from src.predict import active_players, competition_options, load_artifacts, predict_probability, shap_explanation


st.set_page_config(page_title="ATP Match Predictor", page_icon="T", layout="wide")


@st.cache_resource
def cached_artifacts():
    return load_artifacts()


model, metadata, players, h2h, competitions = cached_artifacts()
players = active_players(players, since_year=2024)
player_names = players["player_name"].tolist()
competition_names = competition_options(competitions)

st.title("ATP Tennis Match Predictor")
st.caption("Leakage-free ATP forecasting using Elo, surface Elo, form, H2H, fatigue, serve strength, and ranking signals.")

left, right = st.columns(2)
with left:
    player_a = st.selectbox("Player A", player_names, index=0)
with right:
    default_b = 1 if len(player_names) > 1 else 0
    player_b = st.selectbox("Player B", player_names, index=default_b)

controls = st.columns(4)
with controls[0]:
    surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"])
with controls[1]:
    tournament_level = st.selectbox("Tournament", ["ATP 250 / 500", "Masters 1000", "Grand Slam", "Tour Finals"])
with controls[2]:
    round_name = st.selectbox("Round", ["R128", "R64", "R32", "R16", "QF", "SF", "F"])
with controls[3]:
    best_of = st.selectbox("Best of", [3, 5], index=1 if tournament_level == "Grand Slam" else 0)

competition_name = st.selectbox("Competition", competition_names)

match_date = st.date_input(
    "Match date",
    value=pd.Timestamp.now().date(),
    min_value=pd.Timestamp("2024-01-01").date(),
)
reference_date = pd.Timestamp(match_date)

st.markdown("**Fatigue override (optional)**")
st.caption("Leave at -1 to use automatic calculation from match history")

fatigue_a_override = st.number_input(
    f"{player_a} matches in last 14 days",
    min_value=-1,
    max_value=15,
    value=-1,
)
fatigue_b_override = st.number_input(
    f"{player_b} matches in last 14 days",
    min_value=-1,
    max_value=15,
    value=-1,
)

player_a_row = players[players["player_name"] == player_a].iloc[0]
player_b_row = players[players["player_name"] == player_b].iloc[0]

fatigue_diff: int
if fatigue_a_override >= 0 and fatigue_b_override >= 0:
    fatigue_diff = fatigue_a_override - fatigue_b_override
else:
    from src.predict import compute_fatigue

    fatigue_diff = compute_fatigue(player_a_row, reference_date) - compute_fatigue(player_b_row, reference_date)

if st.button("Predict Match", type="primary"):
    try:
        probability, h2h_text, competition_text, feature_row = predict_probability(
            model=model,
            metadata=metadata,
            players=players,
            h2h=h2h,
            competitions=competitions,
            player_a_name=player_a,
            player_b_name=player_b,
            surface=surface,
            competition_name=competition_name,
            tournament_level=tournament_level,
            round_name=round_name,
            best_of=best_of,
            reference_date=reference_date,
            fatigue_diff=fatigue_diff,
        )

        st.metric(f"{player_a} win probability", f"{probability:.1%}")
        st.progress(probability)
        st.write(h2h_text)
        st.write(competition_text)

        st.subheader("Top Matchup Signals")
        signal_columns = [
            "rank_diff",
            "rank_points_diff",
            "elo_diff",
            "surface_elo_diff",
            "h2h_win_rate",
            "recent_form_diff",
            "surface_form_diff",
            "serve_dom_diff",
            "competition_win_rate_diff",
            "competition_experience_diff",
        ]
        st.dataframe(feature_row[signal_columns].T.rename(columns={0: "value"}), use_container_width=True)

        st.subheader("Why This Prediction")
        explanation = shap_explanation(model, feature_row)
        display_explanation = explanation.copy()
        display_explanation["approx_probability_impact"] = (
            display_explanation["approx_probability_impact"] * 100
        ).round(2)
        display_explanation["shap_value"] = display_explanation["shap_value"].round(4)
        st.bar_chart(
            explanation.set_index("feature")["approx_probability_impact"],
            use_container_width=True,
        )
        st.dataframe(display_explanation, use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
