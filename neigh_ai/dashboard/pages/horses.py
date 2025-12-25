from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import percentileofscore

from neigh_ai.feature_extraction.races import RaceModel

st.title("Horses")

st.sidebar.header("Filters")
category = st.sidebar.selectbox(
    "Select category",
    ["A", "B", "C"],
)


# Cache the data loading so it's only executed once
@st.cache_data
def load_race_model():
    race_model = RaceModel(Path("/Users/hayden/Downloads/racing_api_horse_results_202512181056.csv"), debug=True)
    return race_model


race_model = load_race_model()
race_df = race_model.race_df
df = race_model.horse_df


def plot_race_scatter(df: pd.DataFrame, horse_racing_api_id: str, model, num_races: int) -> None:
    fig = go.Figure()

    df_other = df[df["horse_racing_api_id"] != horse_racing_api_id]
    fig.add_trace(
        go.Scattergl(
            x=df_other["distance_meters"],
            y=df_other["speed"],
            mode="markers",
            marker={"color": "blue", "opacity": 0.5},
            name="Other horses",
            text=df_other["horse_racing_api_id"],
            hovertemplate="Horse: %{text}<br>Distance: %{x}<br>Speed: %{y}<extra></extra>",
        )
    )

    # Scatter for selected horse
    horse_df = df[df["horse_racing_api_id"] == horse_racing_api_id]
    fig.add_trace(
        go.Scattergl(
            x=horse_df["distance_meters"],
            y=horse_df["speed"],
            mode="markers",
            marker={"color": "red", "size": 12, "line": {"color": "darkred", "width": 2}},
            name="Selected horse",
            text=horse_df["horse_racing_api_id"],
            hovertemplate="Horse: %{text}<br>Distance: %{x}<br>Speed: %{y}<extra></extra>",
        )
    )

    # Add predictions
    df_sorted = df.sort_values("distance_meters")
    X = df_sorted["distance_meters"].to_numpy().reshape(-1, 1)
    y_pred = model.predict(X)
    fig.add_trace(
        go.Scattergl(
            x=df_sorted["distance_meters"],
            y=y_pred,
            mode="lines",
            line={"color": "white", "width": 4},
            name="Average Speed",
        )
    )
    fig.update_layout(
        xaxis_title="Distance (meters)",
        yaxis_title="Speed (m/s)",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        template="plotly_white",
        title=f"Race Speeds over a career of {num_races} races",
    )
    st.plotly_chart(fig, width="stretch")


def plot_hist(df: pd.DataFrame, column: str, horse_id: str, title: str) -> None:
    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        opacity=0.7,
    )

    fig.add_vline(
        x=df.loc[df["horse_racing_api_id"] == horse_id, column].item(),
        line_dash="dash",
        line_color="red",
        annotation_text=horse_id,
        annotation_position="top right",
    )

    fig.update_layout(xaxis_title=column, yaxis_title="Count", bargap=0.1)

    with st.expander(f"{title}: {percentileofscore(df[column], df.iloc[idx][column], kind='rank'):.0f}%"):
        st.plotly_chart(fig, width="stretch")


def pprint_df(df: pd.DataFrame, key: str) -> None:
    st.dataframe(
        df[
            [
                *["horse_name", "dam_name", "sire_name", "race_score"],
                *race_model.model_cols,
                *race_model.ps_features,
                "race_score_pred_diff",
            ]
        ],
        selection_mode="single-row",
        width="stretch",
        key=key,
        on_select="rerun",
        hide_index=True,
    )


table_key = "horse_table"
pprint_df(df, table_key)

horse_name = "Please select a horse from the table by clicking the check box"

state = st.session_state.get(table_key, {})
rows = state.get("selection", {}).get("rows", [])

# TODO: Analytics page. Show performance with no pedigree and all.

if rows:
    idx = rows[0]
    horse_df = df.iloc[idx]
    st.subheader(horse_df["horse_name"])

    col1, col2 = st.columns(2)

    col1.metric(
        "Race Score Percentile", f"{percentileofscore(df['race_score'], horse_df['race_score'], kind='rank'):.1f}%"
    )
    col1.metric(
        "Predicted Race Score", f"{percentileofscore(df['race_score'], horse_df['race_score_pred'], kind='rank'):.1f}%"
    )

    col2.metric("Lifetime winnings", f"${int(horse_df['total_prize_money']):,}")
    col2.metric("Avg winnings per race", f"${int(np.exp(horse_df['log_avg_prize_money'])):,}")

    plot_hist(df, column="race_score", horse_id=horse_df["horse_racing_api_id"], title="Race score")

    fig = plot_race_scatter(
        df=race_df,
        horse_racing_api_id=horse_df["horse_racing_api_id"],
        model=race_model.avg_race_speed_model,
        num_races=int(df.iloc[idx]["num_races"]),
    )

    st.subheader("Performance Features (by percentile and distribution)")
    plot_hist(df, column="avg_speed_diff", horse_id=horse_df["horse_racing_api_id"], title="Speed")
    plot_hist(df, column="log_avg_prize_money", horse_id=horse_df["horse_racing_api_id"], title="Money per race")
    plot_hist(df, column="avg_g1_finish", horse_id=horse_df["horse_racing_api_id"], title="Average G1 Finish")
    plot_hist(df, column="avg_g2_finish", horse_id=horse_df["horse_racing_api_id"], title="Average G2 Finish")
    plot_hist(df, column="avg_g3_finish", horse_id=horse_df["horse_racing_api_id"], title="Average G3 Finish")

    # Pedigree charts
    st.subheader("Pedigree Scores (by percentile and distribution)")

    if pd.notna(df.iloc[idx]["race_score_sire"]):
        st.text(
            f"Sire: {df.iloc[idx]['sire_name']} - {percentileofscore(df['race_score_sire'].dropna(), df.iloc[idx]['race_score_sire'], kind='rank'):.0f}%"
        )
    if pd.notna(df.iloc[idx]["race_score_dam"]):
        st.text(
            f"Dam: {df.iloc[idx]['dam_name']} - {percentileofscore(df['race_score_dam'].dropna(), df.iloc[idx]['race_score_dam'], kind='rank'):.0f}%"
        )

    def pedigree_charts(pedigree_df: pd.DataFrame, column: str) -> None:
        clean_name: str = column.replace("_", " ").title()
        if not len(pedigree_df):
            return
        with st.expander(
            f"{clean_name}: Avg={percentileofscore(df['race_score'], df.iloc[idx][f'avg_{column}'], kind='rank'):.0f}%, Max={percentileofscore(df['race_score'], df.iloc[idx][f'max_{column}'], kind='rank'):.0f}%"
        ):
            st.subheader(clean_name)
            assert np.isclose(df.iloc[idx][f"avg_{column}"], np.mean(pedigree_df["race_score"]))
            pprint_df(pedigree_df, key=column)

    # Dam Siblings
    pedigree_charts(
        df[
            (df["dam_id"] == df.iloc[idx]["dam_id"])
            & (df["horse_racing_api_id"] != df.iloc[idx]["horse_racing_api_id"])
        ],
        "dam_sibling_score",
    )
    # Sire Siblings
    pedigree_charts(
        df[
            (df["sire_id"] == df.iloc[idx]["sire_id"])
            & (df["horse_racing_api_id"] != df.iloc[idx]["horse_racing_api_id"])
        ],
        "sire_sibling_score",
    )
    # Siredam Cousins
    pedigree_charts(
        df[
            (df["siredam_id"] == df.iloc[idx]["siredam_id"])
            & (df["horse_racing_api_id"] != df.iloc[idx]["horse_racing_api_id"])
        ],
        "siredam_cousin_score",
    )
    # Siresire Cousins
    pedigree_charts(
        df[
            (df["siresire_id"] == df.iloc[idx]["siresire_id"])
            & (df["horse_racing_api_id"] != df.iloc[idx]["horse_racing_api_id"])
        ],
        "siresire_cousin_score",
    )
    # Siresire Aunt/Uncle
    pedigree_charts(df[df["sire_id"] == df.iloc[idx]["siresire_id"]], "siresire_auntuncle_score")
    # Siredam Aunt/Uncle
    pedigree_charts(df[df["dam_id"] == df.iloc[idx]["siredam_id"]], "siredam_auntuncle_score")

else:
    st.write("_Click a row above to see details here_")
