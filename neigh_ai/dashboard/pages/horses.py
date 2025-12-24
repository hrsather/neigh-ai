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
    race_model = RaceModel(Path("/Users/hayden/Downloads/racing_api_horse_results_202512181056.csv"))
    return race_model


race_model = load_race_model()
race_df = race_model.race_df
df = race_model.horse_df


def plot_race_scatter(df: pd.DataFrame, horse_racing_api_id: str, model) -> None:
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
            name="Model prediction",
        )
    )
    fig.update_layout(
        xaxis_title="Distance (meters)",
        yaxis_title="Speed (m/s)",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        template="plotly_white",
    )
    st.plotly_chart(fig, width="stretch")


def plot_hist(df: pd.DataFrame, column: str, horse_id: str, title: str) -> None:
    # Extract the horse's value
    horse_val = df.loc[df["horse_racing_api_id"] == horse_id, column].item()

    # Create histogram
    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        opacity=0.7,
    )

    # Add vertical line for selected horse
    fig.add_vline(
        x=horse_val,
        line_dash="dash",
        line_color="red",
        annotation_text=horse_id,
        annotation_position="top right",
    )

    # Update layout
    fig.update_layout(xaxis_title=column, yaxis_title="Count", bargap=0.1)

    # Display in Streamlit
    with st.expander(f"{title}: {percentileofscore(df[column], df.iloc[idx][column], kind='rank'):.2f}th percentile"):
        st.plotly_chart(fig, width="stretch")


# Display dataframe
st.dataframe(df, selection_mode="single-row", width="stretch", key="horse_table", on_select="rerun", hide_index=True)

horse_name = "Please select a horse from the table by clicking the check box"

state = st.session_state.get("horse_table", {})
rows = state.get("selection", {}).get("rows", [])

# TODO: Add analytics page

if rows:
    idx = rows[0]
    horse_df = df.iloc[idx]
    st.subheader(horse_df["horse_name"])

    col1, col2 = st.columns(2)

    col1.metric(
        "Race score Percentile", f"{percentileofscore(df['race_score'], horse_df['race_score'], kind='rank'):.2f}%"
    )
    col1.metric("Num races", df.iloc[idx]["num_races"])

    col2.metric("Lifetime winnings", f"${int(horse_df['total_prize_money']):,}")
    col2.metric("Avg winnings per race", f"${int(np.exp(horse_df['log_avg_prize_money'])):,}")

    fig = plot_race_scatter(df=race_df, horse_racing_api_id=horse_df["horse_racing_api_id"], model=race_model.model)
    plot_hist(df, column="avg_speed_diff", horse_id=horse_df["horse_racing_api_id"], title="Speed")
    plot_hist(df, column="log_avg_prize_money", horse_id=horse_df["horse_racing_api_id"], title="Money per race")
    plot_hist(df, column="avg_g1_finish", horse_id=horse_df["horse_racing_api_id"], title="Average G1 Finish")
    plot_hist(df, column="avg_g2_finish", horse_id=horse_df["horse_racing_api_id"], title="Average G2 Finish")
    plot_hist(df, column="avg_g3_finish", horse_id=horse_df["horse_racing_api_id"], title="Average G3 Finish")

    # Sire Siblings
    st.subheader("Sire Siblings")
    pedigree_df = df[
        (df["sire_id"] == df.iloc[idx]["sire_id"]) & (df["horse_racing_api_id"] != df.iloc[idx]["horse_racing_api_id"])
    ]
    st.text(f"Max race score: {df.iloc[idx]['max_sire_sibling_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_sire_sibling_score']}")
    st.dataframe(pedigree_df, width="stretch", key="sibling_table", hide_index=True)

    # Siredam Cousins
    pedigree_df = df[
        (df["siredam_id"] == df.iloc[idx]["siredam_id"])
        & (df["horse_racing_api_id"] != df.iloc[idx]["horse_racing_api_id"])
    ]
    if len(pedigree_df):
        st.subheader("Siredam Cousins")
        st.text(f"Max race score: {df.iloc[idx]['max_siredam_cousin_score']}")
        st.text(f"Avg race score: {df.iloc[idx]['avg_siredam_cousin_score']}")
        st.dataframe(pedigree_df, width="stretch", key="siredam_cousin_table", hide_index=True)

    # Siresire Cousins
    pedigree_df = df[
        (df["siresire_id"] == df.iloc[idx]["siresire_id"])
        & (df["horse_racing_api_id"] != df.iloc[idx]["horse_racing_api_id"])
    ]
    if len(pedigree_df):
        st.subheader("Siresire Cousins")
        st.text(f"Max race score: {df.iloc[idx]['max_siresire_cousin_score']}")
        st.text(f"Avg race score: {df.iloc[idx]['avg_siresire_cousin_score']}")
        assert np.isclose(df.iloc[idx]["avg_siresire_cousin_score"], np.mean(pedigree_df["race_score"]))
        st.dataframe(pedigree_df, width="stretch", key="siresire_cousin_table", hide_index=True)

    # Siresire Aunt/Uncle
    pedigree_df = df[df["sire_id"] == df.iloc[idx]["siresire_id"]]
    if len(pedigree_df):
        st.subheader("Siresire Aunt/Uncle")
        st.text(f"Max race score: {df.iloc[idx]['max_siresire_auntuncle_score']}")
        st.text(f"Avg race score: {df.iloc[idx]['avg_siresire_auntuncle_score']}")
        assert np.isclose(df.iloc[idx]["avg_siresire_auntuncle_score"], np.mean(pedigree_df["race_score"]))
        st.dataframe(pedigree_df, width="stretch", key="siresire_auntuncle_table", hide_index=True)

    # Siredam Aunt/Uncle
    pedigree_df = df[df["dam_id"] == df.iloc[idx]["siredam_id"]]
    if len(pedigree_df):
        st.subheader("Siredam Aunt/Uncle")
        st.text(f"Max race score: {df.iloc[idx]['max_siredam_auntuncle_score']}")
        st.text(f"Avg race score: {df.iloc[idx]['avg_siredam_auntuncle_score']}")
        assert np.isclose(df.iloc[idx]["avg_siredam_auntuncle_score"], np.mean(pedigree_df["race_score"]))
        st.dataframe(pedigree_df, width="stretch", key="siredam_auntuncle_table", hide_index=True)

    # Damdam Aunt/Uncle
    pedigree_df = df[df["dam_id"] == df.iloc[idx]["damdam_id"]]
    if len(pedigree_df):
        st.subheader("Damdam Aunt/Uncle")
        st.text(f"Max race score: {df.iloc[idx]['max_damdam_auntuncle_score']}")
        st.text(f"Avg race score: {df.iloc[idx]['avg_damdam_auntuncle_score']}")
        assert np.isclose(df.iloc[idx]["avg_damdam_auntuncle_score"], np.mean(pedigree_df["race_score"]))
        st.dataframe(pedigree_df, width="stretch", key="damdam_auntuncle_table", hide_index=True)

    # Damsire Aunt/Uncle
    pedigree_df = df[df["sire_id"] == df.iloc[idx]["damsire_id"]]
    if len(pedigree_df):
        st.subheader("Damsire Aunt/Uncle")
        st.text(f"Max race score: {df.iloc[idx]['max_damsire_auntuncle_score']}")
        st.text(f"Avg race score: {df.iloc[idx]['avg_damsire_auntuncle_score']}")
        assert np.isclose(df.iloc[idx]["avg_damsire_auntuncle_score"], np.mean(pedigree_df["race_score"]))
        st.dataframe(pedigree_df, width="stretch", key="damsire_auntuncle_table", hide_index=True)

else:
    st.write("_Click a row above to see details here_")
