from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


def plot_race_scatter(df: pd.DataFrame, horse_racing_api_id: str) -> None:
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
    df_selected = df[df["horse_racing_api_id"] == horse_racing_api_id]
    fig.add_trace(
        go.Scattergl(
            x=df_selected["distance_meters"],
            y=df_selected["speed"],
            mode="markers",
            marker={"color": "red", "size": 12, "line": {"color": "darkred", "width": 2}},
            name="Selected horse",
            text=df_selected["horse_racing_api_id"],
            hovertemplate="Horse: %{text}<br>Distance: %{x}<br>Speed: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Distance (meters)",
        yaxis_title="Speed (m/s)",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        template="plotly_white",
    )

    st.plotly_chart(fig, width="stretch")


def plot_hist(df: pd.DataFrame, column: str, horse_racing_api_id: str) -> None:
    # Extract the horse's value
    horse_val = df.loc[df["horse_racing_api_id"] == horse_racing_api_id, column].item()

    # Create histogram
    fig = px.histogram(
        df,
        x=column,
        nbins=50,
        title=column.replace("_", " ").title(),
        opacity=0.7,
    )

    # Add vertical line for selected horse
    fig.add_vline(
        x=horse_val,
        line_dash="dash",
        line_color="red",
        annotation_text=horse_racing_api_id,
        annotation_position="top right",
    )

    # Update layout
    fig.update_layout(xaxis_title=column, yaxis_title="Count", bargap=0.1)

    # Display in Streamlit
    st.plotly_chart(fig, width="stretch")


# Display dataframe
st.dataframe(df, selection_mode="single-row", width="stretch", key="horse_table", on_select="rerun", hide_index=True)

selected_name = "Please select a horse from the table by clicking the check box"

state = st.session_state.get("horse_table", {})
rows = state.get("selection", {}).get("rows", [])

# TODO: Add analytics page

if rows:
    idx = rows[0]
    selected_id = df.iloc[idx]["horse_racing_api_id"]
    selected_name = df.iloc[idx]["horse_name"]
    st.subheader(selected_name)

    # TODO: Percentiles
    st.text(f"Race score: {df.iloc[idx]['race_score']}")
    st.text(f"Num races: {df.iloc[idx]['num_races']}")
    st.text(f"Lifetime winnings: ${int(df.iloc[idx]['total_prize_money'])}")
    st.text(f"Average winnings per race: ${int(np.exp(df.iloc[idx]['log_avg_prize_money']))}")
    st.text(f"Avg G1 finish: {df.iloc[idx]['avg_g1_finish']}")
    st.text(f"Avg G2 finish: {df.iloc[idx]['avg_g2_finish']}")
    st.text(f"Avg G3 finish: {df.iloc[idx]['avg_g3_finish']}")

    fig = plot_race_scatter(df=race_df, horse_racing_api_id=selected_id)  # TODO: Overlay model line
    plot_hist(df, column="avg_speed_diff", horse_racing_api_id=selected_id)
    plot_hist(df, column="log_avg_prize_money", horse_racing_api_id=selected_id)

    # Sire Siblings
    st.subheader("Sire Siblings")
    selected_df = df[df["sire_id"] == df.iloc[idx]["sire_id"]]
    selected_df = selected_df[selected_df["horse_racing_api_id"] != selected_id]
    st.text(f"Max race score: {df.iloc[idx]['max_sire_sibling_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_sire_sibling_score']}")
    st.dataframe(selected_df, width="stretch", key="sibling_table", hide_index=True)

    # Siredam Cousins
    st.subheader("Siredam Cousins")
    selected_df = df[df["siredam_id"] == df.iloc[idx]["siredam_id"]]
    selected_df = selected_df[selected_df["horse_racing_api_id"] != selected_id]
    st.text(f"Max race score: {df.iloc[idx]['max_siredam_cousin_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_siredam_cousin_score']}")
    st.dataframe(selected_df, width="stretch", key="siredam_cousin_table", hide_index=True)

    # Siresire Cousins
    st.subheader("Siresire Cousins")
    selected_df = df[df["siresire_id"] == df.iloc[idx]["siresire_id"]]
    selected_df = selected_df[selected_df["horse_racing_api_id"] != selected_id]
    st.text(f"Max race score: {df.iloc[idx]['max_siresire_cousin_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_siresire_cousin_score']}")
    st.dataframe(selected_df, width="stretch", key="siresire_cousin_table", hide_index=True)

    # Siresire Aunt/Uncle
    st.subheader("Siresire Aunt/Uncle")
    selected_df = df[df["sire_id"] == df.iloc[idx]["siresire_id"]]
    st.text(f"Max race score: {df.iloc[idx]['max_siresire_auntuncle_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_siresire_auntuncle_score']}")
    st.dataframe(selected_df, width="stretch", key="siresire_auntuncle_table", hide_index=True)

    # Siredam Aunt/Uncle
    st.subheader("Siredam Aunt/Uncle")
    selected_df = df[df["dam_id"] == df.iloc[idx]["siredam_id"]]
    st.text(f"Max race score: {df.iloc[idx]['max_siredam_auntuncle_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_siredam_auntuncle_score']}")
    st.dataframe(selected_df, width="stretch", key="siredam_auntuncle_table", hide_index=True)

    # Damdam Aunt/Uncle
    st.subheader("Damdam Aunt/Uncle")
    selected_df = df[df["dam_id"] == df.iloc[idx]["damdam_id"]]
    st.text(f"Max race score: {df.iloc[idx]['max_damdam_auntuncle_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_damdam_auntuncle_score']}")
    st.dataframe(selected_df, width="stretch", key="damdam_auntuncle_table", hide_index=True)

    # Damsire Aunt/Uncle
    st.subheader("Damsire Aunt/Uncle")
    selected_df = df[df["sire_id"] == df.iloc[idx]["damsire_id"]]
    st.text(f"Max race score: {df.iloc[idx]['max_damsire_auntuncle_score']}")
    st.text(f"Avg race score: {df.iloc[idx]['avg_damsire_auntuncle_score']}")
    st.dataframe(selected_df, width="stretch", key="damsire_auntuncle_table", hide_index=True)

else:
    st.write("_Click a row above to see details here_")
