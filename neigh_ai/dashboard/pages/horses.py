import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import percentileofscore

from neigh_ai.feature_extraction.races import RaceModel

TABLE_KEY = "horse_table"


# Cache the data loading so it's only executed once
@st.cache_data
def load_race_model():
    race_model = RaceModel()
    return race_model


# TODO: Analytics page. Show performance with no pedigree and all.
def main() -> None:
    st.title("Horses")

    race_model = load_race_model()
    pprint_df(race_model.horse_df, TABLE_KEY, race_model)

    show_hide_horse_info(race_model)


def show_hide_horse_info(race_model) -> None:
    state = st.session_state.get(TABLE_KEY, {})

    if state.get("selection", {}).get("rows", []):
        display(race_model)
    else:
        st.write("_Click a row above to see details here_")


def display(race_model: RaceModel) -> None:
    horse_df = race_model.horse_df
    horse_df_selected = horse_df.iloc[0]

    st.subheader(horse_df_selected["horse_name"])

    show_tables(horse_df, horse_df_selected)

    plot_hist(horse_df, horse_df_selected, column="race_score", title="Race score")

    plot_race_scatter(race_model, horse_df_selected)

    st.subheader("Performance Features (by percentile and distribution)")
    plot_hist(horse_df, horse_df_selected=horse_df_selected, column="avg_speed_diff", title="Speed")
    plot_hist(horse_df, horse_df_selected=horse_df_selected, column="log_avg_prize_money", title="Money per race")
    plot_hist(horse_df, horse_df_selected=horse_df_selected, column="avg_g1_finish", title="Average G1 Finish")
    plot_hist(horse_df, horse_df_selected=horse_df_selected, column="avg_g2_finish", title="Average G2 Finish")
    plot_hist(horse_df, horse_df_selected=horse_df_selected, column="avg_g3_finish", title="Average G3 Finish")

    # Pedigree charts
    st.subheader("Pedigree Scores (by percentile and distribution)")

    for dam_sire in ["dam", "sire"]:
        damsire_pedigree_charts(horse_df=horse_df, horse_df_selected=horse_df_selected, dam_sire=dam_sire)

    for horse_df_key, horse_df_selected_key, new_column_name in [
        ("dam_id", "dam_id", "dam_sibling_score"),
        ("sire_id", "sire_id", "sire_sibling_score"),
        ("siredam_id", "siredam_id", "siredam_cousin_score"),
        ("siresire_id", "siresire_id", "siresire_cousin_score"),
        ("dam_id", "siredam_id", "siredam_auntuncle_score"),
        ("sire_id", "siresire_id", "siresire_auntuncle_score"),
    ]:
        pedigree_charts(
            horse_df=horse_df,
            horse_df_selected=horse_df_selected,
            horse_df_key=horse_df_key,
            horse_df_selected_key=horse_df_selected_key,
            new_column_name=new_column_name,
            race_model=race_model,
        )


def plot_race_scatter(race_model: RaceModel, horse_df_selected: pd.DataFrame) -> None:
    race_df = race_model.race_df
    fig = go.Figure()

    df_other = race_df[race_df["horse_racing_api_id"] != horse_df_selected["horse_racing_api_id"]]
    fig.add_trace(
        go.Scattergl(
            x=df_other["distance_meters"],
            y=df_other["speed"],
            mode="markers",
            marker={"color": "blue", "opacity": 0.5},
            name="Other horses",
            text=df_other["horse_racing_api_id"],
        )
    )

    # Scatter for selected horse
    race_df_selected = race_df[race_df["horse_racing_api_id"] == horse_df_selected["horse_racing_api_id"]]
    fig.add_trace(
        go.Scattergl(
            x=race_df_selected["distance_meters"],
            y=race_df_selected["speed"],
            mode="markers",
            marker={"color": "red", "size": 12, "line": {"color": "darkred", "width": 2}},
            name="Selected horse",
            text=race_df_selected["horse_racing_api_id"],
            hovertemplate="Horse: %{text}<br>",
        )
    )

    # Add predictions
    df_sorted = race_df.sort_values("distance_meters")
    X = df_sorted["distance_meters"].to_numpy().reshape(-1, 1)
    y_pred = race_model.avg_race_speed_model.predict(X)
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
        title=f"Race Speeds over a career of {len(race_df_selected)} races",
    )
    st.plotly_chart(fig, width="stretch")


def plot_hist(horse_df: pd.DataFrame, horse_df_selected: pd.DataFrame, column: str, title: str) -> None:
    fig = px.histogram(horse_df, x=column, nbins=50, opacity=0.7)

    fig.add_vline(x=horse_df_selected[column], line_dash="dash", line_color="red")

    fig.update_layout(xaxis_title=column, yaxis_title="Count", bargap=0.1)

    with st.expander(f"{title}: {percentileofscore(horse_df[column], horse_df_selected[column], kind='rank'):.0f}%"):
        st.plotly_chart(fig, width="stretch")


def pprint_df(df: pd.DataFrame, key: str, race_model: RaceModel) -> None:
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


def show_tables(horse_df: pd.DataFrame, horse_df_selected: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)

    col1.metric(
        "Race Score Percentile",
        f"{percentileofscore(horse_df['race_score'], horse_df_selected['race_score'], kind='rank'):.1f}%",
    )
    col1.metric(
        "Predicted Race Score",
        f"{percentileofscore(horse_df['race_score'], horse_df_selected['race_score_pred'], kind='rank'):.1f}%",
    )
    col2.metric("Lifetime winnings", f"${int(horse_df_selected['total_prize_money']):,}")
    col2.metric("Avg winnings per race", f"${int(np.exp(horse_df_selected['log_avg_prize_money'])):,}")


def damsire_pedigree_charts(horse_df: pd.DataFrame, horse_df_selected: pd.DataFrame, dam_sire: str) -> None:
    if pd.isna(horse_df_selected[f"race_score_{dam_sire}"]):
        return

    pct = percentileofscore(
        horse_df[f"race_score_{dam_sire}"].dropna(),
        horse_df_selected[f"race_score_{dam_sire}"],
        kind="rank",
    )

    st.text(f"{dam_sire.title()}: {horse_df_selected[f'{dam_sire}_name']} - {pct:.0f}%")


def pedigree_charts(
    horse_df: pd.DataFrame,
    horse_df_selected: pd.DataFrame,
    horse_df_key: str,
    horse_df_selected_key: str,
    new_column_name: str,
    race_model: RaceModel,
) -> None:
    relatives_df = horse_df[
        (horse_df[horse_df_key] == horse_df_selected[horse_df_selected_key])
        & (horse_df["horse_racing_api_id"] != horse_df_selected["horse_racing_api_id"])
    ]
    clean_name: str = new_column_name.replace("_", " ").title()
    if not len(relatives_df):
        return
    with st.expander(
        f"{clean_name}: "
        f"Avg={percentileofscore(horse_df['race_score'], horse_df_selected[f'avg_{new_column_name}'], kind='rank'):.0f}%, "
        f"Max={percentileofscore(horse_df['race_score'], horse_df_selected[f'max_{new_column_name}'], kind='rank'):.0f}%"
    ):
        st.subheader(clean_name)
        assert np.isclose(horse_df_selected[f"avg_{new_column_name}"], np.mean(relatives_df["race_score"]))
        pprint_df(df=relatives_df, key=new_column_name, race_model=race_model)


if __name__ == "__main__":
    main()
