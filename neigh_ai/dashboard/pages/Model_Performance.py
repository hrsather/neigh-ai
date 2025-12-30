import pickle
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from neigh_ai.feature_extraction.races import RaceModel


# Cache the data loading so it's only executed once
@st.cache_data
def load_race_model(load_precomputed: bool):
    race_model = RaceModel(load_precomputed=load_precomputed)
    return race_model


NO_FILTER = "No filter"
FULL_PEDIGREE_HORSES = "Full Pedigree Horses"
NO_PEDIGREE_HORSES = "No Pedigree Horses"


def main() -> None:
    st.title("Model Performance")
    st.sidebar.header("Filters")

    category = st.sidebar.selectbox(
        "Select Filter",
        [NO_FILTER, FULL_PEDIGREE_HORSES, NO_PEDIGREE_HORSES],
    )

    load_precomputed = False
    data = load_data_preds(category) if load_precomputed else get_data_preds(category)

    metrics_table(data["y_test"], data["y_pred"], data["y_train"])

    plot_preds_vs_gt(data["y_test"], data["y_pred"])

    st.subheader("Detection of Top Horses")
    percentage_table(data["y_test"], data["y_pred"])

    st.subheader("Correlation and Importance of Feature")
    correlations_table(data["X"], data["y"], data["importance"])


def load_data_preds(category: str) -> dict[str, Any]:
    with open(f"model_performance_data_{category}.pkl", "rb") as f:
        return pickle.load(f)


def filter_data(category: str, X, y):
    if category == FULL_PEDIGREE_HORSES:
        mask = (
            X[
                [
                    "avg_sire_sibling_score",
                    "avg_siredam_cousin_score",
                    "avg_siredam_auntuncle_score",
                    "avg_siresire_auntuncle_score",
                    "avg_siresire_cousin_score",
                ]
            ]
            .isna()
            .any(axis=1)
        )
        X = X[~mask]
        y = y[~mask]
    elif category == NO_PEDIGREE_HORSES:
        mask = (
            X[
                [
                    "avg_sire_sibling_score",
                    "avg_siredam_cousin_score",
                    "avg_siredam_auntuncle_score",
                    "avg_siresire_auntuncle_score",
                    "avg_siresire_cousin_score",
                ]
            ]
            .isna()
            .all(axis=1)
        )
        X = X[mask]
        y = y[mask]

    return X, y


def get_data_preds(category: str) -> dict[str, Any]:
    race_model = load_race_model(load_precomputed=True)

    X = race_model.horse_df[race_model.horse_df["race_score"].notna()][race_model.model_cols]
    y = race_model.horse_df[race_model.horse_df["race_score"].notna()]["race_score"]

    X, y = filter_data(category, X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    data = {
        "X": X,
        "y": y,
        "y_train": y_train,
        "y_pred": y_pred,
        "y_test": y_test,
        "importance": rf.feature_importances_,
    }

    with open(f"model_performance_data_{category}.pkl", "wb") as f:
        pickle.dump(data, f)

    return data


def metrics_table(y_test, y_pred, y_train) -> None:
    col1, col2 = st.columns(2)
    me_mean = mean_absolute_error(y_test, np.full_like(y_test, y_train.mean()))
    me_model = mean_absolute_error(y_test, y_pred)
    col1.metric("MAE - Model (less is better)", f"{me_model:.3f}")
    col2.metric("MAE - Mean baseline (less is better)", f"{me_mean:.3f}")


def correlations_table(X, y, importance) -> None:
    correlations = X.corrwith(y).rename("correlation_to_y")
    importances = pd.Series(importance, index=X.columns, name="rf_importance")
    feature_table = (
        pd.concat([correlations, importances], axis=1)
        .reset_index()
        .rename(
            columns={
                "index": "Feature",
                "correlation_to_y": "Correlation to Race Score",
                "rf_importance": "Model Feature Importance",
            }
        )
        .sort_values("Correlation to Race Score", ascending=False)
    )
    st.dataframe(feature_table, hide_index=True)


def percentage_table(y_test, y_pred) -> None:
    df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    # Define percentiles
    fractions = [50, 25, 10, 5, 1]

    rows = []

    for f in fractions:
        n_top = int(len(df) * (f / 100))  # number of top horses to consider

        # Sort by true score descending
        df_sorted_true = df.sort_values("y_true", ascending=False)
        top_true_idx = df_sorted_true.head(n_top).index

        # Sort by predicted score descending
        df_sorted_pred = df.sort_values("y_pred", ascending=False)
        top_pred_idx = df_sorted_pred.head(n_top).index

        # Coverage rate = fraction of actual top-n predicted in top-n
        coverage_rate = len(set(top_true_idx) & set(top_pred_idx)) / n_top if n_top > 0 else 0

        rows.append({
            "Top Fraction": f"{f}%",
            "Num Horses": n_top,
            "Coverage Rate": f"{coverage_rate:.2%}",
        })

    table = pd.DataFrame(rows)

    # Display in Streamlit
    st.dataframe(table, hide_index=True)


def plot_preds_vs_gt(y_test, y_pred) -> None:
    # Create scatter plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            marker={"color": "blue", "opacity": 0.6, "line": {"color": "black", "width": 1}},
            name="Predictions",
        )
    )

    # Add diagonal line for perfect prediction
    fig.add_trace(
        go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode="lines",
            line={"color": "red", "dash": "dash"},
            name="Perfect Prediction",
        )
    )

    fig.update_layout(
        title="Model Predictions vs Ground Truth",
        xaxis_title="True Race Score",
        yaxis_title="Predicted Race Score",
        xaxis={"range": [-2, 2]},
        yaxis={"range": [-2, 2]},
        template="plotly_white",
    )

    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
