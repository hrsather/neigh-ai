import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from neigh_ai.feature_extraction.races import RaceModel


def main() -> None:
    race_model = RaceModel()

    recalculate = True

    if recalculate:
        yearling_df = pd.read_csv("/Users/hayden/Downloads/vw_yearlings_202509011838.csv")
        yearling_df = (
            yearling_df
            # Merge sire id
            .merge(
                yearling_df[["id", "name"]].rename(columns={"id": "sire_id", "name": "sire_name"}),
                left_on="sire",
                right_on="sire_name",
                how="left",
            )
            # Merge dam id
            .merge(
                yearling_df[["id", "name"]].rename(columns={"id": "dam_id", "name": "dam_name"}),
                left_on="dam",
                right_on="dam_name",
                how="left",
            )
            # Assign scores using the RaceModel
            .assign(
                sire_score=lambda df: df["sire_id"].apply(lambda x: str(race_model.get_score_from_id(x))),
                dam_score=lambda df: df["dam_id"].apply(lambda x: str(race_model.get_score_from_id(x))),
                self_score=lambda df: df["id"].apply(lambda x: str(race_model.get_score_from_id(x))),
            )
        )
        yearling_df.to_pickle("pickles/yearling_df.pkl")
    else:
        yearling_df = pd.read_pickle("pickles/yearling_df.pkl")

    print(yearling_df["sire_id"])
    return
    print(len(yearling_df))
    yearling_df = yearling_df[yearling_df["self_score"] != 0]
    print(len(yearling_df))

    plt.scatter(yearling_df["dam_score"], yearling_df["self_score"])
    plt.show()
    plt.scatter(yearling_df["sire_score"], yearling_df["self_score"])
    plt.show()

    X = yearling_df[["dam_score", "sire_score"]].fillna(race_model.new_score)

    y = yearling_df["self_score"]

    # model = RandomForestRegressor()
    model = LinearRegression()
    model.fit(X, y)

    yearling_df["predicted_self_score"] = model.predict(X)

    # Compute MAE
    mae = mean_absolute_error(yearling_df["self_score"], yearling_df["predicted_self_score"])
    print("Mean Absolute Error:", mae)


if __name__ == "__main__":
    main()
