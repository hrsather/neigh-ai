from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from neigh_ai.feature_extraction.races import RaceModel


def main() -> None:
    race_model = RaceModel()

    yearling_df = pd.read_csv("/Users/hayden/Downloads/vw_yearlings_202509011838.csv")
    yearling_df["id"] = yearling_df["id"].astype(str)
    name_to_id = defaultdict(int, zip(yearling_df["name"], yearling_df["id"]))
    yearling_df["sire_id"] = yearling_df["sire"].map(name_to_id)
    yearling_df["dam_id"] = yearling_df["dam"].map(name_to_id)

    id_to_score = defaultdict(int, zip(race_model.z_df["yearling_id"], race_model.z_df["race_score"]))
    yearling_df["sire_score"] = yearling_df["sire_id"].map(id_to_score)
    yearling_df["dam_score"] = yearling_df["dam_id"].map(id_to_score)
    yearling_df["self_score"] = yearling_df["id"].map(id_to_score)

    print(len(yearling_df))
    yearling_df = yearling_df[yearling_df["sire_score"] != 0]
    yearling_df = yearling_df[yearling_df["dam_score"] != 0]
    yearling_df = yearling_df[yearling_df["self_score"] != 0]
    print(len(yearling_df))

    plt.scatter(yearling_df["dam_score"], yearling_df["self_score"])
    plt.show()
    plt.scatter(yearling_df["sire_score"], yearling_df["self_score"])
    plt.show()

    X = yearling_df[["dam_score", "sire_score"]]

    y = yearling_df["self_score"]

    # model = RandomForestRegressor()
    # model = LinearRegression()
    model = RandomForestRegressor()
    model.fit(X, y)

    yearling_df["predicted_self_score"] = model.predict(X)

    # Compute MAE
    mae = mean_absolute_error(yearling_df["self_score"], yearling_df["predicted_self_score"])
    print("Mean Absolute Error:", mae)


if __name__ == "__main__":
    main()
