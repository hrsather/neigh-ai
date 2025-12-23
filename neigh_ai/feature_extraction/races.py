from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas._libs import NaTType
from scipy.stats import hmean, zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


class RaceModel:
    def __init__(self, race_results_path: Path) -> None:
        self._race_results_path = race_results_path
        self.feature_cols: list[str] = [
            "avg_speed_diff",
            "avg_g1_finish",
            "avg_g2_finish",
            "avg_g3_finish",
            "log_avg_prize_money",
        ]
        self._get_dfs()

    @classmethod
    def _parse_race_time(cls, val: str) -> pd.Timedelta | NaTType:
        try:
            minutes, seconds = val.split(":")
            minutes = int(minutes)
            seconds = float(seconds)
            return pd.to_timedelta(minutes * 60 + seconds, unit="s")
        except Exception:
            return pd.NaT

    @classmethod
    def _score_finish(cls, pos: int, max_points: int = 10) -> int:
        # 1st=max_points, 2nd=max_points-1... 10th place=1. >10th=0
        return max_points + 1 - np.minimum(pos, max_points + 1)

    def _get_dfs(self) -> None:
        self.race_df: pd.DataFrame = (
            pd.read_csv(self._race_results_path)
            .assign(
                horse_racing_api_id=lambda df: df["horse_racing_api_id"].astype(str),
                distance_meters=lambda df: pd.to_numeric(df["distance_yards"] * 0.9144, errors="coerce"),
                distance_furlongs=lambda df: pd.to_numeric(df["distance_yards"] / 220, errors="coerce"),
                speed=lambda df: pd.to_numeric(
                    df["distance_meters"]
                    / df["race_time"].apply(self._parse_race_time).apply(lambda x: x.total_seconds()),
                    errors="coerce",
                ),
                prize_money=lambda df: pd.to_numeric(df["prize_money"], errors="coerce"),
                pattern=lambda df: pd.to_numeric(df["pattern"].str.replace(r"\D", "", regex=True)),
                g1_finish=lambda df: np.where(df["pattern"] == 1, self._score_finish(df["finish_position"]), np.nan),
                g2_finish=lambda df: np.where(df["pattern"] == 2, self._score_finish(df["finish_position"]), np.nan),
                g3_finish=lambda df: np.where(df["pattern"] == 3, self._score_finish(df["finish_position"]), np.nan),
            )
            .dropna(subset=["speed"])
            # .query("12 < speed < 19")
            # .query("going == 'Fast'")
            # .query("surface == 'Dirt'")
            # .query("7 <= distance_furlongs <= 13")
        )

        # Train model once
        model = LinearRegression()
        model.fit(self.race_df["distance_meters"].to_numpy().reshape(-1, 1), self.race_df["speed"].to_numpy())

        self.horse_df: pd.DataFrame = (
            self.race_df.assign(
                predicted_speed=lambda df: model.predict(df["distance_meters"].to_numpy().reshape(-1, 1)),
                speed_diff=lambda df: (df["speed"] - df["predicted_speed"]) / df["predicted_speed"],
            )
            .groupby("horse_racing_api_id", as_index=False)
            .agg(
                num_races=("horse_racing_api_id", "size"),
                avg_speed_diff=("speed_diff", "mean"),
                total_prize_money=("prize_money", "sum"),
                avg_g1_finish=("g1_finish", "mean"),
                avg_g2_finish=("g2_finish", "mean"),
                avg_g3_finish=("g3_finish", "mean"),
            )
            .query("1 < total_prize_money")
            .assign(
                log_avg_prize_money=lambda df: np.log(df["total_prize_money"] / df["num_races"]),
                # Fills all na feature_cols with their mean
                **{
                    col: lambda d, col=col: d[col].fillna(d[col].mean(numeric_only=True))
                    for col in self.feature_cols
                    if col != "log_avg_prize_money"
                },
                # Harmonic mean of feature_cols
                race_score=lambda df: (
                    df[self.feature_cols]
                    .apply(zscore, ddof=0)
                    .pipe(lambda z: z + abs(z.min().min()) + 0.01)
                    .apply(hmean, axis=1)
                    .pipe(lambda s: s - (abs(df[self.feature_cols].apply(zscore, ddof=0).min().min()) + 0.01))
                ),
            )
        )

        self.horse_df = self._add_pedigree_info()

    def _add_pedigree_info(self):
        horse_df = (
            self.horse_df.merge(
                pd.read_csv("/Users/hayden/Downloads/racing_api_horses_202512181100.csv")[
                    ["racing_api_id", "sire_id", "dam_id", "horse_name"]
                ],
                left_on="horse_racing_api_id",
                right_on="racing_api_id",
                how="left",
            )
            .assign(sire_id=lambda d: d["sire_id"].str.replace("sir", "hrs", regex=False))
            .assign(dam_id=lambda d: d["dam_id"].str.replace("dam", "hrs", regex=False))
        )
        dam_lookup_id = horse_df.set_index("horse_racing_api_id")["dam_id"]
        sire_lookup_id = horse_df.set_index("horse_racing_api_id")["sire_id"]
        race_lookup_score = horse_df.set_index("horse_racing_api_id")["race_score"]

        # Compute grandparent IDs
        horse_df["damdam_id"] = horse_df["dam_id"].map(dam_lookup_id)
        horse_df["siresire_id"] = horse_df["sire_id"].map(sire_lookup_id)
        horse_df["siredam_id"] = horse_df["sire_id"].map(dam_lookup_id)
        horse_df["damsire_id"] = horse_df["dam_id"].map(sire_lookup_id)

        horse_df["siresiresire_id"] = horse_df["siresire_id"].map(sire_lookup_id)
        horse_df["siresiredam_id"] = horse_df["siresire_id"].map(dam_lookup_id)
        horse_df["siredamsire_id"] = horse_df["siredam_id"].map(sire_lookup_id)
        horse_df["siredamdam_id"] = horse_df["siredam_id"].map(dam_lookup_id)
        horse_df["damsiresire_id"] = horse_df["damsire_id"].map(sire_lookup_id)
        horse_df["damsiredam_id"] = horse_df["damsire_id"].map(dam_lookup_id)
        horse_df["damdamsire_id"] = horse_df["damdam_id"].map(sire_lookup_id)
        horse_df["damdamdam_id"] = horse_df["damdam_id"].map(dam_lookup_id)

        # Compute uncle/aunts scores
        dam_lookup_score = horse_df.groupby("dam_id")["race_score"].apply(list)
        sire_lookup_score = horse_df.groupby("sire_id")["race_score"].apply(list)

        horse_df["avg_damdam_auntuncle_score"] = horse_df["damdam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else np.mean(dam_lookup_score[x])
        )
        horse_df["avg_damsire_auntuncle_score"] = horse_df["damsire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else np.mean(sire_lookup_score[x])
        )
        horse_df["avg_siresire_auntuncle_score"] = horse_df["siresire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else np.mean(sire_lookup_score[x])
        )
        horse_df["avg_siredam_auntuncle_score"] = horse_df["siredam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else np.mean(dam_lookup_score[x])
        )

        horse_df["max_damdam_auntuncle_score"] = horse_df["damdam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else max(dam_lookup_score[x])
        )
        horse_df["max_damsire_auntuncle_score"] = horse_df["damsire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else max(sire_lookup_score[x])
        )
        horse_df["max_siresire_auntuncle_score"] = horse_df["siresire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else max(sire_lookup_score[x])
        )
        horse_df["max_siredam_auntuncle_score"] = horse_df["siredam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else max(dam_lookup_score[x])
        )

        horse_df["min_damdam_auntuncle_score"] = horse_df["damdam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else min(dam_lookup_score[x])
        )
        horse_df["min_damsire_auntuncle_score"] = horse_df["damsire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else min(sire_lookup_score[x])
        )
        horse_df["min_siresire_auntuncle_score"] = horse_df["siresire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else min(sire_lookup_score[x])
        )
        horse_df["min_siredam_auntuncle_score"] = horse_df["siredam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else min(dam_lookup_score[x])
        )

        horse_df["std_damdam_auntuncle_score"] = horse_df["damdam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else float(np.std(dam_lookup_score[x]))
        )
        horse_df["std_damsire_auntuncle_score"] = horse_df["damsire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else float(np.std(sire_lookup_score[x]))
        )
        horse_df["std_siresire_auntuncle_score"] = horse_df["siresire_id"].map(
            lambda x: np.nan if x not in sire_lookup_score else float(np.std(sire_lookup_score[x]))
        )
        horse_df["std_siredam_auntuncle_score"] = horse_df["siredam_id"].map(
            lambda x: np.nan if x not in dam_lookup_score else float(np.std(dam_lookup_score[x]))
        )

        # Compute scores
        horse_df["race_score_dam"] = horse_df["dam_id"].map(race_lookup_score)
        horse_df["race_score_sire"] = horse_df["sire_id"].map(race_lookup_score)
        horse_df["race_score_siredam"] = horse_df["siredam_id"].map(race_lookup_score)
        horse_df["race_score_siresire"] = horse_df["siresire_id"].map(race_lookup_score)
        horse_df["race_score_damdam"] = horse_df["damdam_id"].map(race_lookup_score)
        horse_df["race_score_damsire"] = horse_df["damsire_id"].map(race_lookup_score)
        horse_df["race_score_siresiresire"] = horse_df["siresiresire_id"].map(race_lookup_score)
        horse_df["race_score_siresiredam"] = horse_df["siresiredam_id"].map(race_lookup_score)
        horse_df["race_score_siredamsire"] = horse_df["siredamsire_id"].map(race_lookup_score)
        horse_df["race_score_siredamdam"] = horse_df["siredamdam_id"].map(race_lookup_score)
        horse_df["race_score_damsiresire"] = horse_df["damsiresire_id"].map(race_lookup_score)
        horse_df["race_score_damsiredam"] = horse_df["damsiredam_id"].map(race_lookup_score)
        horse_df["race_score_damdamsire"] = horse_df["damdamsire_id"].map(race_lookup_score)
        horse_df["race_score_damdamdam"] = horse_df["damdamdam_id"].map(race_lookup_score)

        horse_df["avg_dam_sibling_score"] = horse_df.groupby(["dam_id"])["race_score"].transform(
            lambda x: (x.sum() - x) / (len(x) - 1)
        )
        horse_df["avg_sire_sibling_score"] = horse_df.groupby(["sire_id"])["race_score"].transform(
            lambda x: (x.sum() - x) / (len(x) - 1)
        )
        horse_df["avg_damdam_cousin_score"] = horse_df.groupby(["damdam_id"])["race_score"].transform(
            lambda x: (x.sum() - x) / (len(x) - 1)
        )
        horse_df["avg_damsire_cousin_score"] = horse_df.groupby(["damsire_id"])["race_score"].transform(
            lambda x: (x.sum() - x) / (len(x) - 1)
        )
        horse_df["avg_siresire_cousin_score"] = horse_df.groupby(["siresire_id"])["race_score"].transform(
            lambda x: (x.sum() - x) / (len(x) - 1)
        )
        horse_df["avg_siredam_cousin_score"] = horse_df.groupby(["siredam_id"])["race_score"].transform(
            lambda x: (x.sum() - x) / (len(x) - 1)
        )

        def max_excluding_self(x):
            out = pd.Series(index=x.index, dtype=float)
            for idx in x.index:
                # drop current index
                others = x.drop(idx)
                out[idx] = others.max() if len(others) > 0 else 0
            return out

        def std_excluding_self(x):
            out = pd.Series(index=x.index, dtype=float)
            for idx in x.index:
                # drop current index
                others = x.drop(idx)
                out[idx] = others.std() if len(others) > 0 else 0
            return out

        def min_excluding_self(x):
            out = pd.Series(index=x.index, dtype=float)
            for idx in x.index:
                # drop current index
                others = x.drop(idx)
                out[idx] = others.min() if len(others) > 0 else 0
            return out

        horse_df["max_dam_sibling_score"] = horse_df.groupby("dam_id")["race_score"].transform(max_excluding_self)
        horse_df["max_sire_sibling_score"] = horse_df.groupby("sire_id")["race_score"].transform(max_excluding_self)
        horse_df["max_damdam_cousin_score"] = horse_df.groupby("damdam_id")["race_score"].transform(max_excluding_self)
        horse_df["max_damsire_cousin_score"] = horse_df.groupby("damsire_id")["race_score"].transform(
            max_excluding_self
        )
        horse_df["max_siresire_cousin_score"] = horse_df.groupby("siresire_id")["race_score"].transform(
            max_excluding_self
        )
        horse_df["max_siredam_cousin_score"] = horse_df.groupby("siredam_id")["race_score"].transform(
            max_excluding_self
        )

        horse_df["std_dam_sibling_score"] = horse_df.groupby("dam_id")["race_score"].transform(std_excluding_self)
        horse_df["std_sire_sibling_score"] = horse_df.groupby("sire_id")["race_score"].transform(std_excluding_self)
        horse_df["std_damdam_cousin_score"] = horse_df.groupby("damdam_id")["race_score"].transform(std_excluding_self)
        horse_df["std_damsire_cousin_score"] = horse_df.groupby("damsire_id")["race_score"].transform(
            std_excluding_self
        )
        horse_df["std_siresire_cousin_score"] = horse_df.groupby("siresire_id")["race_score"].transform(
            std_excluding_self
        )
        horse_df["std_siredam_cousin_score"] = horse_df.groupby("siredam_id")["race_score"].transform(
            std_excluding_self
        )

        horse_df["min_dam_sibling_score"] = horse_df.groupby("dam_id")["race_score"].transform(min_excluding_self)
        horse_df["min_sire_sibling_score"] = horse_df.groupby("sire_id")["race_score"].transform(min_excluding_self)
        horse_df["min_damdam_cousin_score"] = horse_df.groupby("damdam_id")["race_score"].transform(min_excluding_self)
        horse_df["min_damsire_cousin_score"] = horse_df.groupby("damsire_id")["race_score"].transform(
            min_excluding_self
        )
        horse_df["min_siresire_cousin_score"] = horse_df.groupby("siresire_id")["race_score"].transform(
            min_excluding_self
        )
        horse_df["min_siredam_cousin_score"] = horse_df.groupby("siredam_id")["race_score"].transform(
            min_excluding_self
        )

        return horse_df


def show_best_horses(horse_df: pd.DataFrame, races_df: pd.DataFrame) -> None:
    for horse_racing_api_id in horse_df.sort_values(by="race_score", ascending=False)["horse_racing_api_id"]:
        plot_id(df=races_df, horse_racing_api_id=horse_racing_api_id)

        for column in horse_df.columns:
            if column in ["horse_racing_api_id", "num_races"]:
                continue

            plot_hist(horse_df, column, horse_racing_api_id)


def show_features_info(race_model: RaceModel) -> None:
    cols = list(set(race_model.feature_cols) - {"race_score"})
    df = race_model.horse_df[cols]
    for target_col in cols:
        plt.hist(df[target_col], bins=50)
        plt.title(target_col)
        plt.show()

        X = df.drop(columns=[target_col])
        y = df[target_col]

        model = RandomForestRegressor()
        model.fit(X, y)
        y_pred = model.predict(X)
        print(mean_absolute_error(y, y_pred))

        print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))
        print()


def plot_corr(x_col: str, y_col: str, df: pd.DataFrame) -> None:
    x = df[x_col]
    y = df[y_col]
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    plt.scatter(x, y)
    m, b = np.polyfit(x, y, 1)  # 1 means linear
    plt.axvline(x=0, color="black", linestyle="--", label="x = 0")
    plt.plot(x, m * x + b, color="red", label="Linear fit")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    corr = df[x_col].corr(df[y_col])
    plt.title(f"Correlation: {corr:.4f}")
    plt.show()


def plot_id(df: pd.DataFrame, horse_racing_api_id: str) -> None:
    yearling_races = df[df["horse_racing_api_id"] == horse_racing_api_id]

    plot_power(df["distance_meters"], df["speed"])

    plt.scatter(yearling_races["distance_meters"], yearling_races["speed"], color="red", label="Selected horse")

    plt.title(f"Yearling: {horse_racing_api_id}")
    plt.legend()
    plt.show()


def plot_power(x: pd.Series, y: pd.Series, show=False):
    model = LinearRegression()
    model.fit(x.to_numpy().reshape(-1, 1), y)

    plt.scatter(x, y, color="blue", alpha=0.5, label="Data")

    x_fit = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_fit = model.predict(x_fit)
    plt.plot(x_fit, y_fit, color="red", linewidth=2)

    plt.xlabel("Distance (meters)")
    plt.ylabel("Speed (m/s)")
    plt.title("Race Speed vs Distance with Linear Fit")
    plt.legend()
    plt.grid(True)

    if show:
        plt.show()


def plot_hist(df: pd.DataFrame, column: str, horse_racing_api_id: str) -> None:
    plt.hist(df[column], bins=50)
    plt.title(f"Yearling: {horse_racing_api_id} - {column}")
    plt.axvline(
        x=df[df["horse_racing_api_id"] == horse_racing_api_id][column].item(),
        color="red",
        linestyle="--",
        label="x = 0",
    )
    plt.show()


def main():
    race_model = RaceModel(Path("/Users/hayden/Downloads/racing_api_horse_results_202512181056.csv"))
    horse_df = race_model.horse_df

    raw_feature_cols = [
        "race_score",
        "race_score_dam",
        "race_score_sire",
        "race_score_siredam",
        "race_score_siresire",
        "race_score_damdam",
        "race_score_damsire",
        "avg_dam_sibling_score",
        "avg_sire_sibling_score",
        "max_dam_sibling_score",
        "max_sire_sibling_score",
        "std_dam_sibling_score",
        "std_sire_sibling_score",
        "min_sire_sibling_score",
        "min_dam_sibling_score",
        "avg_damdam_cousin_score",
        "avg_damsire_cousin_score",
        "avg_siresire_cousin_score",
        "avg_siredam_cousin_score",
        "max_damdam_cousin_score",
        "max_damsire_cousin_score",
        "max_siresire_cousin_score",
        "max_siredam_cousin_score",
        "std_damdam_cousin_score",
        "std_damsire_cousin_score",
        "std_siresire_cousin_score",
        "std_siredam_cousin_score",
        "min_damdam_cousin_score",
        "min_damsire_cousin_score",
        "min_siresire_cousin_score",
        "min_siredam_cousin_score",
        "avg_damdam_auntuncle_score",
        "avg_damsire_auntuncle_score",
        "avg_siresire_auntuncle_score",
        "avg_siredam_auntuncle_score",
        "max_damdam_auntuncle_score",
        "max_damsire_auntuncle_score",
        "max_siresire_auntuncle_score",
        "max_siredam_auntuncle_score",
        "min_damdam_auntuncle_score",
        "min_damsire_auntuncle_score",
        "min_siresire_auntuncle_score",
        "min_siredam_auntuncle_score",
        "std_damdam_auntuncle_score",
        "std_damsire_auntuncle_score",
        "std_siresire_auntuncle_score",
        "std_siredam_auntuncle_score",
    ]

    feature_cols = []
    for col in raw_feature_cols:
        num_na_rows = horse_df[col].isna().sum()
        if num_na_rows > 0.5 * len(horse_df):
            print(f"Dropping {col}")
            continue
        feature_cols.append(col)

    print(horse_df[feature_cols].corr())
    feature_cols.remove("race_score")

    X = horse_df[feature_cols].fillna(0)
    y = horse_df["race_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color="blue", edgecolor="k")
    plt.xlabel("Actual Race Score")
    plt.ylabel("Predicted Race Score")
    plt.title("Random Forest Predictions vs Actual")
    plt.grid(True)
    plt.show()

    mean_pred = np.full_like(y_test, y_train.mean())
    me_mean = mean_squared_error(y_test, mean_pred)
    me_model = mean_squared_error(y_test, y_pred)

    print(f"MSE - Model: {me_model:.3f}, Mean baseline: {me_mean:.3f}")


# print(race_model.horse_df.drop(columns=["horse_racing_api_id", "num_races"]).corr())
# plot_corr("race_score", "avg_speed_diff", race_model.horse_df)
# plot_corr("race_score", "avg_g1_finish", race_model.horse_df)
# plot_corr("race_score", "avg_g2_finish", race_model.horse_df)
# plot_corr("race_score", "log_avg_prize_money", race_model.horse_df)
#
# show_best_horses(race_model.horse_df, race_model.race_df)

# show_features_info(race_model)


if __name__ == "__main__":
    main()
