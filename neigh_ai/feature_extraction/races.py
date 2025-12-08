import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas._libs import NaTType
from scipy.stats import hmean, zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


class RaceModel:
    def __init__(self) -> None:
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
            pd.read_csv("/Users/hayden/Downloads/vw_race_results_202509011836.csv")
            .assign(
                yearling_id=lambda df: df["yearling_id"].astype(str),
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
            .query("12 < speed < 19")
            .query("going == 'Fast'")
            .query("surface == 'Dirt'")
            .query("7 <= distance_furlongs <= 13")
            .pipe(lambda df: pd.concat([df, pd.get_dummies(df["going"], prefix="going")], axis=1))
            .dropna(subset=["speed", "pattern"])
        )

        # Train model once
        model = LinearRegression()
        model.fit(self.race_df["distance_meters"].to_numpy().reshape(-1, 1), self.race_df["speed"].to_numpy())

        self.horse_df: pd.DataFrame = (
            self.race_df.assign(
                predicted_speed=lambda df: model.predict(df["distance_meters"].to_numpy().reshape(-1, 1)),
                speed_diff=lambda df: (df["speed"] - df["predicted_speed"]) / df["predicted_speed"],
            )
            .groupby("yearling_id", as_index=False)
            .agg(
                num_races=("yearling_id", "size"),
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
                    col: lambda d, col=col: d[col].fillna(int(d[col].mean(numeric_only=True)))
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


def main():
    race_model = RaceModel()

    # print(race_model.horse_df.drop(columns=["yearling_id", "num_races"]).corr())
    # plot_corr("race_score", "avg_speed_diff", race_model.horse_df)
    # plot_corr("race_score", "avg_g1_finish", race_model.horse_df)
    # plot_corr("race_score", "avg_g2_finish", race_model.horse_df)
    # plot_corr("race_score", "log_avg_prize_money", race_model.horse_df)

    show_best_horses(race_model.horse_df, race_model.race_df)

    # show_features_info(race_model)


def show_best_horses(horse_df: pd.DataFrame, races_df: pd.DataFrame) -> None:
    for yearling_id in horse_df.sort_values(by="race_score", ascending=False)["yearling_id"]:
        plot_id(df=races_df, yearling_id=yearling_id)

        for column in horse_df.columns:
            if column in ["yearling_id", "num_races"]:
                continue

            plot_hist(horse_df, column, yearling_id)


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


def plot_id(df: pd.DataFrame, yearling_id: str) -> None:
    yearling_races = df[df["yearling_id"] == yearling_id]

    plot_power(df["distance_meters"], df["speed"])

    plt.scatter(yearling_races["distance_meters"], yearling_races["speed"], color="red", label="Selected horse")

    plt.title(f"Yearling: {yearling_id}")
    plt.legend()
    plt.show()


def plot_power(x: np.ndarray, y: np.ndarray, show=False):
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


def plot_hist(df: pd.DataFrame, column: str, yearling_id: str) -> None:
    plt.hist(df[column], bins=50)
    plt.title(f"Yearling: {yearling_id} - {column}")
    plt.axvline(x=df[df["yearling_id"] == yearling_id][column].item(), color="red", linestyle="--", label="x = 0")
    plt.show()


if __name__ == "__main__":
    main()
