import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas._libs import NaTType
from scipy.stats import hmean, zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


class RaceModel:
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

    def get_raw_df(self) -> pd.DataFrame:
        return (
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
            .query("14.5 < speed < 17.5")
            .query("surface == 'Dirt'")
            .query("9 <= distance_furlongs <= 13")
            .pipe(lambda df: pd.concat([df, pd.get_dummies(df["going"], prefix="going")], axis=1))
            .dropna(subset=["speed", "pattern"])
        )

    @classmethod
    def _get_imputes(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, KNNImputer]:
        keep_cols: list[str] = ["yearling_id"]
        impute_cols: list[str] = [col for col in df.columns if col not in keep_cols]

        knn_imputer = KNNImputer()

        imputed_values = pd.DataFrame(
            knn_imputer.fit_transform(df[impute_cols]),
            columns=impute_cols,
            index=df.index,  # keep exact row alignment
        )

        imputed_df = pd.concat([df[keep_cols], imputed_values], axis=1)

        return imputed_df, knn_imputer

    def get_clean_df(self) -> pd.DataFrame:
        raw_df = self.get_raw_df()

        model = LinearRegression()
        model.fit(raw_df["distance_meters"].to_numpy().reshape(-1, 1), raw_df["speed"].to_numpy())

        return (
            raw_df.assign(
                predicted_speed=lambda df: model.predict(df["distance_meters"].to_numpy().reshape(-1, 1)),
                speed_diff=lambda df: (df["speed"] - df["predicted_speed"]) / df["predicted_speed"],
            )
            .groupby(["yearling_id"], as_index=False)
            .agg(
                num_races=("yearling_id", "size"),
                avg_speed_diff=("speed_diff", "mean"),
                total_prize_money=("prize_money_numeric", "sum"),
                avg_g1_finish=("g1_finish", "mean"),
                avg_g2_finish=("g2_finish", "mean"),
                avg_g3_finish=("g3_finish", "mean"),
            )
            .query("1 < total_prize_money")
            .assign(log_avg_prize_money=lambda df: np.log(df["total_prize_money"] / df["num_races"]))
            .drop(columns=["total_prize_money"])
        )

    def get_z_df(self) -> pd.DataFrame:
        clean_df = self.get_clean_df()
        impute_df, _ = self._get_imputes(clean_df)
        impute_df = impute_df.drop(columns=["yearling_id", "num_races"])
        z_df = impute_df.apply(zscore, ddof=0)
        shift = abs(z_df.min().min()) + 0.01  # small epsilon to avoid zero
        z_df = z_df + shift
        z_df["race_score"] = z_df.apply(hmean, axis=1)
        z_df = z_df - shift

        z_df["yearling_id"] = clean_df["yearling_id"]

        return z_df


def main():
    race_model = RaceModel()
    clean_df = race_model.get_clean_df()
    raw_df = race_model.get_raw_df()

    z_df = race_model.get_z_df()

    print(clean_df.drop(columns=["yearling_id", "num_races"]).corr())
    show_best_horses(z_df, clean_df, raw_df)
    show_plot_xy(z_df)
    show_features_info(z_df)


def show_best_horses(z_df: pd.DataFrame, clean_df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    for yearling_id in z_df.sort_values(by="race_score", ascending=False)["yearling_id"]:
        yearling_df = clean_df[clean_df["yearling_id"] == yearling_id]
        plot_id(df=raw_df, yearling_id=yearling_id)

        for column in z_df.columns:
            if column in ["yearling_id", "race_score"]:
                continue

            yearling_value = yearling_df[column].iloc[0]
            if not (yearling_value >= 0 or yearling_value < 0):  # nan. TODO: Improve
                continue

            plot_hist(clean_df, column, yearling_value, yearling_id)


def show_features_info(z_df: pd.DataFrame) -> None:
    for target_col in z_df.columns:
        if target_col == "yearling_id":
            continue

        plt.hist(z_df[target_col], bins=50)
        plt.title(target_col)
        plt.show()

        X = z_df.drop(columns=[target_col])
        y = z_df[target_col]

        print(target_col)
        model = RandomForestRegressor()
        model.fit(X, y)
        y_pred = model.predict(X)
        print(mean_absolute_error(y, y_pred))

        print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))
        print()


def show_plot_xy(z_df: pd.DataFrame) -> None:
    plot_xy("avg_speed_diff", "log_avg_prize_money", z_df)
    plot_xy("avg_speed_diff", "avg_g1_finish", z_df)
    plot_xy("avg_speed_diff", "avg_g2_finish", z_df)
    plot_xy("log_avg_prize_money", "avg_g1_finish", z_df)
    plot_xy("log_avg_prize_money", "avg_g2_finish", z_df)
    plot_xy("log_avg_prize_money", "race_score", z_df)


def plot_xy(x_col: str, y_col: str, df: pd.DataFrame) -> None:
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


def plot_hist(df: pd.DataFrame, column: str, yearling_value: float, yearling_id: str) -> None:
    plt.hist(df[column], bins=50)
    plt.title(f"Yearling: {yearling_id} - {column}")
    plt.axvline(x=yearling_value, color="red", linestyle="--", label="x = 0")
    plt.show()


if __name__ == "__main__":
    main()
