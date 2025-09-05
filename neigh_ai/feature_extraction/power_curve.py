import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def parse_race_time(val: str):
    try:
        minutes, seconds = val.split(":")
        minutes = int(minutes)
        seconds = float(seconds)
        return pd.to_timedelta(minutes * 60 + seconds, unit="s")
    except Exception:
        return pd.NaT


def get_df():
    # yearling_df = pd.read_csv("/Users/hayden/Downloads/vw_yearlings_202509011838.csv")
    race_df = pd.read_csv("/Users/hayden/Downloads/vw_race_results_202509011836.csv")
    race_df["speed"] = (
        race_df["distance_yards"]
        * 0.9144
        / race_df["race_time"].apply(parse_race_time).apply(lambda x: x.total_seconds())
    )

    race_df["distance_yards"] = pd.to_numeric(race_df["distance_yards"], errors="coerce")
    race_df["speed"] = pd.to_numeric(race_df["speed"], errors="coerce")

    race_df.dropna(subset=["speed"], inplace=True)

    return race_df


def main():
    df = get_df()
    x = df["distance_yards"]
    x = x.to_numpy().reshape(-1, 1)
    y = df["speed"].to_numpy()

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(x, y)

    my_func = rf

    mse = mean_squared_error(y, my_func.predict(x))

    print("MSE:", mse)
    plot(x, y, my_func)


def plot(x, y, model):
    plt.scatter(x, y, color="blue", alpha=0.5, label="Data")

    # Plot fitted curve
    x_fit = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_fit = model.predict(x_fit)
    plt.plot(x_fit, y_fit, color="red", linewidth=2)

    plt.xlabel("Distance (yards)")
    plt.ylabel("Speed (seconds)")
    plt.title("Race Speed vs Distance with Power Law Fit")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
