import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pandas._libs import NaTType
from scipy.stats import hmean, zscore
from sklearn.ensemble import RandomForestRegressor


class RaceModel:
    def __init__(
        self,
        race_results_path: str = "/Users/hayden/Downloads/racing_api_horse_results_202512181056.csv",
        pedigree_info_path: str = "/Users/hayden/Downloads/racing_api_horses_202512181100.csv",
    ) -> None:
        self._race_results_path = Path(race_results_path)
        self._pedigree_info_path = Path(pedigree_info_path)

        with open("configs.yaml") as f:
            config = yaml.safe_load(f)

            self.ps_features: list[str] = config["performance_score_features"]
            self.not_model_features: list[str] = config["not_model_features"]

        self.horse_df = pd.read_pickle("horse_df.pkl")
        self.race_df = pd.read_pickle("race_df.pkl")
        with open("avg_race_speed_model.pkl", "rb") as f:
            self.avg_race_speed_model = pickle.load(f)
        return

        self._get_dfs()
        self._fill_race_score_preds()

        self.horse_df.to_pickle("horse_df.pkl")
        self.race_df.to_pickle("race_df.pkl")
        with open("avg_race_speed_model.pkl", "wb") as f:
            pickle.dump(self.avg_race_speed_model, f)

    @property
    def model_cols(self) -> list[str]:
        ps_features: list[str] = []
        for col in list(set(self.horse_df.columns) - set(self.not_model_features)):
            num_na_rows = self.horse_df[col].isna().sum()
            if num_na_rows > 0.5 * len(self.horse_df):
                continue
            ps_features.append(col)
        return ps_features

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
        # 1st=max_points, 2nd=max_points-1, ..., 10th place=1, >10th=0
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
            .dropna(subset=["speed"])  # Drop races with no speed
            .query("10 < speed < 19")  # Drop outliers
            # .query("going == 'Fast'")
            # .query("surface == 'Dirt'")
            # .query("7 <= distance_furlongs <= 13")
        )

        self.avg_race_speed_model = RandomForestRegressor(max_depth=3).fit(
            self.race_df["distance_meters"].to_numpy().reshape(-1, 1), self.race_df["speed"].to_numpy()
        )

        self.horse_df: pd.DataFrame = (
            self.race_df.assign(
                predicted_speed=lambda df: self.avg_race_speed_model.predict(
                    df["distance_meters"].to_numpy().reshape(-1, 1)
                ),
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
                avg_g1_finish=lambda df: df["avg_g1_finish"].fillna(0),
                avg_g2_finish=lambda df: np.fmax(df["avg_g1_finish"], df["avg_g2_finish"]).fillna(0),
                avg_g3_finish=lambda df: np.fmax(df["avg_g2_finish"], df["avg_g3_finish"]).fillna(0),
                avg_prize_money=lambda df: df["total_prize_money"] / df["num_races"],
                log_avg_prize_money=lambda df: np.log(df["total_prize_money"] / df["num_races"]),
                # Harmonic mean of PS features
                race_score=lambda df: (
                    df[self.ps_features]
                    .apply(zscore, ddof=0)
                    .pipe(lambda z: z + abs(z.min().min()) + 0.01)
                    .apply(hmean, axis=1)
                    .pipe(lambda s: s - (abs(df[self.ps_features].apply(zscore, ddof=0).min().min()) + 0.01))
                ),
            )
        )

        self.horse_df = self._add_pedigree_stats()

    def _get_pedigree_info(self):
        horse_df = self.horse_df.merge(
            pd.read_csv(self._pedigree_info_path)[["racing_api_id", "sire_id", "dam_id", "horse_name"]],
            left_on="horse_racing_api_id",
            right_on="racing_api_id",
            how="left",
        ).assign(
            sire_id=lambda d: d["sire_id"].str.replace("sir", "hrs", regex=False),
            dam_id=lambda d: d["dam_id"].str.replace("dam", "hrs", regex=False),
            country=lambda d: d["horse_name"].str.extract(r"\((.*?)\)")[0],
            horse_name=lambda d: d["horse_name"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip(),
        )
        id_to_dam_id = horse_df.set_index("horse_racing_api_id")["dam_id"]
        id_to_sire_id = horse_df.set_index("horse_racing_api_id")["sire_id"]
        id_to_score = horse_df.set_index("horse_racing_api_id")["race_score"]
        id_to_name = horse_df.set_index("horse_racing_api_id")["horse_name"]

        for parent in ["dam", "sire"]:
            horse_df[f"race_score_{parent}"] = horse_df[f"{parent}_id"].map(id_to_score)
            horse_df[f"{parent}_name"] = horse_df[f"{parent}_id"].map(id_to_name)
            for gparent in ["dam", "sire"]:
                parent_lookup_id = id_to_dam_id if gparent == "dam" else id_to_sire_id
                horse_df[f"{parent}{gparent}_id"] = horse_df[f"{parent}_id"].map(parent_lookup_id)

                for ggparent in ["dam", "sire"]:
                    parent_lookup_id = id_to_dam_id if ggparent == "dam" else id_to_sire_id
                    horse_df[f"{parent}{gparent}{ggparent}_id"] = horse_df[f"{parent}{gparent}_id"].map(
                        parent_lookup_id
                    )
        return horse_df

    @classmethod
    def _stat_excluding_self(cls, x, stat: str) -> pd.Series:
        out = pd.Series(index=x.index, dtype=float)
        for idx in x.index:
            others = x.drop(idx)
            if len(others) == 0:
                out[idx] = np.nan
            elif stat == "max":
                out[idx] = others.max()
            elif stat == "min":
                out[idx] = others.std()
            elif stat == "std":
                out[idx] = others.min()
            elif stat == "avg":
                out[idx] = others.mean()
        return out

    def _add_pedigree_stats(self) -> pd.DataFrame:
        horse_df = self._get_pedigree_info()

        dam_stats = horse_df.groupby("dam_id")["race_score"].agg(["mean", "max", "min", "std"]).add_prefix("dam_")
        sire_stats = horse_df.groupby("sire_id")["race_score"].agg(["mean", "max", "min", "std"]).add_prefix("sire_")

        for stat in ["avg", "max", "min", "std"]:
            for parent in ["dam", "sire"]:
                horse_df[f"{stat}_{parent}_sibling_score"] = horse_df.groupby(f"{parent}_id")["race_score"].transform(
                    lambda x: self._stat_excluding_self(x, stat)  # noqa: B023
                )

                for gparent in ["dam", "sire"]:
                    horse_df[f"{stat}_{parent}{gparent}_cousin_score"] = horse_df.groupby(f"{parent}{gparent}_id")[
                        "race_score"
                    ].transform(lambda x: self._stat_excluding_self(x, stat))  # noqa: B023

                    map_obj = dam_stats if gparent == "dam" else sire_stats
                    horse_df[f"{stat}_{parent}{gparent}_auntuncle_score"] = horse_df[f"{parent}{gparent}_id"].map(
                        map_obj[f"{gparent}_{stat if stat != 'avg' else 'mean'}"]
                    )

        return horse_df

    def _fill_race_score_preds(self):
        X = self.horse_df[self.model_cols].fillna(0)
        y = self.horse_df["race_score"]

        rf = RandomForestRegressor()
        rf.fit(X, y)

        self.horse_df["race_score_pred"] = rf.predict(X)
        self.horse_df["race_score_pred_diff"] = self.horse_df["race_score_pred"] - self.horse_df["race_score"]
