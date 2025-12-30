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
        yearling_info_path: str = "/Users/hayden/Downloads/yearling_data_202512271425.csv",
        load_precomputed: bool = False,
    ) -> None:
        self._race_results_path = Path(race_results_path)
        self._pedigree_info_path = Path(pedigree_info_path)
        self._yearling_info_path = Path(yearling_info_path)

        with open("configs.yaml") as f:
            config = yaml.safe_load(f)

            self.ps_features: list[str] = config["performance_score_features"]
            self.not_model_features: list[str] = config["not_model_features"]

        if load_precomputed:
            self.horse_df = pd.read_pickle("horse_df.pkl")
            self.race_df = pd.read_pickle("race_df.pkl")
            with open("avg_race_speed_model.pkl", "rb") as f:
                self.avg_race_speed_model = pickle.load(f)
        else:
            self._get_dfs()
            self._fill_race_score_preds()

            self.horse_df.to_pickle("horse_df.pkl")
            self.race_df.to_pickle("race_df.pkl")
            with open("avg_race_speed_model.pkl", "wb") as f:
                pickle.dump(self.avg_race_speed_model, f)

    def get_pedigree_df(self) -> pd.DataFrame:
        pedigree_df = (
            pd.read_csv(self._pedigree_info_path)
            .assign(horse_name=lambda df: df["horse_name"].str.replace(r"\s*\([A-Z]{2,3}\)$", "", regex=True))[
                ["racing_api_id", "horse_name", "sire_name", "sire_id", "dam_name", "dam_id"]
            ]
            .drop_duplicates(subset=["horse_name"])
        )

        yearling_df = (
            pd.read_csv(self._yearling_info_path, dtype={"hip": str, "covering_sire": str, "photo_link": str})
            .drop_duplicates(subset=["name"])
            .rename(columns={"name": "horse_name_temp"})
            # Merge racing api id
            .merge(
                pedigree_df[["horse_name", "racing_api_id"]],
                left_on="horse_name_temp",
                right_on="horse_name",
                how="left",
            )
            .rename(columns={"racing_api_id": "horse_racing_api_id"})
            # Merge dam info
            .merge(
                pedigree_df[["horse_name", "racing_api_id"]],
                left_on="dam",
                right_on="horse_name",
                how="left",
                suffixes=("", "_dam"),
            )
            .rename(columns={"racing_api_id": "dam_id", "horse_name": "dam_name"})
            # Merge sire info
            .merge(
                pedigree_df[["horse_name", "racing_api_id"]],
                left_on="sire",
                right_on="horse_name",
                how="left",
                suffixes=("", "_sire"),
            )
            .rename(
                columns={
                    "racing_api_id": "sire_id",
                    "horse_name": "sire_name",
                    "horse_name_temp": "horse_name",
                }
            )
            .astype({"sale_id": "Int64"})
            .dropna(subset=["horse_name"])
        )[["sire_id", "dam_id", "horse_name", "sale_id", "sire_name", "dam_name", "hip", "horse_racing_api_id"]]

        pedigree_df = pedigree_df.rename(columns={"racing_api_id": "horse_racing_api_id"})

        # TODO: Remove when more sales!
        yearling_df = yearling_df[yearling_df["sale_id"] == 34]

        # Remove yearling rows already represented in pedigree_df
        yearling_df = yearling_df[~yearling_df["horse_racing_api_id"].isin(pedigree_df["horse_racing_api_id"])]

        return pd.concat([pedigree_df, yearling_df], ignore_index=True, sort=False)

    @property
    def model_cols(self) -> list[str]:
        ps_features: list[str] = []
        # consider only rows where race_score is not NaN
        df = self.horse_df[self.horse_df["race_score"].notna()]

        for col in set(df.columns) - set(self.not_model_features):
            num_na_rows = df[col].isna().sum()
            if num_na_rows > 0.5 * len(df):
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
                speed_diff=lambda df: df["speed"] - df["predicted_speed"],
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

        self._add_pedigree_stats()

    def _get_pedigree_info(self):
        pedigree_df = self.get_pedigree_df()

        self.horse_df = (
            self.horse_df.merge(pedigree_df, on="horse_racing_api_id", how="left")
            .assign(
                sire_id=lambda d: d["sire_id"].str.replace("sir", "hrs", regex=False),
                dam_id=lambda d: d["dam_id"].str.replace("dam", "hrs", regex=False),
                horse_name=lambda d: d["horse_name"].str.replace(r"\s*\(.*?\)", "", regex=True).str.strip(),
            )
            .dropna(subset=["horse_name"])
        )

        # Create a new DataFrame with exactly the columns of self.horse_df
        new_rows = pedigree_df.copy()

        # Fill any missing columns in horse_df that aren't in pedigree_df
        for col in self.horse_df.columns:
            if col not in new_rows.columns:
                new_rows[col] = pd.NA

        # Reorder columns to match self.horse_df
        new_rows = new_rows[self.horse_df.columns]

        # Concatenate
        new_rows = new_rows[~new_rows["horse_racing_api_id"].isin(self.horse_df["horse_racing_api_id"])]
        self.horse_df = pd.concat([self.horse_df, new_rows], ignore_index=True)

        id_to_dam_id = self.horse_df.dropna(subset=["horse_racing_api_id"]).set_index("horse_racing_api_id")["dam_id"]
        id_to_sire_id = self.horse_df.dropna(subset=["horse_racing_api_id"]).set_index("horse_racing_api_id")["sire_id"]
        id_to_score = self.horse_df.dropna(subset=["horse_racing_api_id"]).set_index("horse_racing_api_id")[
            "race_score"
        ]
        id_to_name = self.horse_df.dropna(subset=["horse_racing_api_id"]).set_index("horse_racing_api_id")["horse_name"]

        for parent in ["dam", "sire"]:
            self.horse_df[f"race_score_{parent}"] = self.horse_df[f"{parent}_id"].map(id_to_score)
            self.horse_df[f"{parent}_name"] = self.horse_df[f"{parent}_id"].map(id_to_name)
            for gparent in ["dam", "sire"]:
                parent_lookup_id = id_to_dam_id if gparent == "dam" else id_to_sire_id
                self.horse_df[f"{parent}{gparent}_id"] = self.horse_df[f"{parent}_id"].map(parent_lookup_id)

                for ggparent in ["dam", "sire"]:
                    parent_lookup_id = id_to_dam_id if ggparent == "dam" else id_to_sire_id
                    self.horse_df[f"{parent}{gparent}{ggparent}_id"] = self.horse_df[f"{parent}{gparent}_id"].map(
                        parent_lookup_id
                    )

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

    def _add_pedigree_stats(self) -> None:
        self._get_pedigree_info()

        dam_stats = self.horse_df.groupby("dam_id")["race_score"].agg(["mean", "max", "min", "std"]).add_prefix("dam_")
        sire_stats = (
            self.horse_df.groupby("sire_id")["race_score"].agg(["mean", "max", "min", "std"]).add_prefix("sire_")
        )

        for stat in ["avg", "max", "min", "std"]:
            for parent in ["dam", "sire"]:
                self.horse_df[f"{stat}_{parent}_sibling_score"] = self.horse_df.groupby(f"{parent}_id")[
                    "race_score"
                ].transform(
                    lambda x: self._stat_excluding_self(x, stat)  # noqa: B023
                )

                for gparent in ["dam", "sire"]:
                    self.horse_df[f"{stat}_{parent}{gparent}_cousin_score"] = self.horse_df.groupby(
                        f"{parent}{gparent}_id"
                    )["race_score"].transform(lambda x: self._stat_excluding_self(x, stat))  # noqa: B023

                    map_obj = dam_stats if gparent == "dam" else sire_stats
                    self.horse_df[f"{stat}_{parent}{gparent}_auntuncle_score"] = self.horse_df[
                        f"{parent}{gparent}_id"
                    ].map(map_obj[f"{gparent}_{stat if stat != 'avg' else 'mean'}"])

    def _fill_race_score_preds(self):
        # Only keep rows where race_score is not NA
        X_train = self.horse_df[self.horse_df["race_score"].notna()][self.model_cols].fillna(0)
        y_train = self.horse_df[self.horse_df["race_score"].notna()]["race_score"]

        assert y_train.isna().sum() == 0, "y_train still has NaNs!"

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)

        # Predict for all rows
        X_all = self.horse_df[self.model_cols].fillna(0)
        self.horse_df["race_score_pred"] = rf.predict(X_all)

        # Compute difference only where race_score exists
        self.horse_df["race_score_pred_diff"] = self.horse_df["race_score_pred"] - self.horse_df["race_score"]
