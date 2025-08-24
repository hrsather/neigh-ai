from collections import defaultdict
from typing import Optional

import pandas as pd

from neigh_ai.constants import get_hip_sheet_path


def get_df() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(get_hip_sheet_path())

    # Drop rows where there are name duplicates
    unique_names = df["name"].value_counts()
    unique_names = list(unique_names[unique_names == 1].index)  # type: ignore[reportAttributeAccessIssue]
    df = pd.DataFrame(df[df["name"].isin(unique_names)])

    return df


def get_family_tree() -> dict[str, dict[str, str]]:
    df = get_df()
    tree_df: pd.DataFrame = df[
        ["name", "dam", "sire", "paternal_grandsire", "paternal_granddam", "maternal_grandsire", "maternal_granddam"]
    ]  # type: ignore[reportAssignmentType]
    tree: dict[str, dict[str, str]] = defaultdict(
        lambda: {
            "dam": "N/A",
            "sire": "N/A",
            "paternal_grandsire": "N/A",
            "paternal_granddam": "N/A",
            "maternal_grandsire": "N/A",
            "maternal_granddam": "N/A",
        }
    )

    def safe_get(row: pd.Series, col: str, default: str = "N/A") -> str:
        val: Optional[str | float] = row.get(col, default)
        if val is None or isinstance(val, float):
            return default
        return val.strip() if val.strip() else default

    for _, row in tree_df.iterrows():
        tree[safe_get(row, "name")] = {
            "dam": safe_get(row, "dam"),
            "sire": safe_get(row, "sire"),
            "paternal_grandsire": safe_get(row, "paternal_grandsire"),
            "paternal_granddam": safe_get(row, "paternal_granddam"),
            "maternal_grandsire": safe_get(row, "maternal_grandsire"),
            "maternal_granddam": safe_get(row, "maternal_granddam"),
        }

    # Add parents to data, if missing
    for family in list(tree.values()):
        if family["sire"] not in tree:
            tree[family["sire"]]["sire"] = family["paternal_grandsire"]
            tree[family["sire"]]["dam"] = family["paternal_granddam"]
        if family["dam"] not in tree:
            tree[family["dam"]]["sire"] = family["maternal_grandsire"]
            tree[family["dam"]]["dam"] = family["maternal_granddam"]

    # Link grandparents manually, if missing
    for name, family in list(tree.items()):
        grandparent_map = {
            "paternal_grandsire": ("sire", "sire"),
            "paternal_granddam": ("sire", "dam"),
            "maternal_grandsire": ("dam", "sire"),
            "maternal_granddam": ("dam", "dam"),
        }

        for key, (parent_key, parent_side) in grandparent_map.items():
            if tree[name][key] == "N/A" and tree[family[parent_key]][parent_side] != "N/A":
                tree[name][key] = tree[family[parent_key]][parent_side]

    return tree


def get_names() -> list[str]:
    return list(get_df()["name"])
