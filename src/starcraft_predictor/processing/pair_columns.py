from pydantic import BaseModel


class PairColumns(BaseModel):
    """
    A class to hold all columns which appear in pairs in the dataset.

    These are a specific raw feature type which appear as two columns in the replay dataframe, where one column
    corresponds to player_1, and the other corresponds to player_2.

    eg, 'food_used' appears as 'food_used_1' and 'food_used_2' in the replay dataframe.
    """

    base_columns: list[str] = []

    @property
    def all_columns(self) -> list[str]:
        """Returns all column pairs as a flat list."""
        return [f"{col}_{prefix}" for col in self.base_columns for prefix in [1, 2]]

    @property
    def column_tuples(self) -> list[tuple[str, str]]:
        """Returns all column pairs as a list of tuples."""
        return [(f"{col}_1", f"{col}_2") for col in self.base_columns]


BASE_PAIR_COLUMNS = [
    "food_made",
    "food_used",
    "minerals_collection_rate",
    "minerals_lost",
    "minerals_lost_army",
    "minerals_used_current",
    "minerals_used_current_army",
    "minerals_used_current_economy",
    "vespene_collection_rate",
    "vespene_lost",
    "vespene_lost_army",
    "vespene_used_current",
    "vespene_used_current_army",
    "vespene_used_current_economy",
    "vespene_used_current_technology",
    "workers_active_count",
]


PAIR_COLUMNS = PairColumns(base_columns=BASE_PAIR_COLUMNS)
