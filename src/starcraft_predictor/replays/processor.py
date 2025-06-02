import pandas as pd

from starcraft_predictor.replays.replay import Replay
from starcraft_predictor.replays.consts import EVENT_DATA_FIELDS
from starcraft_predictor.replays.event_pair import EventPair, build_event_pairs


class ReplayProcessor:
    """
    Class to process Replay objects into a Pandas DataFrame.

    Attributes
    ----------
    data_fields : list[str]
        List of fields to extract from each PlayerStatsEvent.
    """

    def __init__(self, data_fields: list[str] = EVENT_DATA_FIELDS):
        self.data_fields = data_fields
        self.player_data_fields = [
            f"{field}_{player_number}"
            for field in data_fields
            for player_number in [1, 2]
        ]

    def load_replay(self, replay_path: str) -> Replay:
        """Load a Replay object from a file path."""
        return Replay.from_path(replay_path)

    def process_replay(self, replay: Replay) -> pd.DataFrame:
        """Process an individual Replay object into a Pandas Dataframe."""
        event_pairs = build_event_pairs(replay.events)

        processed_event_pairs = [
            self._process_event_pair(event_pair) for event_pair in event_pairs
        ]
        data = pd.DataFrame(processed_event_pairs)

        data["filehash"] = replay.filehash
        data["winner"] = replay.winner - 1  # convert to binary
        data["player_1_race"] = replay.player_1_race.value
        data["player_2_race"] = replay.player_2_race.value

        return data

    def process_batch(self, replays: list[Replay]) -> pd.DataFrame:
        """Process a batch of replays into a single Pandas Dataframe"""
        dataframes = []

        for i, replay in enumerate(replays):
            print(f"Building batch: {i+1}/{len(replays)}", end="\r")
            try:
                df = self.process_replay(replay)
                dataframes.append(df)
            except Exception as e:
                print(f"\n{i+1} failed: {e}")

        return pd.concat(dataframes)

    def _process_event_pair(self, event_pair: EventPair) -> pd.Series:
        """Process an EventPair into a Pandas Series."""
        event_series = pd.Series(dtype="object")
        event_series["seconds"] = event_pair.event_1.second

        for field in self.data_fields:
            event_series[f"{field}_1"] = getattr(event_pair.event_1, field, None)
            event_series[f"{field}_2"] = getattr(event_pair.event_2, field, None)

        return event_series
