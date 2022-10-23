from typing import List
import pandas as pd
from functools import reduce

from starcraft_predictor.replay import Replay
from starcraft_predictor.event_processors import PlayerStatsEventProcessor


class ReplayEngine:
    """Class for loading and processing sc2 replay information into a readable
    structured table.

    The ReplayEngine has functionality for single replay loading, as well as
    batch replay loading.

    """

    event_processors = [
        PlayerStatsEventProcessor
    ]

    # TODO: Handle these fields with a MetaDataProcessor
    METADATA_FIELDS = [
        "filehash",
        "winner",
        "player_1_race",
        "player_2_race",
        "seconds",
    ]

    @classmethod
    def process_replays(cls, replays: List[Replay]):
        """
        Process a batch of replays.
        """
        pass

    @classmethod
    def process_replay(cls, replay: Replay):
        """
        Processes a replay by calling each event processor in turn and merging
        the resulting dataframes into one single dataframe.
        """
        df_list = [
            event_processor.process_events(replay=replay)
            for event_processor in cls.event_processors
        ]
        df = reduce(lambda x, y: pd.merge(x, y, on="seconds"), df_list)
        return df


if __name__ == "__main__":
    import starcraft_predictor as scp
    replay_path = "example_data/example_replay.SC2Replay"
    replay = scp.Replay.from_path(
        path=replay_path
    )
    df = ReplayEngine.process_replay(replay=replay)
    print(df)
