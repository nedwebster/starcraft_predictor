from typing import List
import pandas as pd
from functools import reduce

from starcraft_predictor.replay import Replay
from starcraft_predictor.event_processors import PlayerStatsEventProcessor


class ReplayEngine:
    """
    Class for loading and processing sc2 replay information into a readable
    structured table.

    The ReplayEngine has functionality for single replay loading, as well as
    batch replay loading.

    Event processors are used by the replay engine to process specific event
    information for feature construction.
    """

    EVENT_PROCESSORS = [
        PlayerStatsEventProcessor,
    ]

    # Supplied in reverse order for df.insert(loc=0) purposes.
    METADATA_FIELDS = [
        "player_2_race",
        "player_1_race",
        "winner",
        "filehash",
    ]

    @classmethod
    def process_batch(cls, replays: List[Replay]) -> pd.DataFrame:
        """Process a batch of replays."""
        processed_replays = [cls.process_replay(replay) for replay in replays]
        return(pd.concat(processed_replays))

    @classmethod
    def process_replay(cls, replay: Replay) -> pd.DataFrame:
        """Processes a replay by calling each event processor in turn and
        merging the resulting dataframes into one single dataframe. Meta data
        is also appended to the dataframe."""
        df_list = [
            event_processor.process_events(replay=replay)
            for event_processor in cls.EVENT_PROCESSORS
        ]
        df = reduce(lambda x, y: pd.merge(x, y, on="seconds"), df_list)
        for metadata_field in cls.METADATA_FIELDS:
            df.insert(0, metadata_field, getattr(replay, metadata_field))
        return df


if __name__ == "__main__":
    import starcraft_predictor as scp
    replay_path = "example_data/example_replay.SC2Replay"
    replay = scp.Replay.from_path(
        path=replay_path
    )
    df = ReplayEngine.process_replay(replay=replay)
    print(df)
