from typing import List

import pandas as pd
from sc2reader.events.tracker import PlayerStatsEvent

from starcraft_predictor.replay import Replay


class ReplayEngine:
    """Class for loading and processing sc2 replay information into a readable
    structured table.

    The ReplayEngine has functionality for single replay loading, as well as
    batch replay loading.

    """

    METADATA_FIELDS = [
        "filehash",
        "winner",
        "player_1_race",
        "player_2_race",
        "seconds",
    ]

    DATA_FIELDS = [
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

    # duplicate DATA_FIELDS for each player
    PLAYER_DATA_FIELDS = [
        f"{field}_{player_number}" for field in DATA_FIELDS
        for player_number in [1, 2]
    ]

    @classmethod
    def build_dataframe(
        cls, replay: Replay
    ) -> pd.DataFrame:
        """Build a dataframe of structured data from a replay object"""

        if not isinstance(replay, Replay):
            raise TypeError(
                "replay must be a starcraft_predictor.replay.Replay"
            )

        # initialise empty dataframe with correct columns
        data = cls._init_empty_frame()

        # get all player events from replay
        player_events = replay.get_player_events()

        # each timestamp has two events, one for each player
        num_of_events = int(len(player_events) / 2)

        # for each event, construct a single row of data and append it to
        # the dataframe
        for i in range(num_of_events):
            event_pair = cls._get_event_pair(player_events, i)
            event_data = cls._process_event(event_pair)

            data.loc[i, :] = event_data

        # update table with replay metadata
        data["filehash"] = replay.filehash
        data["winner"] = replay.winner
        data["player_1_race"] = replay.player_1_race
        data["player_2_race"] = replay.player_2_race

        return data

    @classmethod
    def build_batch(cls, replays: list) -> pd.DataFrame:
        """Convert a list of replays into a single batch dataframe"""

        # initialise empty list to append replay dataframes to
        dataframes = []

        for i, replay in enumerate(replays):

            print(
                f"Building batch: {i+1}/{len(replays) + 1}",
                end="\r",
            )

            df = cls.build_dataframe(replay)
            dataframes.append(df)


        return pd.concat(dataframes)

    @classmethod
    def _init_empty_frame(cls) -> pd.DataFrame:
        """Initialises empty dataframe to append replay data to"""

        return pd.DataFrame(
            columns=[
                "filehash",
                "winner",
                "player_1_race",
                "player_2_race",
                "seconds",
            ]
            + cls.PLAYER_DATA_FIELDS
        )

    @classmethod
    def _process_event(cls, event: List[PlayerStatsEvent]) -> pd.Series:
        """Process a pair of events (player1 and player2) from a given time
        into a single row of data"""

        if event[0].second != event[1].second:
            raise ValueError("Tuple of events must be from the same game time")

        if event[0].player == event[1].player:
            raise ValueError("Tuple of events can't have same player value")

        event_series = pd.Series(dtype="object")
        event_series["seconds"] = event[0].second

        for field in cls.DATA_FIELDS:
            event_series[field + "_1"] = eval("event[0]." + field)
            event_series[field + "_2"] = eval("event[1]." + field)

        return event_series

    @staticmethod
    def _get_event_pair(
        player_events: List[PlayerStatsEvent], event_number: int
    ) -> List[PlayerStatsEvent]:
        """Gets player1 and player2 event items for a given event number"""

        #  adjust event number for 2 players
        event_number *= 2

        return player_events[event_number: event_number + 2]
