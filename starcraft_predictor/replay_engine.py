from typing import List

import pandas as pd
import sc2reader
from sc2reader.events.tracker import PlayerStatsEvent


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

    @staticmethod
    def load_replay(path: str) -> sc2reader.resources.Replay:
        """Load a sc2 replay file using sc2reader"""

        if not isinstance(path, str):
            raise TypeError("path must be a str")

        if not path.endswith(".SC2Replay"):
            raise ValueError("path must point to a .SC2Replay file")

        return sc2reader.load_replay(path)

    @classmethod
    def build_dataframe(
        cls, replay: sc2reader.resources.Replay
    ) -> pd.DataFrame:
        """Build a dataframe of structured data from a replay object"""

        # TODO: update to use custom replay class when implemented
        if not isinstance(replay, sc2reader.resources.Replay):
            raise TypeError("replay must be a sc2reader.resources.Replay")

        # initialise empty dataframe with correct columns
        data = cls._init_empty_frame()

        # get all player events from replay
        player_events = cls._get_player_stats_events(replay)

        # each timestamp has two events, one for each player
        num_of_events = int(len(player_events) / 2)

        # for each event, construct a single row of data and append it to
        # the dataframe
        for i in range(num_of_events):
            event_pair = cls._get_event_pair(player_events, i)
            event_data = cls._process_event(event_pair)

            data.loc[i, :] = event_data

        # update table with replay metadata
        # TODO: update replay references when replay class is implemented
        data["filehash"] = replay.filehash
        data["winner"] = replay.winner.number
        data["player_1_race"] = replay.players[0].play_race
        data["player_2_race"] = replay.players[1].play_race

        return data

    @classmethod
    def load_batch(cls, paths: list):
        """Load multiple replays from a list of paths"""
        # TODO: add functionality
        pass

    @classmethod
    def build_batch(cls, replays: list):
        """Convert a list of replays into a single batch dataframe"""
        # TODO: add functionality
        pass

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

    @staticmethod
    def _get_player_stats_events(replay: sc2reader.resources.Replay) -> list:
        """Get all events that are relevant to the players statistics."""

        player_stats_events = [
            event for event in replay.events
            if event.name == "PlayerStatsEvent"
        ]

        def events_hotfix(player_events: List[PlayerStatsEvent]):
            """Remove any non-paired events from the end of the player
            events list. Non-paired are any timestamped events where
            only one player has a player event"""

            for i, event in enumerate(player_events):

                if i == len(player_events):
                    return player_events
                else:
                    if event.player == player_events[i + 1].player:
                        return player_events[:i]

        fixed_player_events = events_hotfix(player_stats_events)

        return fixed_player_events

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
