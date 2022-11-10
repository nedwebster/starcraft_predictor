from typing import List
import pandas as pd

from sc2reader.events.tracker import PlayerStatsEvent

from starcraft_predictor.event_processors import BaseEventProcessor
from starcraft_predictor.replay import Replay


class PlayerStatsEventProcessor(BaseEventProcessor):
    """
    Class for processing player stats events from a Replay object.
    """

    ID_FIELDS = [
        "seconds"
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
    def process_events(cls, replay: Replay) -> pd.DataFrame:
        """Process player stats events across all times for a given replay and
        return them in a dataframe."""

        # initialize empty dataframe with correct column
        df = cls._init_empty_frame()

        # get  all player events from replay
        events_list = cls._get_events(replay=replay)

        # each timestamp has two events, one for each player
        num_of_events = int(len(events_list) / 2)

        # for each event, construct a single row of data and append it to the
        # dataframe
        for i in range(num_of_events):
            event_pair = cls._get_event_pair(events_list, i)
            event_data = cls._process_event(event_pair)
            df.loc[i, :] = event_data

        return df

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

    @classmethod
    def _init_empty_frame(cls) -> pd.DataFrame:
        """Initialises empty dataframe to append replay data to"""

        return pd.DataFrame(
            columns=cls.ID_FIELDS + cls.PLAYER_DATA_FIELDS
        )

    @classmethod
    def _get_events(cls, replay: Replay) -> List[PlayerStatsEvent]:
        events = [
            event for event in replay.events
            if event.name == "PlayerStatsEvent"
        ]

        def events_hotfix(player_events: List[PlayerStatsEvent]):
            """Remove any non-paired events from the end of the player
            events list. Non-paired means any timestamped events where
            only one player has a player event.

            Any number of non-paired player events can happen at the end of
            the game once one player has surrendered and exited, hence the
            calculation is not as trivial as looking for an even/odd number of
            events."""
            for i, event in enumerate(player_events):
                if i + 1 == len(player_events):
                    if (i + 1) % 2 == 0:
                        return player_events
                    else:
                        return player_events[:i]
                else:
                    if event.player == player_events[i + 1].player:
                        return player_events[:i]
        fixed_events = events_hotfix(events)
        return fixed_events

    @staticmethod
    def _get_event_pair(
        player_events: List[PlayerStatsEvent], event_number: int
    ) -> List[PlayerStatsEvent]:
        """Gets player1 and player2 event items for a given event number"""

        #  adjust event number for 2 players
        event_number *= 2

        return player_events[event_number: event_number + 2]


if __name__ == "__main__":
    import starcraft_predictor as scp
    replay_path = "example_data/example_replay.SC2Replay"
    replay = scp.Replay.from_path(
        path=replay_path
    )
    print(replay)
    player_event_df = PlayerStatsEventProcessor.process_events(replay=replay)
    print(player_event_df)
