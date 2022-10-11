import pandas as pd

from sc2reader.events.tracker import PlayerStatsEvent

from starcraft_predictor.event_processors import BaseEventProcessor
from starcraft_predictor import Replay


class PlayerStatsEventProcessor(BaseEventProcessor):

    @classmethod
    def process_events(cls, replay: Replay) -> pd.DataFrame:
        player_events = [
            event for event in self.events
            if event.name == "PlayerStatsEvent"
        ]

        def events_hotfix(player_events: List[PlayerStatsEvent]):
            """Remove any non-paired events from the end of the player
            events list. Non-paired means any timestamped events where
            only one player has a player event"""

            for i, event in enumerate(player_events):

                if i + 1 == len(player_events):
                    return player_events
                else:
                    if event.player == player_events[i + 1].player:
                        return player_events[:i]

        fixed_player_events = events_hotfix(player_events)

        return fixed_player_events

