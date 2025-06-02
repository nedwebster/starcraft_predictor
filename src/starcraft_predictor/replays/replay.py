from enum import Enum
from typing import Annotated
from pydantic import AfterValidator, BaseModel, ConfigDict, computed_field
import sc2reader
from sc2reader.events.tracker import PlayerStatsEvent


def validate_events(events: list[PlayerStatsEvent]) -> list[PlayerStatsEvent]:
    """Pydantic validator for replay events."""

    if not all(isinstance(event, PlayerStatsEvent) for event in events):
        raise TypeError("All events must be PlayerStatsEvent objects")

    if len(events) < 2:
        raise ValueError("At least two PlayerStatsEvents are required")

    if len(events) % 2 != 0:
        raise ValueError("Number of events must be divisible by 2")

    if events[-1].player == events[-2].player:
        raise ValueError("End of game events must be from alternating players")

    return events


class Race(Enum):
    TERRAN = "Terran"
    PROTOSS = "Protoss"
    ZERG = "Zerg"
    RANDOM = "Random"


class Replay(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    filehash: str
    winner: int
    player_1_race: Race
    player_2_race: Race
    events: Annotated[list[PlayerStatsEvent], AfterValidator(validate_events)]

    @computed_field
    @property
    def game_length(self) -> int:
        """Calculate the game length in seconds from the last event."""
        return self.events[-1].second

    @classmethod
    def from_replay(cls, replay: sc2reader.resources.Replay):
        """Constructor to build class from sc2reader replay"""

        if not isinstance(replay, sc2reader.resources.Replay):
            raise TypeError("replay_file must be sc2reader.resources.Replay")

        if len(replay.players) > 2:
            raise ValueError(
                "replay_file must be a 1v1 game; team games are not supported"
            )

        player_events = filter_replay_events(replay.events)

        return cls(
            filehash=replay.filehash,
            winner=replay.winner.number,
            player_1_race=replay.players[0].play_race,
            player_2_race=replay.players[1].play_race,
            events=player_events,
        )

    @classmethod
    def from_path(cls, path: str):
        """Constructor to build class from replay file path"""
        replay = sc2reader.load_replay(path)
        return cls.from_replay(replay=replay)


def filter_replay_events(events: list) -> list:
    """Filter all events in a replay that aren't relevant to the players statistics."""

    player_events = [event for event in events if isinstance(event, PlayerStatsEvent)]

    def filter_unpaired_events(player_events: list[PlayerStatsEvent]):
        """
        Remove any non-paired events from the end of the player events list. Non-paired means any timestamped events
        where only one player has a player event. This can occure when a player leaves the game, and the other player
        continues playing.
        """

        if len(player_events) % 2 == 1:
            # if the number of events is odd, we have an unpaired event at the end
            player_events.pop()

        # remove any pairs of events  at the end which come from a single player
        while True:
            if player_events[-1].player == player_events[-2].player:
                player_events = player_events[:-2]
            else:
                break

        return player_events

    fixed_player_events = filter_unpaired_events(player_events)

    return fixed_player_events
