from typing import List

import sc2reader
from sc2reader.events.tracker import PlayerStatsEvent


class Replay:
    """Stripped back custom version of the sc2reader Replay class."""

    def __init__(
        self,
        filehash: str,
        winner: int,
        player_1_race: str,
        player_2_race: str,
        events: List[sc2reader.events],
    ):

        self._filehash = filehash
        self._winner = winner
        self._player_1_race = player_1_race
        self._player_2_race = player_2_race
        self._events = events

    @property
    def filehash(self):
        return self._filehash

    @property
    def winner(self):
        return self._winner

    @property
    def player_1_race(self):
        return self._player_1_race

    @property
    def player_2_race(self):
        return self._player_2_race

    @property
    def events(self):
        return self._events

    @property
    def game_length(self):
        len_in_seconds = self.events[-1].second
        return f"{len_in_seconds} seconds"

    @classmethod
    def from_replay(cls, replay: sc2reader.resources.Replay):
        """Constructor to build class from sc2reader replay"""

        if not isinstance(replay, sc2reader.resources.Replay):
            raise TypeError("replay_file must be sc2reader.resources.Replay")

        return cls(
            filehash=replay.filehash,
            winner=replay.winner.number,
            player_1_race=replay.players[0].play_race,
            player_2_race=replay.players[1].play_race,
            events=replay.events,
        )

    @classmethod
    def from_path(cls, path: str):
        """Constructor to build class from replay file path"""

        if not isinstance(path, str):
            raise TypeError("path must be a str")

        if not path.endswith(".SC2Replay"):
            raise ValueError("path must point to a .SC2Replay file")

        replay = sc2reader.load_replay(path)

        return cls.from_replay(replay=replay)

    def get_player_events(self) -> list:
        """Return all events that are relevant to the players statistics."""

        player_events = [
            event for event in self.events
            if event.name == "PlayerStatsEvent"
        ]

        def events_hotfix(player_events: List[PlayerStatsEvent]):
            """Remove any non-paired events from the end of the player
            events list. Non-paired means any timestamped events where
            only one player has a player event"""

            for i, event in enumerate(player_events):

                if i == len(player_events):
                    return player_events
                else:
                    if event.player == player_events[i + 1].player:
                        return player_events[:i]

        fixed_player_events = events_hotfix(player_events)

        return fixed_player_events

    def __repr__(self):
        return (
            f"Replay(filehash={self.filehash}, "
            f"winner={self.winner}, "
            f"player_1_race={self.player_1_race}, "
            f"player_2_race={self.player_2_race}, "
            "events=[...])"
        )

    def __str__(self):
        return (
            "Replay Object \n"
            f"  Filehash: {self.filehash} \n"
            f"  Winner: {self.winner}"
        )
