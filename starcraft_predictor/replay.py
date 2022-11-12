import sc2reader


class Replay:
    """Stripped back custom version of the sc2reader Replay class."""

    def __init__(
        self,
        filehash: str,
        winner: int,
        player_1_race: str,
        player_2_race: str,
        events: list,
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

        if len(replay.players) > 2:
            raise ValueError(
                "replay_file must be a 1v1 game; team games are not supported"
            )

        return cls(
            filehash=replay.filehash,
            winner=replay.winner.number - 1,  # convert winner to [0, 1]
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
