import sc2reader

from starcraft_predictor.errors import InvalidReplayError
from starcraft_predictor.replays.matchup import Matchup


class ReplayValidator:
    """Class for validating replays.

    The class validates:
    - that the replay is a 1v1 game
    - that the matchup is correct, based on the player races in the replay and a defined Matchup object

    Attributes
    ----------
    replay : sc2reader.resources.Replay
        The replay to validate.
    matchup : str | Matchup
        The matchup to validate the replay against.

    """

    def __init__(self, replay: sc2reader.resources.Replay, matchup: str | Matchup) -> None:
        self.replay = replay
        self.matchup = matchup

    def validate(self) -> None:
        """Validate the replay."""
        if not all([
            self._validate_game_type(),
            self._validate_matchup(),
        ]):
            raise InvalidReplayError

    def _validate_game_type(self) -> bool:
        """Validate that the replay is a 1v1 game."""
        return self.replay.type == "1v1"

    def _validate_matchup(self) -> bool:
        """Validate that the player races in the replay match the specified Matchup."""
        return set([player.play_race for player in self.replay.players]) == set(self.matchup.races)
