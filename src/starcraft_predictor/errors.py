class UnpairedPlayerStatEventError(Exception):
    """Raised when player stats events for the same player are received twice."""

    def __init__(
        self, message="Player stats events for the same player received twice."
    ):
        self.message = message
        super().__init__(self.message)


class IncorrectMatchupError(Exception):
    """Raised when the matchup of the replay does not match the expected one."""

    def __init__(self, message="Incorrect matchup in the replay."):
        self.message = message
        super().__init__(self.message)
