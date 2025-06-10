class UnpairedPlayerStatEventError(Exception):
    """Raised when player stats events for the same player are received twice."""

    def __init__(self):
        self.message = "Player stats events for the same player received twice."
        super().__init__(self.message)


class IncorrectMatchupError(Exception):
    """Raised when the matchup of the replay does not match the expected one."""

    def __init__(self):
        self.message = "Incorrect matchup in the replay."
        super().__init__(self.message)


class ReplayIngestionError(Exception):
    """Raised when there is an error ingesting a replay."""

    def __init__(self):
        self.message = "Either provide a replay_path or a replay object, not both!"
        super().__init__(self.message)
