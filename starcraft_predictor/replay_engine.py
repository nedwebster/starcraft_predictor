import sc2reader


class ReplayEngine:
    """Class for loading and processing sc2 replay information into a readable
    structured table.

    The ReplayEngine functionality for has single replay loading, as well as
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
    def _get_player_stats_events(replay: sc2reader.resources.Replay) -> list:
        """Get all events that are relevant to the players statistics"""

        player_stats_events = [
                event for event in replay.events
                if event.name == "PlayerStatsEvent"
        ]

        # player events come in pairs (1 for each player every 10 seconds)
        # drop the last event if there is an odd number of events
        if len(player_stats_events) % 2 == 1:
            player_stats_events = player_stats_events[:-1]

        return player_stats_events

    @staticmethod
    def load_replay(path: str) -> sc2reader.resources.Replay:
        """Load a sc2 replay file using sc2reader"""

        if not isinstance(path, str):
            raise TypeError("path must be a str")

        if not path.endswith(".SC2Replay"):
            raise ValueError("path must point to a .SC2Replay file")

        return sc2reader.load_replay(path)

    @classmethod
    def build_dataframe(cls, replay):
        """Convert a replay into a readable dataframe"""
        # TODO: add functionality
        pass

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
