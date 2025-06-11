import sc2reader

from starcraft_predictor.errors import UnpairedPlayerStatEventError

PLAYER_STATS_FIELDS = [
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


class PlayerStatsTracker:
    """Track player stats events and generate new rows of data."""

    def __init__(self, players: list[sc2reader.objects.Participant]) -> None:
        self.players = players
        self.player_stats_event_cache = []
        self.player_stats_data = {}

    def handle_player_stats_event(self, event: sc2reader.events.PlayerStatsEvent) -> bool:
        """Handle a PlayerStatsEvent and update the player stats data.

        PlayerStatsEvents come in pairs, one for each player every 10 seconds. We want to ingest them in pairs.
        """
        self.player_stats_event_cache.append(event)
        if self.validate_player_stats_events_cache():
            self.player_stats_data = {
                "seconds": self.player_stats_event_cache[0].second,
            }

            for field in PLAYER_STATS_FIELDS:
                self.player_stats_data[f"player_1_{field}"] = getattr(
                    self.player_stats_event_cache[0], field,
                )
                self.player_stats_data[f"player_2_{field}"] = getattr(
                    self.player_stats_event_cache[1], field,
                )

            return True
        return False

    def validate_player_stats_events_cache(self) -> bool:
        """Validate whether the current PlayerStatsEvents are ready to be processed.

        It checks that there are two events in the cache, one for each player, and that they are from the same
        timestamp in game.
        """
        player_stat_limit = len(self.players)

        if len(self.player_stats_event_cache) > player_stat_limit:
            msg = "Error ingesting player stats events: more than two player stats events in the cache."
            raise ValueError(msg)
        if len(self.player_stats_event_cache) == player_stat_limit:
            if any(self.player_stats_event_cache[i].player != self.players[i] for i in range(player_stat_limit)):
                raise UnpairedPlayerStatEventError
            if (
                self.player_stats_event_cache[0].second
                != self.player_stats_event_cache[1].second
            ):
                msg = "Error ingesting player stats events: events for both players should be at the same second."
                raise ValueError(msg)
            return True
        return False

    def reset_state(self) -> None:
        """Reset the state of the PlayerStatsTracker."""
        self.player_stats_event_cache = []
        self.player_stats_data = {}
