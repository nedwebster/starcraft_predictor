import logging

import pandas as pd
import sc2reader
from sc2reader.events.tracker import (
    PlayerStatsEvent,
    UnitBornEvent,
    UnitDiedEvent,
    UnitDoneEvent,
    UnitInitEvent,
    UnitTypeChangeEvent,
)

from starcraft_predictor.errors import UnpairedPlayerStatEventError
from starcraft_predictor.replays import load_replay
from starcraft_predictor.replays.matchup import Matchup
from starcraft_predictor.replays.player_stats_tracker import PlayerStatsTracker
from starcraft_predictor.replays.unit_tracker import UnitTracker
from starcraft_predictor.replays.validator import ReplayValidator

logger = logging.getLogger(__name__)


class ReplayIngester:
    """Class for ingesting replays and converting them into a Pandas DataFrame.

    The class converts replays to DataFrames by ingesting events sequentially, processing the information from each
    event and updating the internal state accordingly. At every PlayerStatsEvent (every 10 seconds in the replay) the
    data is collected and a new row is generated in the DataFrame. The DataFrame contains metadata about the replay,
    player stats, and the current state of tracked units for both players. Once all events have been ingested, the
    dataframe is returned.

    Example
    -------
    >>> ingester = ReplayIngester(matchup="Terran vs Zerg")
    >>> data = ingester.ingest_replay(replay)

    Parameters
    ----------
    matchup : str | Matchup
        The matchup type that the replay ingester can process. Ingesters only handle a specific match up as the unit
        types, which end up as columns in the return DataFrame, are matchup dependant.

    Attributes
    ----------
    players : list
        List of players in the replay.
    replay_metadata : dict
        A dictionary of metadata items ingested from the replay at the start.
    data : pd.DataFrame
        A Pandas DataFrame that will contain the ingested data, with columns for replay metadata, player stats,
        and current units. Each row will be a 10 second interval in the game.
    inverse_players : bool
        A boolean to indicate whether the players in the replay should be inverted. This is to ensure consistent race
        orderings in non-mirror matchups. The inverse_players is also used as an integer to extract the correct player
        at various times, where:
            player_0 = self.players[inverse_players], and player_1 = self.players[1 - inverse_players].
        When the inverse_players is True, player_0 will be the second player in the replay, and player_1 will be the
        first player.
    ingestion_map : dict
        A mapping of event types to their corresponding ingestion methods. Any event types without a defined ingestion
        method will be skipped.
    unit_tracker : UnitTracker
        An instance of UnitTracker that handles all unit-related events and maintains the current state of units.
    player_stats_tracker : PlayerStatsTracker
        An instance of PlayerStatsTracker that handles all player stats events and maintains the current state of
        player stats.

    """

    def __init__(self, matchup: str | Matchup) -> None:
        if isinstance(matchup, str):
            matchup = Matchup(matchup)
        self.matchup = matchup
        self.reset_state()
        self.ingestion_map = {
            UnitBornEvent: lambda e: self.unit_tracker.handle_unit_born(e),
            UnitDiedEvent: lambda e: self.unit_tracker.handle_unit_died(e),
            UnitTypeChangeEvent: lambda e: self.unit_tracker.handle_unit_type_change(e),
            PlayerStatsEvent: lambda e: self.player_stats_tracker.handle_player_stats_event(e),
            UnitDoneEvent: lambda e: self.unit_tracker.handle_unit_done(e),
            UnitInitEvent: lambda e: self.unit_tracker.handle_unit_init(e),
        }

    @classmethod
    def from_replay(
        cls,
        replay_path: str | None = None,
        replay: sc2reader.resources.Replay | None = None,
    ) -> "ReplayIngester":
        """Class method to create a ReplayIngester instance from a replay file path.

        The method loads the replay file, extracts the matchup information, and initializes a ReplayIngester with the
        correct matchup.

        Parameters
        ----------
        replay_path : str | None = None
            The path to the .SC2Replay file. If provided, it will be used to load the replay.

        replay : sc2reader.resources.Replay | None = None
            An already loaded replay object. If provided, it will be used to extract the matchup information.

        Returns
        -------
        ReplayIngester
            An instance of ReplayIngester initialized with the matchup from the replay.

        """
        replay = load_replay(replay_path, replay)
        return cls(Matchup.from_replay(replay))

    def reset_state(self) -> None:
        """Reset the internal state of the ingester."""
        self.players = []
        self.replay_metadata = {}
        self.data = pd.DataFrame()
        self.inverse_players = False
        self.unit_tracker = None
        self.player_stats_tracker = None

    def ingest_replay(
        self,
        replay_path: str | None = None,
        replay: sc2reader.resources.Replay | None = None,
    ) -> pd.DataFrame:
        """Ingest a replay from a path pointing to a .SC2Replay file.

        Parameters
        ----------
        replay_path : str | None = None
            A path pointing to a .SC2Replay file. If provided, the replay will be loaded from this path.
        replay : sc2reader.resources.Replay | None = None
            An already loaded replay object. If provided, the replay will be used to ingest the data.

        Returns
        -------
        data : pd.DataFrame
            A Pandas DataFrame containing the ingested replay data, with columns for replay metadata, player stats,
            and current units. Each row corresponds to a 10 second interval in the game.

        """
        replay = load_replay(replay_path, replay)
        ReplayValidator(replay, self.matchup).validate()

        self.reset_state()
        self.init_replay_tracking(replay)

        self.ingest_events(replay.events)

        logger.info("Replay finished loading")
        return self.data

    def ingest_events(self, events: list[sc2reader.events.Event]) -> None:
        """Ingest a list of events and update the internal state of the ingester."""
        try:
            for event in events:
                ingestion_method = self.ingestion_map.get(
                    type(event),
                    lambda e: logger.debug("Event type not tracked, skipping.", extra={"event_type": type(e)}),
                )
                output = ingestion_method(event)
                if output:
                    self.generate_new_row()
        except UnpairedPlayerStatEventError:
            # Unpaired player events happen at the end of the replay once one player has left the game
            logger.warning("Unpaired player event found, ending event ingestion")

    def init_replay_tracking(
        self, replay: sc2reader.resources.Replay,
    ) -> None:
        """Initialise the replay tracking by resetting the initial state and extracting metadata."""
        self.players = replay.players
        self.inverse_players = self.players[0].play_race == self.matchup.race1
        self.unit_tracker = UnitTracker(self.players, self.inverse_players)
        self.player_stats_tracker = PlayerStatsTracker(self.players, self.inverse_players)

        self.replay_metadata = {
            "filehash": replay.filehash,
            "winner": abs(replay.winner.players[0].pid - 1 - int(self.inverse_players)),
            "player_1_race": replay.players[self.inverse_players].play_race,
            "player_2_race": replay.players[1 - self.inverse_players].play_race,
        }

    def generate_new_row(self) -> None:
        """Generate a new row in the Data, combining replay metadata, player stats data, and tracked units.

        The new row is concatened onto the existing DataFrame in the `data` attribute. Once the row is generated, the
        player_stats_tracker is reset for the next pair of PlayerStatsEvents.
        """
        new_row = {
            **self.replay_metadata,
            **self.player_stats_tracker.player_stats_data,
            **self.unit_tracker.get_current_units(),
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
        self.player_stats_tracker.reset_state()
