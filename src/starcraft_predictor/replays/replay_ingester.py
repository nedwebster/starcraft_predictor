import logging
from collections import Counter

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

from starcraft_predictor.errors import (
    IncorrectMatchupError,
    ReplayIngestionError,
    UnpairedPlayerStatEventError,
)
from starcraft_predictor.replays import EVENT_DATA_FIELDS, TRACKED_UNIT_TYPES, Matchup

logger = logging.getLogger(__name__)


def load_replay(
    replay_path: str | None = None,
    replay: sc2reader.resources.Replay | None = None,
) -> sc2reader.resources.Replay:
    """Load a replay from a path or an already loaded replay object."""
    if replay_path is not None and replay is not None:
        raise ReplayIngestionError
    if replay_path is not None:
        return sc2reader.load_replay(replay_path)
    return replay



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
    tracked_units : dict
        A dictionary containing the currently tracked units for both players, indexed by player ID. Eg:
        {1: {unit_id: unit_type_name, ...}, 2: {unit_id: unit_type_name, ...}}
    tracked_initialisations : dict
        A dictionary containing the currently tracked initialisations for both players, indexed by player ID.
        Initilisations include upgrades, buildings, and units which are warped-in/morphed. Once the initialisation is
        completed, the units/upgrades are added to the tracked units/upgradeds/buildings dict.
    data : pd.DataFrame
        A Pandas DataFrame that will contain the ingested data, with columns for replay metadata, player stats, and
        current units. Each row will be a 10 second interval in the game.
    player_stats_event_cache : list[PlayerStatsEvent]
        A cache for PlayerStatsEvents, used to ensure that player stats events are processed in pairs (one for each
        player) before generating a new row in the DataFrame.
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

    """

    def __init__(self, matchup: str | Matchup) -> None:
        if isinstance(matchup, str):
            matchup = Matchup(matchup)
        self.matchup = matchup
        self._reset_state()
        self.ingestion_map = {
            UnitBornEvent: self.ingest_unit_born_event,
            UnitDiedEvent: self.ingest_unit_died_event,
            UnitTypeChangeEvent: self.ingest_unit_type_change_event,
            PlayerStatsEvent: self.ingest_player_stats_event,
            UnitDoneEvent: self.ingest_unit_done_event,
            UnitInitEvent: self.ingest_unit_init_event,
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

    def _reset_state(self) -> None:
        """Reset the internal state of the ingester."""
        self.players = []
        self.replay_metadata = {}
        self.tracked_units = {0: {}, 1: {}}
        self.tracked_initialisations = {0: {}, 1: {}}
        self.data = pd.DataFrame()
        self.player_stats_event_cache = []
        self.inverse_players = False

    def ingest_replay(self, replay_path: str | None = None, replay: sc2reader.resources.Replay | None = None) -> pd.DataFrame:
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

        self._reset_state()
        self.init_replay_tracking(replay)

        try:
            for event in replay.events:
                ingestion_function = self.ingestion_map.get(
                    type(event), self.ignore_event,
                )
                ingestion_function(event)
        except UnpairedPlayerStatEventError:
            logger.warning("Unpaired player event found, ending event ingestion")

        logger.info("Replay finished loading")
        return self.data

    def init_replay_tracking(
        self, replay: sc2reader.resources.Replay,
    ) -> tuple[str, str, str]:
        """Initialise the replay tracking by resetting the initial state and extracting metadata."""
        self._validate_replay(replay)

        self.players = replay.players

        self.replay_metadata = {
            "filehash": replay.filehash,
            "winner": self.map_winner(replay),
            "player_1_race": replay.players[self.inverse_players].play_race,
            "player_2_race": replay.players[1 - self.inverse_players].play_race,
        }

    def _validate_replay(self, replay: sc2reader.resources.Replay) -> None:
        """Validate that the replay is a 1v1 replay and matches the expected matchup."""
        if replay.type != "1v1":
            msg = "Replay must be a 1v1 replay."
            raise ValueError(msg)

        player_1_race = replay.players[0].play_race
        player_2_race = replay.players[1].play_race

        if (player_1_race == self.matchup.race1) and (
            player_2_race == self.matchup.race2
        ):
            self.inverse_players = False
        elif (player_1_race == self.matchup.race2) and (
            player_2_race == self.matchup.race1
        ):
            self.inverse_players = True
        else:
            raise IncorrectMatchupError

    def map_winner(self, replay: sc2reader.resources.Replay) -> int:
        """Map the game winner from [1, 2] to binary [0, 1], accounting for the inverse players flag."""
        return abs(replay.winner.players[0].pid - 1 - int(self.inverse_players))

    def ingest_unit_born_event(self, event: UnitBornEvent) -> None:
        """Ingest a UnitBornEvent and update the units dictionary."""
        if event.control_pid == 0:
            return

        player = self.players[event.control_pid - 1]

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            self.tracked_units[event.control_pid - 1][
                event.unit_id
            ] = event.unit_type_name

    def ingest_unit_died_event(self, event: UnitDiedEvent) -> None:
        """Ingest a UnitDiedEvent and update the units dictionary.

        UnitDiedEvents do not have a player id, so we attempt to remove the unit from both players' tracked units.
        """
        try:
            self.tracked_units[0].pop(event.unit_id, None)
        except KeyError:
            try:
                self.tracked_units[1].pop(event.unit_id, None)
            except KeyError:
                logger.warning("Unit not found in tracked units for both players.", extra={"unit_id": event.unit_id})

    def ingest_unit_type_change_event(self, event: UnitTypeChangeEvent) -> None:
        """Ingest a UnitTypeChangeEvent and update the units dictionary.

        UnitTypeChangeEvents do not have a player id, so we attempt to update the unit type for both players' tracked
        units.
        """
        not_found_unit = 0

        for i, player in enumerate(self.players):
            if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
                try:
                    self.tracked_units[i][event.unit_id] = event.unit_type_name
                except KeyError:
                    not_found_unit += 1

        if not_found_unit > 1:
            logger.warning(
                "Warning: UnitTypeChangeEvent unit not found in tracked units for both players.",
                extra={"unit_id": event.unit_id},
            )

    def ingest_unit_init_event(self, event: UnitInitEvent) -> None:
        """Ingest a UnitInitEvent and update the initialisations dictionary."""
        if event.control_pid == 0:
            return

        player = self.players[event.control_pid - 1]

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            self.tracked_initialisations[event.control_pid - 1][
                event.unit_id
            ] = event.unit_type_name

    def ingest_unit_done_event(self, event: UnitDoneEvent) -> None:
        """Ingest a UnitDoneEvent and update the units dictionary."""
        try:
            unit_type_name = self.tracked_initialisations[0].pop(event.unit_id)
            self.tracked_units[0][event.unit_id] = unit_type_name
        except KeyError:
            try:
                unit_type_name = self.tracked_initialisations[1].pop(event.unit_id)
                self.tracked_units[1][event.unit_id] = unit_type_name
            except KeyError:
                logger.debug(
                    "Warning: UnitDoneEvent for not found in tracked units for both players.",
                    extra={"unit_id": event.unit_id},
                )

    def ingest_player_stats_event(self, event: PlayerStatsEvent) -> dict:
        """Ingest a PlayerStatsEvent and update the player stats data.

        The PlayerStatsEvent is also the trigger for producing a new row of data, so once the two consecutive
        PlayerStatsEvents for both player have been ingested,
        a new row is added to the data attribute.
        """
        # player stats events come in pairs, one for each player every 10 seconds. We want to ingest them in pairs.
        self.player_stats_event_cache.append(event)
        if self.validate_player_stats_events_cache():
            self.player_stats_data = {
                "seconds": self.player_stats_event_cache[0].second,
            }

            for field in EVENT_DATA_FIELDS:
                self.player_stats_data[f"player_1_{field}"] = getattr(
                    self.player_stats_event_cache[self.inverse_players], field,
                )
                self.player_stats_data[f"player_2_{field}"] = getattr(
                    self.player_stats_event_cache[1 - self.inverse_players], field,
                )

            self.generate_new_row()
        else:
            pass

    def ignore_event(self, event: sc2reader.events.Event) -> None:
        """Ignore an event by logging a debug message.

        This method is used for events that are not tracked or processed by the ingester.
        """
        logger.debug("Event type not tracked, skipping.", extra={"event_type": type(event)})

    def validate_player_stats_events_cache(self) -> bool:
        """Validate whether the current PlayerStatsEvents are ready to be processed.

        It checks that there are two events in the cache, one for each player, and that they are from the same
        timestamp in game.
        """
        if len(self.player_stats_event_cache) > 2:
            msg = "Error ingesting player stats events: more than two player stats events in the cache."
            raise ValueError(msg)
        if len(self.player_stats_event_cache) == 2:
            if (self.player_stats_event_cache[0].player != self.players[0]) and (
                self.player_stats_event_cache[1].player != self.players[1]
            ):
                raise UnpairedPlayerStatEventError
            if (
                self.player_stats_event_cache[0].second
                != self.player_stats_event_cache[1].second
            ):
                msg = "Error ingesting player stats events: events for both players should be at the same second."
                raise ValueError(msg)
            return True
        return False

    def generate_new_row(self) -> None:
        """Generate a new row in the Data, combining replay metadata, player stats data, and current units.

        The new row is concatened onto the existing DataFrame in the `data` attribute. Once the row is generated, the
        player stats event cache and player stats data are reset for the next pair of PlayerStatsEvents.
        """
        new_row = {
            **self.replay_metadata,
            **self.player_stats_data,
            **self.get_current_units(),
        }
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
        self.player_stats_event_cache = (
            []
        )  # Reset the cache for the next pair of PlayerStatsEvents
        self.player_stats_data = (
            {}
        )  # Reset the player stats data for the next pair of PlayerStatsEvents

    def get_current_units(self) -> Counter:
        """Get the current units for both players based on their tracked units, returned as a Counter object."""
        player_1_units = Counter(self.tracked_units[self.inverse_players].values())
        player_2_units = Counter(self.tracked_units[1 - self.inverse_players].values())

        player_1_units = {
            f"player_1_{x}": player_1_units.get(x, 0)
            for x in TRACKED_UNIT_TYPES[self.players[self.inverse_players].play_race]
        }
        player_2_units = {
            f"player_2_{x}": player_2_units.get(x, 0)
            for x in TRACKED_UNIT_TYPES[
                self.players[1 - self.inverse_players].play_race
            ]
        }

        combined_units = {**player_1_units, **player_2_units}

        if any(count < 0 for count in combined_units.values()):
            logger.warning("Warning: unit count is less than 0.")

        return combined_units
