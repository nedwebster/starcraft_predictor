from collections import Counter
import logging

import pandas as pd
import sc2reader
from sc2reader.events.tracker import (
    UnitBornEvent,
    UnitTypeChangeEvent,
    UnitDiedEvent,
    PlayerStatsEvent,
)

from starcraft_predictor.replays import TRACKED_UNIT_TYPES, EVENT_DATA_FIELDS, Matchup
from starcraft_predictor.errors import (
    IncorrectMatchupError,
    UnpairedPlayerStatEventError,
)


logger = logging.getLogger(__name__)


class ReplayIngester:
    """
    Class for ingesting replays and converting them into a Pandas DataFrame.

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

    def __init__(self, matchup: str | Matchup):
        if isinstance(matchup, str):
            matchup = Matchup(matchup)
        self.matchup = matchup
        self._reset_state()
        self.ingestion_map = {
            UnitBornEvent: self.ingest_unit_born_event,
            UnitDiedEvent: self.ingest_unit_died_event,
            UnitTypeChangeEvent: self.ingest_unit_type_change_event,
            PlayerStatsEvent: self.ingest_player_stats_event,
        }

    @classmethod
    def from_replay(
        self,
        replay_path: str | None = None,
        replay: sc2reader.resources.Replay | None = None,
    ) -> "ReplayIngester":
        """
        Class method to create a ReplayIngester instance from a replay file path.

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
        if replay is not None and replay_path is not None:
            raise ValueError(
                "Either provide a replay_path or a replay object, not both."
            )

        if replay_path is not None:
            replay = sc2reader.load_replay(replay_path)
        return ReplayIngester(Matchup.from_replay(replay))

    def _reset_state(self) -> None:
        """Reset the internal state of the ingester."""
        self.players = []
        self.replay_metadata = {}
        self.tracked_units = {0: {}, 1: {}}
        self.data = pd.DataFrame()
        self.player_stats_event_cache = []
        self.inverse_players = False

    def init_replay_tracking(
        self, replay: sc2reader.resources.Replay
    ) -> tuple[str, str, str]:
        """Initialise the replay tracking by resetting the initial state and extracting metadata."""
        self._reset_state()
        self._validate_replay(replay)

        self.players = replay.players

        self.replay_metadata = {
            "filehash": replay.filehash,
            "winner": self.get_winner(replay),
            "player_1_race": replay.players[self.inverse_players].play_race,
            "player_2_race": replay.players[1 - self.inverse_players].play_race,
        }

    def get_winner(self, replay: sc2reader.resources.Replay) -> int:
        if self.inverse_players:
            winner = 2 - replay.winner.players[0].pid
        else:
            winner = replay.winner.players[0].pid - 1
        return winner

    def _validate_replay(self, replay: sc2reader.resources.Replay) -> None:
        """Validate that the replay is a 1v1 replay and matches the expected matchup."""
        if replay.type != "1v1":
            raise ValueError("Replay must be a 1v1 replay.")

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
            raise IncorrectMatchupError()

    def ingest_replay(self, replay_path: str) -> pd.DataFrame:
        """
        Ingest a replay from a path pointing to a .SC2Replay file.

        Parameters
        ----------
        replay_path : str
            A path pointing to a .SC2Replay file.

        Returns
        -------
        data : pd.DataFrame
            A Pandas DataFrame containing the ingested replay data, with columns for replay metadata, player stats,
            and current units. Each row corresponds to a 10 second interval in the game.
        """
        replay = sc2reader.load_replay(replay_path)
        self.init_replay_tracking(replay)

        try:
            for event in replay.events:
                self.ingest_event(event)
        except UnpairedPlayerStatEventError:
            logger.warning("Unpaired player event found, ending event ingestion")

        logger.info("Replay finished loading")
        return self.data

    def ingest_event(self, event: sc2reader.events.Event) -> None:
        """Ingest a single event from the replay."""
        if type(event) in self.ingestion_map:
            self.ingestion_map[type(event)](event)
        else:
            logger.debug(f"Event type {type(event)} not tracked, skipping.")

    def ingest_unit_born_event(self, event: UnitBornEvent) -> None:
        """Ingest a UnitBornEvent and update the units dictionary."""
        if event.control_pid == 0:
            return None

        player = self.players[event.control_pid - 1]

        if event.unit_type_name in TRACKED_UNIT_TYPES[player.play_race]:
            self.tracked_units[event.control_pid - 1][
                event.unit_id
            ] = event.unit_type_name

    def ingest_unit_died_event(self, event: UnitDiedEvent) -> None:
        """
        Ingest a UnitDiedEvent and update the units dictionary.

        UnitDiedEvents do not have a player id, so we attempt to remove the unit from both players' tracked units.
        """
        not_found_unit = 0

        for i, _ in enumerate(self.players):
            try:
                self.tracked_units[i].pop(event.unit_id, None)
            except KeyError:
                not_found_unit += 1
                pass

        if not_found_unit > 1:
            logger.warning(
                f"Warning: UnitTypeChangeEvent for unit_id {event.unit_id} not found in tracked units for both players."
            )

    def ingest_unit_type_change_event(self, event: UnitTypeChangeEvent) -> None:
        """
        Ingest a UnitTypeChangeEvent and update the units dictionary.

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
                    pass

        if not_found_unit > 1:
            logger.warning(
                f"Warning: UnitTypeChangeEvent for unit_id {event.unit_id} not found in tracked units for both players."
            )

    def ingest_player_stats_event(self, event: PlayerStatsEvent) -> dict:
        """
        Ingest a PlayerStatsEvent and update the player stats data.

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
                    self.player_stats_event_cache[self.inverse_players], field
                )
                self.player_stats_data[f"player_2_{field}"] = getattr(
                    self.player_stats_event_cache[1 - self.inverse_players], field
                )

            self.generate_new_row()
        else:
            pass

    def validate_player_stats_events_cache(self) -> bool:
        """
        Validate whether the current PlayerStatsEvents are ready to be processed.

        It checks that there are two events in the cache, one for each player, and that they are from the same
        timestamp in game.
        """
        if len(self.player_stats_event_cache) > 2:
            raise ValueError(
                "Error ingesting player stats events: more than two player stats events in the cache. This should not "
                "happen."
            )
        if len(self.player_stats_event_cache) == 2:
            if (self.player_stats_event_cache[0].player != self.players[0]) and (
                self.player_stats_event_cache[1].player != self.players[1]
            ):
                raise UnpairedPlayerStatEventError()
            if (
                self.player_stats_event_cache[0].second
                != self.player_stats_event_cache[1].second
            ):
                raise ValueError(
                    "Error ingesting player stats events: player stats events for both players should be at the same "
                    "second."
                )
            return True
        else:
            return False

    def generate_new_row(self) -> None:
        """
        Generate a new row in the Data, combining replay metadata, player stats data, and current units.

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
