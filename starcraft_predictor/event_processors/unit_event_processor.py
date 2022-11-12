from typing import List, Union
import pandas as pd

from sc2reader.events.tracker import UnitBornEvent, UnitDoneEvent
from sc2reader.data import Unit

from starcraft_predictor.event_processors import BaseEventProcessor
from starcraft_predictor.replay import Replay
from starcraft_predictor.utils import round_down_tens
from starcraft_predictor.units import AIR_UNITS, ANTI_AIR_UNITS


class UnitEventProcessor(BaseEventProcessor):
    """
    Class for processing unit born and unit done events. These events are
    primarily used for creating "army information" features. Unit born events
    cover units which are trained over time (e.g. zerglings, marines). Unit
    done events are applicable to units which are warped in (e.g. stalkers).
    """

    EVENT_NAMES = [
        "UnitBornEvent",
        "UnitDoneEvent",
    ]

    ID_FIELDS = [
        "seconds",
    ]

    DATA_FIELDS = [
        "air_supply",
        "anti_air_supply",
    ]

    # duplicate DATA_FIELDS for each player
    PLAYER_DATA_FIELDS = [
        f"{field}_{player_number}" for field in DATA_FIELDS
        for player_number in [1, 2]
    ]

    @classmethod
    def process_events(cls, replay: Replay) -> pd.DataFrame:
        """Process unit born and unit create events across all times for a
        given replay and return them in a dataframe."""
        df = cls._init_starting_frame(replay=replay)

        units = cls._get_units(replay=replay)

        for unit in units:
            if unit.owner is None:
                next
            unit_created, unit_died = cls._get_unit_living_range(
                unit=unit, max_died_at=replay.game_length_int + 10
            )
            if unit.name in AIR_UNITS:
                df.loc[
                    df.seconds.between(unit_created, unit_died),
                    f"air_supply_{unit.owner.team_id}"
                ] = df.loc[
                    df.seconds.between(unit_created, unit_died),
                    f"air_supply_{unit.owner.team_id}"
                ] + unit.supply
            if unit.name in ANTI_AIR_UNITS:
                df.loc[
                    df.seconds.between(unit_created, unit_died),
                    f"anti_air_supply_{unit.owner.team_id}"
                ] = df.loc[
                    df.seconds.between(unit_created, unit_died),
                    f"anti_air_supply_{unit.owner.team_id}"
                ] + unit.supply

        return df

    @classmethod
    def _get_units(
        cls, replay: Replay
    ) -> List[Union[UnitBornEvent, UnitDoneEvent]]:

        units = [
            event.unit for
            event in replay.events if
            event.name in cls.EVENT_NAMES
        ]

        return units

    @classmethod
    def _get_unit_living_range(cls, unit: Unit, max_died_at: int) -> tuple:
        unit_created = round_down_tens(unit.finished_at / 16)
        unit_died = (
            round_down_tens(max_died_at) if
            unit.died_at is None else
            round_down_tens(unit.died_at / 16)
        )

        return unit_created, unit_died

    @classmethod
    def _init_starting_frame(cls, replay: Replay) -> pd.DataFrame:
        """Initialises empty dataframe to append replay data to"""

        data = pd.DataFrame({
            "seconds": range(0, replay.game_length_int, 10)
        })

        for field in cls.PLAYER_DATA_FIELDS:
            data[field] = 0

        return data
