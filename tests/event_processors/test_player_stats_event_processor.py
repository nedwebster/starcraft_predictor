import warnings

import numpy as np
import pandas as pd
import pytest
from starcraft_predictor import Replay
from starcraft_predictor.event_processors import PlayerStatsEventProcessor

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


class DummyEvent:

    def __init__(self, second=10, player="player_1", name="PlayerStatsEvent"):

        self.second = second
        self.player = player
        self.name = name
        self.food_made = 20


class TestPlayerStatsEventProcessor:

    def test_player_data_fields(self):

        assert (
            len(PlayerStatsEventProcessor.PLAYER_DATA_FIELDS)
            == 2 * len(PlayerStatsEventProcessor.DATA_FIELDS)
        )

    def test_init_empty_frame(self):

        output = PlayerStatsEventProcessor._init_empty_frame()

        assert isinstance(output, pd.DataFrame)
        assert output.shape == (0, 33)

    def test_process_events(self):

        event_pair = [
            DummyEvent(player="player_1"),
            DummyEvent(player="player_2"),
        ]

        # temporarily overwrite class attribute
        old_data_fields = PlayerStatsEventProcessor.DATA_FIELDS
        PlayerStatsEventProcessor.DATA_FIELDS = ["food_made"]

        output = PlayerStatsEventProcessor._process_event(event_pair)

        # revert temporary change
        PlayerStatsEventProcessor.DATA_FIELDS = old_data_fields

        assert (
            output.values
            == np.array([10, 20, 20], dtype="object")
        ).all()

    def test_get_events(self):

        replay = Replay(**{
            "filehash": "test",
            "winner": 1,
            "player_1_race": "z",
            "player_2_race": "t",
            "events": [
                DummyEvent(second=10, player="1", name="PlayerStatsEvent"),
                DummyEvent(second=10, player="2", name="PlayerStatsEvent"),
                DummyEvent(second=10, player="1", name="OtherEvent"),
                DummyEvent(second=20, player="1", name="PlayerStatsEvent"),
            ],
        })

        output = PlayerStatsEventProcessor._get_events(replay)

        assert len(output) == 2

    @pytest.mark.parametrize(
        "input_val, expected_output", [(0, [1, 2]), (2, [5, 6])]
    )
    def test_get_event_pair(self, input_val, expected_output):

        events = [1, 2, 3, 4, 5, 6, 7]

        output = PlayerStatsEventProcessor._get_event_pair(events, input_val)

        assert output == expected_output
