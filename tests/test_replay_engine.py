import warnings

import numpy as np
import pandas as pd
import pytest
from starcraft_predictor import Replay
from starcraft_predictor import ReplayEngine


warnings.filterwarnings(action="ignore", category=DeprecationWarning)


class DummyEvent:

    def __init__(self, second=10, player="player_1"):

        self.second = second
        self.player = player
        self.food_made = 20


class TestReplayEngine:

    def test_player_data_fields(self):

        assert (
            len(ReplayEngine.PLAYER_DATA_FIELDS)
            == 2 * len(ReplayEngine.DATA_FIELDS)
        )

    def test_init_empty_frame(self):

        output = ReplayEngine._init_empty_frame()

        assert isinstance(output, pd.DataFrame)
        assert output.shape == (0, 37)

    def test_process_events(self):

        event_pair = [
            DummyEvent(player="player_1"),
            DummyEvent(player="player_2"),
        ]

        # temporarily overwrite class attribute
        old_data_fields = ReplayEngine.DATA_FIELDS
        ReplayEngine.DATA_FIELDS = ["food_made"]

        output = ReplayEngine._process_event(event_pair)

        assert (
            output.values
            == np.array([10, 20, 20], dtype="object")
        ).all()

        # revert temporary change
        ReplayEngine.DATA_FIELDS = old_data_fields

    @pytest.mark.parametrize(
        "input_val, expected_output", [(0, [1, 2]), (2, [5, 6])]
    )
    def test_get_event_pair(self, input_val, expected_output):

        events = [1, 2, 3, 4, 5, 6]

        output = ReplayEngine._get_event_pair(events, input_val)

        assert output == expected_output

    def test_build_dataframe(self, mocker):

        mocker.patch.object(
            Replay,
            "get_player_events",
            return_value=[
                DummyEvent(player="player_1"),
                DummyEvent(player="player_2"),
            ]
        )

        # temporarily overwrite class attribute
        old_data_fields = ReplayEngine.DATA_FIELDS
        old_player_data_fields = ReplayEngine.PLAYER_DATA_FIELDS
        ReplayEngine.DATA_FIELDS = ["food_made"]
        ReplayEngine.PLAYER_DATA_FIELDS = ["food_made_1", "food_made_2"]

        test_replay = Replay(**{
            "filehash": "test",
            "winner": 1,
            "player_1_race": "z",
            "player_2_race": "t",
            "events": [DummyEvent()],
        })

        output = ReplayEngine.build_dataframe(test_replay)

        assert output.shape == (1, 7)
        assert output["food_made_1"][0] == 20
        assert output["food_made_2"][0] == 20

        # revert temporary change
        ReplayEngine.DATA_FIELDS = old_data_fields
        ReplayEngine.PLAYER_DATA_FIELDS = old_player_data_fields

    def test_build_batch(self, mocker):

        mock_output = pd.DataFrame({"a": [1]})

        mocker.patch.object(
            ReplayEngine,
            "build_dataframe",
            return_value=mock_output,
        )

        replay = Replay(**{
            "filehash": "test",
            "winner": 1,
            "player_1_race": "z",
            "player_2_race": "t",
            "events": [DummyEvent()],
        })

        output = ReplayEngine.build_batch([replay, replay])

        expected_output = pd.DataFrame(
            {"a": [1, 1]},
            index=[0, 0]
        )

        pd.testing.assert_frame_equal(output, expected_output)
