import warnings

import numpy as np
import pandas as pd
import pytest
from starcraft_predictor import Replay
from starcraft_predictor import ReplayEngine
from starcraft_predictor.event_processors import PlayerStatsEventProcessor

warnings.filterwarnings(action="ignore", category=DeprecationWarning)


class DummyEvent:

    def __init__(self, second=10, player="player_1", name="PlayerStatsEvent"):

        self.second = second
        self.player = player
        self.name = name
        self.food_made = 20


class DummyEventProcessor:

    @classmethod
    def process_events(cls, replay):
        return pd.DataFrame({"dummy_field": [1]})


class TestReplayEngine:

    def test_process_replay(self):

        # mocker.patch.object(
        #     Replay,
        #     "get_player_events",
        #     return_value=[
        #         DummyEvent(player="player_1"),
        #         DummyEvent(player="player_2"),
        #     ]
        # )

        test_replay = Replay(**{
            "filehash": "test",
            "winner": 1,
            "player_1_race": "z",
            "player_2_race": "t",
            "events": [
                DummyEvent(player="player_1"),
                DummyEvent(player="player_2"),
            ],
        })

        # temporarily overwrite class attribute
        old_event_processors = ReplayEngine.EVENT_PROCESSORS
        ReplayEngine.EVENT_PROCESSORS = [DummyEventProcessor]

        output = ReplayEngine.process_replay(test_replay)

        # revert temporary change
        ReplayEngine.EVENT_PROCESSORS = old_event_processors

        assert output.shape == (1, 5)
        assert output["filehash"][0] == "test"
        assert output["winner"][0] == 1
        assert output["player_1_race"][0] == "z"
        assert output["player_2_race"][0] == "t"
        assert output["dummy_field"][0] == 1

    def test_build_batch(self):

        test_replay_0 = Replay(**{
            "filehash": "test_0",
            "winner": 1,
            "player_1_race": "z",
            "player_2_race": "t",
            "events": [
                DummyEvent(player="player_1"),
                DummyEvent(player="player_2"),
            ],
        })
        test_replay_1 = Replay(**{
            "filehash": "test_1",
            "winner": 2,
            "player_1_race": "p",
            "player_2_race": "z",
            "events": [
                DummyEvent(player="player_1"),
                DummyEvent(player="player_2"),
            ],
        })

        # temporarily overwrite class attribute
        old_event_processors = ReplayEngine.EVENT_PROCESSORS
        ReplayEngine.EVENT_PROCESSORS = [DummyEventProcessor]

        output = ReplayEngine.process_batch([test_replay_0, test_replay_1])

        # revert temporary change
        ReplayEngine.EVENT_PROCESSORS = old_event_processors

        expected_output = pd.DataFrame(
            {
                "filehash": ["test_0", "test_1"],
                "winner": [1, 2],
                "player_1_race": ["z", "p"],
                "player_2_race": ["t", "z"],
                "dummy_field": [1, 1],
            },
            index=[0, 0],
        )

        pd.testing.assert_frame_equal(output, expected_output)
