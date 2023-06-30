import pytest
import sc2reader

from starcraft_predictor import Replay


class DummyEvent():

    def __init__(
        self,
        name: str = "PlayerStatsEvent",
        player: str = "player"
    ):

        self.second = 100
        self.name = name
        self.player = player


class TestReplay():

    def test_game_length_set_in_innit(self):

        input_args = {
            "filehash": "test",
            "winner": 1,
            "player_1_race": "z",
            "player_2_race": "t",
            "events": [DummyEvent()],
        }

        test_replay = Replay(**input_args)

        assert test_replay.game_length == "100 seconds"

    def test_from_replay_class_method(self):

        replay = sc2reader.load_replay(
            "example_data/example_replay.SC2Replay"
        )

        test_replay = Replay.from_replay(replay)

        assert isinstance(test_replay, Replay)

    def test_from_path_class_method(self):

        test_replay = Replay.from_path(
            "example_data/example_replay.SC2Replay"
        )

        assert isinstance(test_replay, Replay)

    @pytest.mark.parametrize(
        "input_val,expected_output",
        [
            (
                [
                    DummyEvent(name="PlayerStatsEvent", player="player_1"),
                    DummyEvent(name="random_event", player="player_2")
                ],
                1
            ),
            (
                [
                    DummyEvent(name="PlayerStatsEvent", player="player_1"),
                    DummyEvent(name="PlayerStatsEvent", player="player_2"),
                ],
                2
            ),
            (
                [
                    DummyEvent(name="PlayerStatsEvent", player="player_1"),
                    DummyEvent(name="PlayerStatsEvent", player="player_2"),
                    DummyEvent(name="PlayerStatsEvent", player="player_2"),
                ],
                1
            ),
        ],
    )
    def test_get_player_events(self, input_val, expected_output):

        input_args = {
            "filehash": "test",
            "winner": 1,
            "player_1_race": "z",
            "player_2_race": "t",
            "events": input_val,
        }

        test_replay = Replay(**input_args)

        output = test_replay.get_player_events()

        assert len(output) == expected_output

    def test_from_replay_errors(self):

        with pytest.raises(
            TypeError,
            match="replay_file must be sc2reader.resources.Replay",
        ):
            Replay.from_replay(replay="test")

    def test_from_path_error(self):

        with pytest.raises(
            TypeError,
            match="path must be a str",
        ):
            Replay.from_path(123)

        with pytest.raises(
            ValueError,
            match="path must point to a .SC2Replay file"
        ):
            Replay.from_path("test_path")
