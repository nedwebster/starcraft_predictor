import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from starcraft_predictor.scp_api import load_pretrained_model
from starcraft_predictor import StarcraftModelEngine, Replay, ReplayEngine, api


class TestLoadPretrainedModel:

    def test_output(self):

        output = load_pretrained_model()

        assert isinstance(output, StarcraftModelEngine)


class TestScpApi:

    def test_load_replay(self, mocker):

        mocker.patch.object(Replay, "from_path", return_value="test")

        output = api._load_replay("dummy_path")

        assert output == "test"

    def test_process_replay(self, mocker):

        mocker.patch.object(ReplayEngine, "build_dataframe")
        mocker.patch.object(Pipeline, "transform", return_value="test")

        output = api._process_replay("dummy_replay")

        assert output == "test"

    def test_generate_predictions(self, mocker):

        mocker.patch.object(
            StarcraftModelEngine,
            "predict",
            return_value="test",
        )

        output = api._generate_predictions("dummy_data")

        assert output == "test"

    def test_get_plot_params(self):

        input_df = pd.DataFrame({
            "seconds": [10],
            "player_1_race": ["Zerg"],
            "player_2_race": ["Terran"],
        })

        output = api._get_plot_params(
            data=input_df,
            predictions=np.array([0.5])
        )

        assert list(output.keys()) == ["df", "p1_race", "p2_race"]
        pd.testing.assert_frame_equal(
            output["df"],
            pd.DataFrame({"seconds": [10], "win_prob": [0.5]})
        )
        assert output["p1_race"] == "z"
        assert output["p2_race"] == "t"
