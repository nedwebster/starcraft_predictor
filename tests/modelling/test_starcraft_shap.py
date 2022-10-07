import pandas as pd
import pytest
import shap
from starcraft_predictor import StarcraftShap


class DummyShapExplainer():

    def shap_values(self, x):
        return "shap_test"


class TestStarcraftShap:

    def test_init(self, mocker):

        mocker.patch.object(
            StarcraftShap,
            "_get_moment_index",
            return_value="test"
        )

        sc_shap = StarcraftShap(
            processed_replay="test",
            features=["test_feature"],
            predictions=[0],
            model="test",
        )

        assert sc_shap.predictions == [0]
        assert sc_shap.features == ["test_feature"]
        assert sc_shap.moment_index == "test"
        assert sc_shap.model == "test"

    @pytest.mark.parametrize(
        "predictions, expected_output", [
            ([0, 1], [[0, 1]]),
            ([-1, 1, 2], [[0, 2]]),
            ([0, 3, 4, 2, -1, 3], [[0, 2], [2, 4], [4, 5]]),
        ]
    )
    def test_get_min_max_indexes(self, predictions, expected_output):

        sc_shap = StarcraftShap(
            processed_replay="test",
            features=["test_feature"],
            predictions=predictions,
            model="test",
        )

        output = sc_shap._get_min_max_indexes()

        assert output == expected_output

    @pytest.mark.parametrize(
        "index_pair, expected_output", [
            ([0, 1], 10),
            ([1, 3], -8),
            ([2, 4], 12),
        ]
    )
    def test_get_differece(self, index_pair, expected_output):

        sc_shap = StarcraftShap(
            processed_replay="test",
            features=["test_feature"],
            predictions=[0, 10, 3, 2, 15],
            model="test",
        )

        output = sc_shap._get_difference(index_pair)

        assert output == expected_output

    @pytest.mark.parametrize(
        "predictions, expected_output", [
            ([0, 1, 4, 8, 5], [0, 3]),
            ([4, 6, 3, 1, -1, 2], [1, 4]),
        ]
    )
    def test_get_moment_index(self, predictions, expected_output):

        sc_shap = StarcraftShap(
            processed_replay="test",
            features=["test_feature"],
            predictions=predictions,
            model="test",
        )

        output = sc_shap._get_moment_index()

        assert output == expected_output

    def test_get_shap_values(self, mocker):

        mocker.patch.object(
            shap,
            "TreeExplainer",
            return_value=DummyShapExplainer(),
        )

        sc_shap = StarcraftShap(
            processed_replay="test",
            features=0,
            predictions=[0, 1],
            model="test",
        )

        output = sc_shap._get_shap_values()

        assert output == "shap_test"

    @pytest.mark.parametrize(
        "index_pair, shap_values, expected_output", [
            ([0, 1], [[1, 2, 2], [2, 2, 2]], 0),
            ([0, 1], [[2, 1, 2], [2, 1, 2]], 1),
        ]
    )
    def test_get_max_shap_change(
        self, index_pair, shap_values, expected_output
    ):

        index_pair = [0, 1]
        shap_values = [[1, 3, 3], [2, 3, 3]]

        sc_shap = StarcraftShap(
            processed_replay="test",
            features=["test_feature"],
            predictions=[0],
            model="test",
        )

        output = sc_shap._get_max_shap_change(shap_values, index_pair)

        assert output == 0

    def test_get_feature_difference(self):

        test_data = pd.DataFrame({"a": [1, 4, 8]})

        sc_shap = StarcraftShap(
            processed_replay=test_data,
            features=["test_feature"],
            predictions=[0],
            model="test",
        )

        output = sc_shap._get_feature_difference(
            feature="a",
            index_pair=[0, 1],
        )

        assert output == 3

    def test_get_moment(self, mocker):

        mocker.patch.object(StarcraftShap, "_get_shap_values")
        mocker.patch.object(
            StarcraftShap,
            "_get_max_shap_change",
            return_value=0,
        )

        test_data = pd.DataFrame({"a": [1, 4, 8]})

        sc_shap = StarcraftShap(
            processed_replay=test_data,
            features=["a"],
            predictions=[0, 1, 2],
            model="test",
        )

        output = sc_shap.get_moment()

        assert output == ([0, 2], "a", 7)
