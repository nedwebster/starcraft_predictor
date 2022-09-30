import pandas as pd
import pytest
from starcraft_predictor import PlotEngine


class TestPlotEngine():

    @pytest.mark.parametrize(
        "input_val, expected_output", [
            ([1, 2, 3], []),
            ([-2, -1, 3], [(1, 2)]),
            ([-2, 0, 1], []),
            ([-2, -1, 3, 4, -1], [(1, 2), (3, 4)]),
        ]
    )
    def test_find_crossing_pairs(self, input_val, expected_output):

        plot_engine = PlotEngine()

        output = plot_engine._find_crossing_pairs(input_val, 0.0)

        assert output == expected_output

    @pytest.mark.parametrize(
        "x, y, crossing_index, expected_x, expected_y", [
            ([0, 1], [-1, 1], [0, 1], [0, 0.5, 1], [-1, 0, 1]),
        ]
    )
    def test_insert_new_values(
        self, x, y, crossing_index, expected_x, expected_y
    ):

        plot_engine = PlotEngine()

        output = plot_engine._insert_new_values(x, y, crossing_index, 0.0)

        assert output[0] == expected_x
        assert output[1] == expected_y

    @pytest.mark.parametrize(
        "x, y, expected_x, expected_y", [
            ([0, 1, 2, 3], [-1, 1, 2, -3], [0, 0.5, 1], [-1, 0, 1]),
        ]
    )
    def test_add_zero_points(self, x, y, expected_x, expected_y):

        plot_engine = PlotEngine()

        output = plot_engine._add_threshold_points(x, y, 0.0)

        assert output[0] == [0, 0.5, 1, 2, 2.4, 3]
        assert output[1] == [-1, 0, 1, 2, 0, -3]

    @pytest.mark.mpl_image_compare
    def test_win_probability_plot(self):

        sample_df = pd.DataFrame(
            data={
                "seconds": [0, 10, 20, 30, 40, 50],
                "win_prob": [-1, -0.5, 0, 0.5, 1, 1],
            }
        )

        plot_engine = PlotEngine()

        plot = plot_engine.win_probability_plot(
            df=sample_df,
            p1_race="t",
            p2_race="z",
            match_id="111111",
            p1_handle="i_play_a_skill_race",
            p2_handle="i_play_a_noob_race"
        )
        return plot
