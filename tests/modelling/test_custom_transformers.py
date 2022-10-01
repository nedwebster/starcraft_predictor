import pandas as pd
import pytest
from starcraft_predictor.modelling import custom_transformers


class TestColumnDifferenceTransformer:

    @pytest.mark.parametrize(
        "diff_cols, new_cols, drop_cols, error_type", [
            (123, ["col"], True, TypeError),
            (["col"], 123, True, TypeError),
            (["col"], ["col"], 123, TypeError),
            (["col1", "col2"], ["col3"], True, ValueError),
        ]
    )
    def test_init(self, diff_cols, new_cols, drop_cols, error_type):

        with pytest.raises(error_type):

            _ = custom_transformers.ColumnDifferenceTransformer(
                diff_columns=diff_cols,
                new_columns=new_cols,
                drop_cols=drop_cols
            )

    @pytest.mark.parametrize(
        "drop_cols, output", [
            (True, 1),
            (False, 3)
        ]
    )
    def test_output(self, drop_cols, output):

        test_data = pd.DataFrame({
            "col1": [1, 2, 5],
            "col2": [3, 10, 2],
        })

        transformer = custom_transformers.ColumnDifferenceTransformer(
            diff_columns=[("col1", "col2")],
            new_columns=["col3"],
            drop_cols=drop_cols
        )

        output_df = transformer.transform(test_data)

        assert output_df.shape[1] == output
        assert list(output_df["col3"].values) == [-2, -8, 3]
