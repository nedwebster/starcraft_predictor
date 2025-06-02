from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from starcraft_predictor.processing.processing_functions import (
    calculate_column_difference,
    drop_columns,
)
from starcraft_predictor.processing.pair_columns import PAIR_COLUMNS

preprocessing_pipeline = Pipeline(
    [
        (
            "column_difference_transformer",
            FunctionTransformer(
                calculate_column_difference, kw_args={"pair_columns": PAIR_COLUMNS}
            ),
        ),
        (
            "drop columns",
            FunctionTransformer(
                drop_columns,
                kw_args={"columns": PAIR_COLUMNS.all_columns},
            ),
        ),
        (
            "passthrough",
            "passthrough",
        ),
    ]
)
