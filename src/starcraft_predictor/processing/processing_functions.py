import pandas as pd

from starcraft_predictor.processing.pair_columns import PairColumns


def calculate_column_difference(X, pair_columns: PairColumns) -> pd.DataFrame:
    """
    Generate 'difference' between pairs of columns in the dataframe, usable inside a scikit-learn FunctionTransformer.

    Eg, given a tuple of columns ('A', 'B'), the function will create a new column A - B.

    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X should be a pd.DataFrame")

    missing_columns = [x for x in pair_columns.all_columns if x not in X.columns.values]
    if len(missing_columns) != 0:
        raise ValueError(f"Missing columns: {missing_columns}")

    X = X.copy()

    new_columns = [x + "_diff" for x in pair_columns.base_columns]

    for i, col in enumerate(new_columns):
        X[col] = (
            X[pair_columns.column_tuples[i][0]] - X[pair_columns.column_tuples[i][1]]
        )

    return X


def drop_columns(X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns from the dataframe, usable inside a scikit-learn FunctionTransformer."""
    X.drop(columns=columns, inplace=True)
    return X
