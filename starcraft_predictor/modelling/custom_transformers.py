from typing import List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnDifferenceTransformer(BaseEstimator, TransformerMixin):
    """Class to calculate the difference between two columns in a pd.DataFrame.

    Parameters
    ----------
    diff_columns : list
        A list of tuples with column names. For example [(col1, col2), ...],
        where col2 will be subtracted from col1.
    new_columns : list
        List of columns names for the new column created from the subtraction.
    drop_cols : bool
        Boolean flag for whether to drop the old columns after the new column
        is created.
    """

    def __init__(
        self,
        diff_columns: List[tuple],
        new_columns: List[str],
        drop_cols: bool,
    ):

        super().__init__()

        if not isinstance(diff_columns, list):
            raise TypeError("diff columns should be a list of tuples")

        if not isinstance(new_columns, list):
            raise TypeError("new_columns should be a list")

        if not isinstance(drop_cols, bool):
            raise TypeError("drop_cols should be a bool")

        if len(diff_columns) != len(new_columns):
            raise ValueError(
                "diff_columns and new_columns should be the same length"
            )

        self.diff_columns = diff_columns
        self.new_columns = new_columns
        self.drop_cols = drop_cols

    def transform(self, X):
        """Calculates the differences for the provided list of column tuples.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe containing the columns in the diff_columns list.

        Returns
        -------
        X : pd.DataFrame
            Dataframe with the new difference columns, optionally with the old
            columns dropped.
        """

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pd.DataFrame")

        all_columns = [
            val for col_tuple in self.diff_columns for val in col_tuple
        ]

        for col in all_columns:
            missing_columns = []
            if col not in X.columns.values:
                missing_columns.append(col)
            if len(missing_columns) != 0:
                raise ValueError(f"Missing columns: {missing_columns}")

        X = X.copy()

        for i in range(len(self.new_columns)):
            X[self.new_columns[i]] = (
                X[self.diff_columns[i][0]] - X[self.diff_columns[i][1]]
            )

        if self.drop_cols:
            X.drop(columns=all_columns, inplace=True)

        return X
