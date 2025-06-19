import hashlib

import pandas as pd


def hash_split(
    df: pd.DataFrame,
    id_column: str,
    n_splits: int = 2,
    salt: str | None = None,
) -> list[pd.DataFrame]:
    """Split a DataFrame into n parts based on a hashed identifier column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    id_column : str
        Name of the column containing identifiers to hash
    n_splits : int
        Number of splits to create (default: 2)
    salt : str, optional
        Salt to add to the hash for additional randomization

    Returns
    -------
    list[pd.DataFrame]
        List of n DataFrames, each containing a subset of the original data

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found in DataFrame")

    # Create hash values for each row
    def hash_value(x: str) -> int:
        if salt:
            x = str(x) + salt
        return int(hashlib.sha256(str(x).encode()).hexdigest(), 16)

    # Apply hash function and get modulo
    hash_values = df[id_column].apply(hash_value)
    split_indices = hash_values % n_splits

    # Split DataFrame based on hash values
    splits = []
    for i in range(n_splits):
        split_df = df[split_indices == i].copy()
        splits.append(split_df)

    return splits


def train_test_split(
    df: pd.DataFrame,
    id_column: str,
    test_size: float = 0.2,
    salt: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets based on a hashed identifier.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    id_column : str
        Name of the column containing identifiers to hash
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    salt : str, optional
        Salt to add to the hash for additional randomization

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames

    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    # Calculate number of splits needed to achieve desired test size
    n_splits = int(1 / test_size)

    # Get splits
    splits = hash_split(df, id_column, n_splits, salt)

    # Combine all splits except the last one for training
    train_df = pd.concat(splits[:-1])
    test_df = splits[-1]

    return train_df, test_df
