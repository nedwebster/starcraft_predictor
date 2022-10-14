"""Script to build the training dataset from a local store of
repalys"""

from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.append("..")

import starcraft_predictor as scp  # noqa

np.random.seed(2709)


def load_replays(parent_path: str, chunk_number, chunk_size) -> list:
    """Loads the replays into a list of scp.Replay objects"""

    replay_paths = []

    for path in Path(parent_path).rglob("*SC2Replay"):
        replay_paths.append(str(path))

    print("Loading replays...")
    replays = []
    for i, path in enumerate(replay_paths[
        chunk_number*chunk_size:(chunk_number+1)*(chunk_size)
    ]):
        print(f"{i+1}/{len(replay_paths)}", end="\r")

        try:
            replays.append(scp.Replay.from_path(path=path))

        # TODO: Investigate what is causing these errors
        except (IndexError, AttributeError, ValueError) as e:  # noqa: F841
            print(f"{i+1} failed: {e}")

    print("\nReplays loaded")
    return replays


def build_dataframe(replays: list) -> pd.DataFrame:
    """Convert the replays into dataframes using the scp.ReplayEngine,
    then transform the data to be model ready"""

    print("Building dataframe...")
    replay_dataframe = scp.ReplayEngine.build_batch(
        replays=replays,
    )

    # note: the pipeline learns no information, hence no fit is needed
    transformed_data = scp.sc2_preprocessing_pipeline.transform(
        replay_dataframe,
    )

    print("Dataframe built")
    return transformed_data


def build_sample_column(data: pd.DataFrame):
    """Build a train/test sample column, while grouping a given replays
    observations"""

    sample_map = {
        k: np.random.rand() for k in data["filehash"].unique()
    }

    data["sample"] = data["filehash"].map(sample_map)
    data["sample"] = np.where(
        data["sample"] < 0.8, "train", "test",
    )

    return data


def main():

    local_path = (
        "C:/Users/Edward/Documents/replays"
    )

    for i in [2]:
        replays = load_replays(
            parent_path=local_path,
            chunk_number=i,
            chunk_size=500,
        )
        transformed_dataset = build_dataframe(replays=replays)
        transformed_dataset = build_sample_column(transformed_dataset)

        transformed_dataset.to_csv(
            "C:/Users/Edward/Documents/python_project/sc2_data/"
            f"transformed_data_{i}.csv",
            index=False,
        )

        del transformed_dataset


if __name__ == "__main__":

    main()
