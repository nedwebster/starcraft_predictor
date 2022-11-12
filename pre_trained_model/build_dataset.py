"""Script to build the training dataset from a local store of
repalys"""

import glob
import pandas as pd
import numpy as np

import sys
sys.path.append("..")

import starcraft_predictor as scp  # noqa

np.random.seed(2709)


def load_replays(path: str) -> list:
    """Loads the replays into a list of scp.Replay objects"""

    replay_paths = glob.glob(path + "/*SC2Replay")
    print("Loading replays...")
    replays = []
    for i, path in enumerate(replay_paths):
        print(f"{i+1}/{len(replay_paths)}", end="\r")

        try:
            replays.append(scp.Replay.from_path(path=path))

        # TODO: Investigate what is causing these errors
        except (IndexError, AttributeError) as e:  # noqa: F841
            print(f"{i+1} failed: {e}")

    print("\nReplays loaded")
    return replays


def build_dataframe(replays: list) -> pd.DataFrame:
    """Convert the replays into dataframes using the scp.ReplayEngine,
    then transform the data to be model ready"""

    print("Building dataframe...")
    replay_dataframe = scp.ReplayEngine.process_batch(
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
        "C:/Users/Edward/Documents/python_project/sc2_data/replays"
    )

    replays = load_replays(path=local_path)
    transformed_dataset = build_dataframe(replays=replays)
    transformed_dataset = build_sample_column(transformed_dataset)

    transformed_dataset.to_csv(
        "C:/Users/Edward/Documents/python_project/sc2_data/"
        "transformed_data.csv",
        index=False,
    )

    return transformed_dataset


if __name__ == "__main__":

    main()
