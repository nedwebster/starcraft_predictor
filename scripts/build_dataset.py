import glob

import numpy as np
import pandas as pd
import starcraft_predictor as scp
from starcraft_predictor.replays.replay import Replay


np.random.seed(2709)


def load_replays(path: str) -> list:
    """Loads the replays into a list of scp.Replay objects"""

    replay_paths = glob.glob(path + "/**/*SC2Replay", recursive=True)
    print("Loading replays...")
    replays = []
    for i, path in enumerate(replay_paths):
        print(f"{i+1}/{len(replay_paths)}", end="\r")

        try:
            replay = scp.Replay.from_path(path=path)
            replays.append(replay)
        except Exception as e:
            print(f"{i+1} failed: {e}")

    print("\nReplays loaded")
    return replays


def build_dataframe(replays: list[Replay]) -> pd.DataFrame:
    print("Building dataframe...")
    replay_dataframe = scp.replay_processor.process_batch(replays=replays)

    # note: the pipeline learns no information, hence no fit is needed
    transformed_data = scp.preprocessing_pipeline.transform(
        replay_dataframe,
    )

    print("Dataframe built")
    return transformed_data


def build_sample_column(data: pd.DataFrame):
    """Build a train/test sample column, while grouping a given replays
    observations"""

    sample_map = {k: np.random.rand() for k in data["filehash"].unique()}

    data["sample"] = data["filehash"].map(sample_map)
    data["sample"] = np.where(
        data["sample"] < 0.8,
        "train",
        "test",
    )

    return data


def main():
    local_path = "/Users/nedwebster/Documents/python_projects/personal_projects/starcraft_predictor/data/HSC26"
    replays = load_replays(path=local_path)
    data = build_dataframe(replays)
    data = build_sample_column(data)
    data.to_pickle("transformed_data.pkl")


if __name__ == "__main__":
    main()
