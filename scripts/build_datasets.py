"""
Script to process a batch of Starcraft II replays and save them into dataframes.

The user will need to update the REPLAY_PATH global variable to point to a directory on their machine containing the
replays.

There is a dataframe produced for each matchup. The script saves dataframes in the 'data/' directory, with filenames
based on the matchup type.

"""

import glob

import pandas as pd
import sc2reader

from starcraft_predictor.replays.matchup import Matchup
from starcraft_predictor.replays.replay_ingester import ReplayIngester

REPLAY_PATH = "/Users/nedwebster/Documents/python_projects/personal_projects/starcraft_predictor/data/replays/"

MATCHUPS = [
    "Terran vs Zerg",
    "Terran vs Terran",
    "Protoss vs Zerg",
    "Protoss vs Protoss",
    "Protoss vs Terran",
    "Zerg vs Zerg",
]

INGESTERS = {matchup: ReplayIngester(matchup) for matchup in MATCHUPS}

DATAFRAMES = {matchup: pd.DataFrame() for matchup in MATCHUPS}


def ingest_replays(replay_path: str) -> None:
    replay_paths = glob.glob(replay_path + "/**/*SC2Replay", recursive=True)
    for i, replay_path in enumerate(replay_paths):
        print(f"{i+1}/{len(replay_paths)}", end="\r")
        try:
            replay = sc2reader.load_replay(replay_path)
            matchup = Matchup.from_replay(replay)
            ingester = INGESTERS[matchup.value]
            data = ingester.ingest_replay(replay_path)
            DATAFRAMES[matchup.value] = pd.concat(
                [DATAFRAMES[matchup.value], data], ignore_index=True
            )
        except Exception as e:
            print(f"Error ingesting replay {replay_path}: {e}")
            continue


def validate_data(matchup: str, data: pd.DataFrame) -> None:
    print("Validating data for matchup: ", matchup)
    print("Data shape: ", data.shape)
    print("Number of games: ", data["filehash"].nunique())
    print("Number of missings: ", data.isnull().sum().sum())
    print("\n")


def save_dataframes():
    for matchup, df in DATAFRAMES.items():
        if df.shape[0] > 0:
            validate_data(matchup, df)
            df.to_pickle(f"data/{matchup.replace(' ', '_')}.pkl")


if __name__ == "__main__":
    data = ingest_replays(REPLAY_PATH)
    save_dataframes()
