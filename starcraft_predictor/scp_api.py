import joblib
import os

import pandas as pd
import numpy as np

from starcraft_predictor import (
    Replay,
    ReplayEngine,
    sc2_preprocessing_pipeline,
    PlotEngine,
    StarcraftModelEngine,
    StarcraftShap
)


def load_pretrained_model():
    PACKAGE_INSTALLATION_PATH = os.path.dirname(
        os.path.abspath(__file__)
    )

    trained_model = joblib.load(
        PACKAGE_INSTALLATION_PATH + "/modelling/trained_model.pkl"
    )
    starcraft_model = StarcraftModelEngine(
        xgb_model=trained_model,
    )

    return starcraft_model


PRE_TRAINED_MODEL = load_pretrained_model()


class ScpApi:
    """Main user API for evaluating SC2 replays"""

    model = PRE_TRAINED_MODEL

    plot_engine = PlotEngine()

    def __init__(self):
        pass

    @staticmethod
    def _load_replay(path: str):
        return Replay.from_path(path=path)

    @staticmethod
    def _process_replay(replay: Replay):
        data = ReplayEngine.build_dataframe(
            replay,
        )

        return sc2_preprocessing_pipeline.transform(
            data,
        )

    @classmethod
    def _generate_predictions(cls, data: pd.DataFrame):

        predictions = cls.model.predict(
            data,
        )

        return predictions

    @staticmethod
    def _get_plot_params(
        data: pd.DataFrame,
        predictions: np.ndarray,
        moment: tuple,
    ):

        return {
            "df": pd.DataFrame({
                "seconds": data["seconds"],
                "win_prob": predictions,
            }),
            "p1_race": data["player_1_race"].values[0][0].lower(),
            "p2_race": data["player_2_race"].values[0][0].lower(),
            "moment": moment,
        }

    @classmethod
    def _plot_predictions(cls, plot_params):

        plot = cls.plot_engine.win_probability_plot(**plot_params)

        return plot

    @classmethod
    def evaluate_replay(cls, path: str, moment: bool = False):
        replay = cls._load_replay(path=path)
        data = cls._process_replay(replay=replay)
        predictions = cls._generate_predictions(data=data)

        if moment:
            moment = StarcraftShap(
                processed_replay=data,
                features=cls.model.params.FEATURES,
                predictions=predictions,
                model=cls.model.model,
            ).get_moment()

        plot_params = cls._get_plot_params(
            data=data, predictions=predictions, moment=moment,
        )

        cls._plot_predictions(plot_params=plot_params)
