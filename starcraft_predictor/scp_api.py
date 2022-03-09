import joblib
import pandas as pd
import numpy as np

from starcraft_predictor import (
    Replay,
    ReplayEngine,
    StarcraftModelEngine,
    sc2_preprocessing_pipeline,
    PlotEngine,
)


class ScpApi:
    """Main user API for evaluating SC2 replays"""

    model = StarcraftModelEngine(
        xgb_model=joblib.load(
            "../starcraft_predictor/modelling/trained_model.pkl"
        ),
    )

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
    def _get_plot_params(data: pd.DataFrame, predictions: np.ndarray):

        return {
            "df": pd.DataFrame({
                "seconds": data["seconds"],
                "win_prob": predictions,
            }),
            "p1_race": data["player_1_race"].values[0][0].lower(),
            "p2_race": data["player_2_race"].values[0][0].lower(),
        }

    @classmethod
    def _plot_predictions(cls, plot_params):

        plot = cls.plot_engine.win_probability_plot(**plot_params)

        print(plot)

    @classmethod
    def evaluate_replay(cls, path: str):
        replay = cls._load_replay(path=path)
        data = cls._process_replay(replay=replay)
        predictions = cls._generate_predictions(data=data)
        plot_params = cls._get_plot_params(
            data=data, predictions=predictions,
        )

        cls._plot_predictions(plot_params=plot_params)
