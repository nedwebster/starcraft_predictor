from sklearn.pipeline import Pipeline

from starcraft_predictor.replays.replay import Replay
from starcraft_predictor.replays.processor import ReplayProcessor
from starcraft_predictor.modelling.model import StarcraftModel
from starcraft_predictor.plots.plot_engine import PlotEngine
from starcraft_predictor.modelling.starcraft_shap import StarcraftShap


class ReplayScorer:

    def __init__(
        self,
        replay_processor: ReplayProcessor,
        pipeline: Pipeline,
        model: StarcraftModel,
        plot_engine: PlotEngine,
        starcraft_shap: StarcraftShap,
    ):
        self.replay_processor = replay_processor
        self.pipeline = pipeline
        self.model = model
        self.plot_engine = plot_engine
        self.starcraft_shap = starcraft_shap

    def score_replay(self, replay_path: str) -> None:
        """
        Score a Replay object.
        """
        replay = Replay.from_path(replay_path)
        data = self.replay_processor.process_replay(replay)
        data = self.pipeline.transform(data)
        predictions = self.model.predict(data)
        moment = self.starcraft_shap.get_moment(data, predictions)

        fig = self.plot_engine.plot_win_probability(
            replay=replay,
            predictions=predictions,
            moment=moment,
        )
        return fig
