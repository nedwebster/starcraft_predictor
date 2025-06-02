from starcraft_predictor.replays.processor import ReplayProcessor
from starcraft_predictor.processing.pipeline import preprocessing_pipeline
from starcraft_predictor.modelling.model import StarcraftModel
from starcraft_predictor.scorer import ReplayScorer
from starcraft_predictor.plots.plot_engine import PlotEngine
from starcraft_predictor.modelling.starcraft_shap import StarcraftShap


def load_scorer() -> ReplayScorer:
    replay_processor = ReplayProcessor()
    model = StarcraftModel.load()
    starcraft_shap = StarcraftShap(model=model.model, features=model.features)
    plot_engine = PlotEngine()

    replay_scorer = ReplayScorer(
        replay_processor=replay_processor,
        pipeline=preprocessing_pipeline,
        model=model,
        plot_engine=plot_engine,
        starcraft_shap=starcraft_shap,
    )

    return replay_scorer
