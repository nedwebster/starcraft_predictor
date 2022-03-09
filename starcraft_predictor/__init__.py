# flake8: noqa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import joblib

from .plot_engine import PlotEngine
from .replay_engine import ReplayEngine
from .replay import Replay
from .modelling import StarcraftModelEngine
from .modelling import sc2_preprocessing_pipeline
from .scp_api import ScpApi as api

# load pre-trained xgboost model
trained_model = joblib.load(
    "../starcraft_predictor/modelling/trained_model.pkl"
)
starcraft_model = StarcraftModelEngine(
    xgb_model=trained_model,
)

del trained_model, warnings, joblib
