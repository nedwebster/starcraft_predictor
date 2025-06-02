import pickle
import os
from typing import Any

import pandas as pd
import numpy as np
import xgboost as xgb

from starcraft_predictor.modelling.model_params import (
    PARAMS,
    FEATURES,
    TARGET,
    UNIQUE_ID,
)


MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + "/scp_model.pkl"


class StarcraftModel:
    """Model to predict win probability"""

    def __init__(
        self,
        params: dict[str, Any] = PARAMS,
        features: list[str] = FEATURES,
        target: str = TARGET,
        unique_id: str = UNIQUE_ID,
        model: xgb.XGBClassifier | None = None,
        smoothing_ewm: float = 0.5,
    ):
        self.params = params
        self.features = features
        self.target = target
        self.model = model or xgb.XGBClassifier(**self.params)
        self.unique_id = unique_id
        self.smoothing_ewm = smoothing_ewm

    def train_model(self, data: pd.DataFrame):
        """Train a model from a training dataset"""

        self.model.fit(
            X=data[self.features],
            y=data[self.target],
        )

    def predict(self, data: pd.DataFrame, smoothed: bool = True):
        """
        Generate probability predictons from a dataframe.

        Predictions are generated for each unique_id seperately so that they can be smoothed if required.
        """

        prediction_list = []

        for filehash in data[self.unique_id].unique():

            data_subset = data[data[self.unique_id] == filehash][self.features]
            subset_preds = self.model.predict_proba(data_subset)[:, 1]

            if smoothed:
                subset_preds = (
                    pd.Series(subset_preds).ewm(alpha=self.smoothing_ewm).mean().values
                )

            prediction_list.append(subset_preds)

        return np.concatenate(prediction_list)

    def save(self, path: str = MODEL_PATH):
        """Save the model to a file, by default this is the package location."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(self, path: str = MODEL_PATH) -> "StarcraftModel":
        """Load the model from a file, by default this is the package location."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return f"StarcraftModel(params={self.params}, features={self.features}, target={self.target}, unique_id={self.unique_id})"
