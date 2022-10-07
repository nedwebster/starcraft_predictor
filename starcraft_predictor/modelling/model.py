import pandas as pd
import numpy as np
import xgboost as xgb

from starcraft_predictor.modelling import model_params

pd.options.mode.chained_assignment = None


class StarcraftModelEngine:
    """Model to predict win probability"""

    params = model_params

    def __init__(self, xgb_model: xgb.XGBClassifier):
        self.model = xgb_model

    @classmethod
    def train_model(cls, data: pd.DataFrame):
        """Train a model from a training dataset"""

        # HOTFIX: Should be done as a transformer in pipeline
        for col in cls.params.FEATURES:
            data[col] = data[col].astype("float")

        xgb_model = xgb.XGBClassifier(
            **cls.params.PARAMS,
        )

        xgb_model.fit(
            X=data[cls.params.FEATURES],
            y=data[cls.params.RESPONSE],
            eval_metric="auc",
        )

        return cls(xgb_model=xgb_model)

    def predict(self, data: pd.DataFrame, smoothed: bool = True):
        """Generate probability predictons from a dataframe. Predictions
        are generated for each filehash seperately so that they can be
        smoothed if required."""

        for col in self.params.FEATURES:
            data[col] = data[col].astype("float")

        prediction_list = []

        for filehash in data["filehash"].unique():

            data_subset = data[
                data["filehash"] == filehash
            ][self.params.FEATURES]
            subset_preds = self.model.predict_proba(data_subset)[:, 1]

            if smoothed:
                subset_preds = pd.Series(
                    subset_preds
                ).ewm(alpha=0.5).mean().values

            prediction_list.append(subset_preds)

        return np.concatenate(prediction_list)

    def __repr__(self):
        return f"StarcraftModel(xgb_model={self.model})"
