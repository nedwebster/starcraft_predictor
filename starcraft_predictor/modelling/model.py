import pandas as pd
import xgboost as xgb

from starcraft_predictor.modelling import model_params


class StarcraftModelEngine:
    """Model to predict win probability"""

    def __init__(self, xgb_model: xgb.XGBClassifier):
        self.model = xgb_model

    @classmethod
    def train_model(cls, data: pd.DataFrame):
        """Train a model from a training dataset"""

        # HOTFIX: Should be done as a transformer in pipeline
        for col in model_params.FEATURES:
            data[col] = data[col].astype("float")

        xgb_model = xgb.XGBClassifier(
            **model_params.PARAMS,
        )

        xgb_model.fit(
            X=data[model_params.FEATURES],
            y=data[model_params.RESPONSE],
            eval_metric="auc",
        )

        return cls(xgb_model=xgb_model)

    def predict(self, data: pd.DataFrame):
        """Generate probability predictons from a dataframe"""

        for col in model_params.FEATURES:
            data[col] = data[col].astype("float")

        predictions = self.model.predict_proba(
            data[model_params.FEATURES]
        )[:, 1]

        return predictions
