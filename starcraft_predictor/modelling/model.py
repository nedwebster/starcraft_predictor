import pandas as pd
import xgboost as xgb

from starcraft_predictor.modelling import model_params


class StarcraftModel:
    """Model to predict win probability"""

    def __init__(self, xgb_model: xgb.core.Booster):
        self.model = xgb_model

    @classmethod
    def train_model(cls, data: pd.DataFrame):
        """Train a model from a training dataset"""

        # HOTFIX: Should be done as a transformer in pipeline
        for col in model_params.FEATURES:
            data[col] = data[col].astype("float")

        dmatrix = xgb.DMatrix(
            data=data[model_params.FEATURES],
            label=data[model_params.RESPONSE],
        )

        xgb_model = xgb.train(
            params=model_params.PARAMS,
            dtrain=dmatrix,
            num_boost_round=model_params.NUM_BOOST_ROUND,
        )

        return cls(xgb_model=xgb_model)


    def predict(self, data: pd.DataFrame):
        """Generate probability predictons from a dataframe"""

        for col in model_params.FEATURES:
            data[col] = data[col].astype("float")

        dmatrix = xgb.DMatrix(
            data=data[model_params.FEATURES],
        )

        predictions = self.model.predict(dmatrix)

        return predictions
