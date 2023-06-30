import numpy as np
import pandas as pd
from starcraft_predictor import StarcraftModelEngine
from starcraft_predictor.modelling import model_params
import xgboost as xgb


class TestStarcraftModelEngine:

    def test_init(self):

        sme = StarcraftModelEngine(xgb.XGBClassifier())

        assert isinstance(sme.model, xgb.XGBClassifier)

    def test_train_model(self, mocker):

        mocker.patch.object(
            xgb.XGBClassifier,
            "fit",
        )

        output = StarcraftModelEngine.train_model(
            data=pd.DataFrame(
                columns=model_params.FEATURES + [model_params.RESPONSE]
            )
        )

        assert isinstance(output.model, xgb.XGBClassifier)

    def test_predict_no_smoothing(self, mocker):

        mocker.patch.object(
            xgb.XGBClassifier,
            "predict_proba",
            return_value=np.array([[1, 0]]),
        )

        test_data = pd.DataFrame(
            columns=model_params.FEATURES + [
                model_params.RESPONSE, "filehash"
            ],
        )
        test_data.loc[0, :] = 0

        sme = StarcraftModelEngine(xgb.XGBClassifier())

        predictions = sme.predict(test_data, smoothed=False)

        assert predictions == np.array([0])

    def test_predict_with_smoothing(self, mocker):

        mocker.patch.object(
            xgb.XGBClassifier,
            "predict_proba",
            return_value=np.array([[1, 0]]),
        )

        mocker.patch.object(
            pd.Series,
            "ewm",
            return_value=pd.Series([0])
        )

        mocker.patch.object(
            pd.Series,
            "mean",
            return_value=pd.Series([0])
        )

        test_data = pd.DataFrame(
            columns=model_params.FEATURES + [
                model_params.RESPONSE, "filehash"
            ],
        )
        test_data.loc[0, :] = 0

        sme = StarcraftModelEngine(xgb.XGBClassifier())

        predictions = sme.predict(test_data, smoothed=True)

        assert predictions == np.array([0])
