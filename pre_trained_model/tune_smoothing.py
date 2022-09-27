import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append("..")
import starcraft_predictor as scp  # noqa: E402


EWM_ALPHA_LIST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def load_data(path: str) -> pd.DataFrame:
    """Load the data and subset to just test data"""

    data = pd.read_csv(path)
    data = data[data["sample"] == "test"]

    return data


def generate_smoothed_predictions(
    data: pd.DataFrame, alpha: float
) -> np.ndarray:
    """Generate smoothed predictions for a given alpha"""

    predictions = scp.starcraft_model.predict(data, smoothed=False)
    smoothed_predictions = np.array(
        pd.Series(predictions).ewm(alpha=alpha).mean()
    )

    return smoothed_predictions


def generate_auc_scores(data: pd.DataFrame) -> dict:
    """Build AUC scores for base predictions, as well as
    each EWM iteration specified"""

    auc_scores = {}

    base_predictions = scp.starcraft_model.predict(
        data, smoothed=False,
    )

    auc_scores[0] = roc_auc_score(data["winner"], base_predictions)

    for alpha in EWM_ALPHA_LIST:
        smoothed_predictions = generate_smoothed_predictions(
            data=data, alpha=alpha,
        )

        auc_scores[alpha] = roc_auc_score(data["winner"], smoothed_predictions)

    return auc_scores


def evaluate_auc_scores(auc_scores: dict):

    for alpha, auc in auc_scores.items():
        print(f"alpha={alpha}, auc={auc}")

    max_key = max(auc_scores, key=auc_scores.get)

    print(f"\nBest Alpha: {max_key}")

    return max_key


def main():

    path = (
        "C:/Users/Edward/Documents/python_project"
        "/sc2_data/transformed_data.csv"
    )

    data = load_data(path)

    auc_scores = generate_auc_scores(data)

    evaluate_auc_scores(auc_scores)


if __name__ == "__main__":

    main()
