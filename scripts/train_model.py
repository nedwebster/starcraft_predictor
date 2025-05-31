import pandas as pd
from sklearn.metrics import roc_auc_score

from starcraft_predictor.modelling.model import StarcraftModel


def load_data(path: str) -> pd.DataFrame:
    """Load formated dataframe output from build_dataset.py"""

    print("Loading Data...")
    data = pd.read_pickle(path)

    return data


def split_data(data: pd.DataFrame) -> tuple:
    """Split data based on 'sample' column"""

    train_data = data[data["sample"] == "train"]
    test_data = data[data["sample"] == "test"]

    return train_data, test_data


def train_model(data: pd.DataFrame):
    model = StarcraftModel()
    model.train_model(data=data)
    model.save()
    return model


def print_metrics(
    model: StarcraftModel,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
):
    """Prints the AUC for train and test samples"""

    train_preds = model.predict(train_data)
    test_preds = model.predict(test_data)

    train_auc = roc_auc_score(
        y_true=train_data[model.target],
        y_score=train_preds,
    )

    test_auc = roc_auc_score(
        y_true=test_data[model.target],
        y_score=test_preds,
    )

    print(f"Train AUC: {train_auc}")
    print(f"Test AUC: {test_auc}")


def main():

    local_path = (
        "/Users/nedwebster/Documents/python_projects/personal_projects/starcraft_predictor/"
        "transformed_data.pkl"
    )
    data = load_data(path=local_path)
    train_data, test_data = split_data(data)

    model = train_model(train_data)

    print_metrics(
        model=model,
        train_data=train_data,
        test_data=test_data,
    )


if __name__ == "__main__":

    main()
