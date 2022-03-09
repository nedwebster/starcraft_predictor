from boruta import BorutaPy
import joblib
import sys
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd  # noqa
import xgboost as xgb  # noqa

sys.path.append("..")
from starcraft_predictor.modelling import model_params  # noqa


def load_data(path: str) -> pd.DataFrame:
    """Load formated dataframe output from build_dataset.py"""

    print("Loading Data...")
    data = pd.read_csv(path)

    return data


def split_data(data: pd.DataFrame) -> tuple:
    """Split data based on 'sample' column"""

    train_data = data[data["sample"] == "train"]
    test_data = data[data["sample"] == "test"]

    return train_data, test_data


def perform_boruta_feat_selection(data: pd.DataFrame) -> list:
    """Boruta feature selection to choose the final feature list"""

    print("Beforming Boruta Feature Selection...")
    model = xgb.XGBClassifier(
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=5,
        learning_rate=0.1,
        verbosity=0,
        random_state=2709,
    )

    feat_selector = BorutaPy(
        model,
        n_estimators=300,
        verbose=1,
        random_state=2709,
    )

    feat_selector.fit(
        data[model_params.FEATURES].values,
        data[model_params.RESPONSE].values,
    )

    selected_features = []
    for i, feat in enumerate(model_params.FEATURES):
        if feat_selector.support_[i]:
            selected_features.append(feat)

    print(f"\nSelected Features:\n{selected_features}")
    print("\n")

    return selected_features


def perform_grid_search(
    data: pd.DataFrame, features: list, param_grid: dict
) -> dict:
    """Returns the best parameters from the grid search"""

    print("Performing Random Grid Search...")

    model = xgb.XGBClassifier(verbosity=0, random_state=2709)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=15,
        cv=5,
        verbose=1,
        random_state=2709,
    )

    rs_output = random_search.fit(
        data[features],
        data[model_params.RESPONSE],
        eval_metric="auc",
    )

    print(f"Chosen Parameters:\n{rs_output.best_params_}")
    print("\n")
    return rs_output.best_params_


def build_final_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list,
    params: dict
) -> xgb.XGBClassifier:
    """Build the final model using the selected feature list and the
    best parameters"""

    print("Building final model...")
    params["random_state"] = 2709
    model = xgb.XGBClassifier(**params)
    model.fit(
        X=train_data[features],
        y=train_data[model_params.RESPONSE],
        eval_set=[(test_data[features], test_data[model_params.RESPONSE])],
        eval_metric="auc",
        early_stopping_rounds=20,
    )

    print("Final Model Built\n")
    return model


def print_metrics(
    model: xgb.XGBClassifier,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: list
):
    """Prints the AUC for train and test samples"""

    train_preds = model.predict_proba(train_data[features])[:, 1]
    test_preds = model.predict_proba(test_data[features])[:, 1]

    train_auc = roc_auc_score(
        y_true=train_data[model_params.RESPONSE],
        y_score=train_preds,
    )

    test_auc = roc_auc_score(
        y_true=test_data[model_params.RESPONSE],
        y_score=test_preds,
    )

    print(f"Train AUC: {train_auc}")
    print(f"Test AUC: {test_auc}")


def main():

    local_path = (
        "C:/Users/Ned/OneDrive/Documents"
        "/Python Projects/data/transformed_data.csv"
    )

    data = load_data(path=local_path)
    train_data, test_data = split_data(data)

    features = perform_boruta_feat_selection(data=train_data)

    param_grid = {
        "n_estimators": [100, 300, 500, 1000],
        "learning_rate": [0.05, 0.1, 0.2, 0.3],
        "max_depth": [2, 3, 4],
        "subsample": [0.5, 0.75, 0.9],
        "colsample_bytree": [0.5, 0.75, 0.9],
    }

    best_params = perform_grid_search(
        data=train_data, features=features, param_grid=param_grid
    )

    final_model = build_final_model(
        train_data=train_data,
        test_data=test_data,
        features=features,
        params=best_params,
    )

    model_path = "../starcraft_predictor/modelling/trained_model.pkl"
    joblib.dump(final_model, model_path)

    print_metrics(
        model=final_model,
        train_data=train_data,
        test_data=test_data,
        features=features,
    )


if __name__ == "__main__":

    main()
