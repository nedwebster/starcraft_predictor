import numpy as np
import pandas as pd

from keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import sequence  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed  # type: ignore
from tensorflow.keras.models import Model, Sequential, load_model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore


class Sc2LSTM:
    """LSTM model for predicting sc2 game outcomes from time series data."""

    def __init__(
        self,
        features: list[str],
        target: str,
        unique_id: str,
        timestamp: str,
        lstm_params: dict | None = None,
        optimiser_params: dict | None = None,
    ):
        self.features = features
        self.target = target
        self.unique_id = unique_id
        self.timestamp = timestamp
        self.lstm_params = self.get_lstm_params(lstm_params)
        self.optimiser_params = optimiser_params or {
            "optimizer": Adam(learning_rate=0.01),
            "loss": "binary_crossentropy",
            "metrics": ["AUC"],
        }

        self.model = self.init_model()
        self.scaler = None
        self.history = None

    @staticmethod
    def get_lstm_params(lstm_params: dict | None = None) -> dict:
        """Set the LSTM parameters, ensuring that the return_sequences is True."""
        lstm_params = lstm_params or {
            "units": 64,
            "dropout": 0.01,
            "kernel_regularizer": l2(0.01),
            "recurrent_regularizer": l2(0.01),
            "bias_regularizer": l2(0.01),
        }
        lstm_params["return_sequences"] = True
        return lstm_params

    def get_feature_array(self, data: pd.DataFrame) -> np.ndarray:
        """Get the feature array for the LSTM model."""
        return data[self.features].values.reshape(1, data.shape[0], len(self.features))

    def get_target_array(self, data: pd.DataFrame) -> np.ndarray:
        """Get the target array for the LSTM model."""
        return data[[self.target]].values.reshape(1, data.shape[0], 1)

    def init_model(self) -> Model:
        """Build the LSTM model architecture."""
        lstm_layer = LSTM(**self.lstm_params)
        output_layer = TimeDistributed(Dense(1, activation="sigmoid", name="win_probability"))

        model = Sequential(layers=[lstm_layer, output_layer])
        model.compile(**self.optimiser_params)

        self.model = model

    def prepare_training_data(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare the training data for the LSTM model."""
        data = data.copy()
        data.sort_values(by=[self.unique_id, self.timestamp], inplace=True)
        groups = data.groupby(self.unique_id)
        n_examples = data[self.unique_id].nunique()
        n_timesteps = groups[self.target].count().max()

        X_sequences = []
        y_sequences = []

        for group in groups:
            X_data = group[1][self.features].values
            y_data = [group[1][self.target].max()] * n_timesteps
            X_sequences.append(X_data)
            y_sequences.append(y_data)
        X_sequences = sequence.pad_sequences(X_sequences, dtype=float)
        y_sequences = np.array(y_sequences).reshape(n_examples, n_timesteps, 1)

        return X_sequences, y_sequences

    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> None:
        """Train the LSTM model."""
        self.init_model()

        train_X, train_y = self.prepare_training_data(train_data)

        if validation_data is not None:
            validation_arrays = self.prepare_training_data(validation_data)

        callbacks = [
            EarlyStopping(
                monitor="val_loss" if validation_data is not None else "loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if validation_data is not None else "loss",
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        self.history = self.model.fit(
            train_X,
            train_y,
            validation_data=validation_arrays,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict(self, data: pd.DataFrame, verbose: int = 0) -> pd.DataFrame:
        """Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted probabilities for each timestamp

        """
        data = data.copy()
        data.sort_values(by=[self.unique_id, self.timestamp], inplace=True)
        groups = data.groupby(self.unique_id)
        dataframes = []
        for group in groups:
            new_df = group[1][[self.unique_id, self.timestamp]].copy()
            reshaped_features = self.get_feature_array(group[1])
            preds = self.model.predict(reshaped_features, verbose=verbose)
            new_df["predictions"] = preds[0, :, 0]
            dataframes.append(new_df)

        return pd.concat(dataframes)

    def evaluate(self, data: pd.DataFrame) -> dict:
        """Evaluate the model on test data."""

        X_test = self.get_feature_array(data)
        y_test = self.get_target_array(data)

        # Calculate metrics across all timestamps
        test_loss, test_acc = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        return {
            "test_loss": test_loss,
            "test_AUC": test_acc,
        }

    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # Loss
        axes[0].plot(self.history.history["loss"], label="Training Loss")
        if "val_loss" in self.history.history:
            axes[0].plot(self.history.history["val_loss"], label="Validation Loss")
        axes[0].set_title("Model Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()

        # AUC
        axes[1].plot(self.history.history["AUC"], label="Training AUC")
        if "val_AUC" in self.history.history:
            axes[1].plot(
                self.history.history["val_AUC"], label="Validation AUC"
            )
        axes[1].set_title("Model AUC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("AUC")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = load_model(filepath)
