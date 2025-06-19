import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Input, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class Sc2LSTM:
    """LSTM model for predicting sc2 game outcomes from time series data."""

    def __init__(
        self,
        lstm_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
    ):
        """Initialize the LSTM model.

        Args:
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization

        """
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def build_model(self, sequence_length: int, n_features: int = 2) -> Model:
        """Build the LSTM model architecture.

        Args:
            sequence_length: Length of input sequences (minutes in game)
            n_features: Number of features (possession_pct, tackles)

        Returns:
            Compiled Keras model

        """
        # Input layer
        inputs = Input(shape=(sequence_length, n_features), name="match_features")

        # LSTM layers with return_sequences=True for many-to-many prediction
        lstm_out = LSTM(
            units=self.lstm_units,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name="lstm_layer",
        )(inputs)

        # Additional LSTM layer for more complex patterns
        lstm_out = LSTM(
            units=self.lstm_units // 2,
            return_sequences=True,
            dropout=self.dropout_rate,
            recurrent_dropout=self.dropout_rate,
            name="lstm_layer_2",
        )(lstm_out)

        # TimeDistributed Dense layer for predictions at each timestamp
        predictions = TimeDistributed(
            Dense(1, activation="sigmoid", name="win_probability"),
            name="time_distributed_output",
        )(lstm_out)

        # Create and compile model
        model = Model(inputs=inputs, outputs=predictions, name="sports_match_lstm")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"],
        )

        self.model = model
        return model

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        id_column: str = "filehash",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and preprocess the training data from a pandas DataFrame.

        Args:
            df: DataFrame containing match data with id column to separate matches
            feature_columns: List of column names to use as features
            target_column: Name of the target column (match winner)
            id_column: Name of the column that identifies different matches
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test

        """
        # Group data by match id
        match_groups = df.groupby(id_column)
        
        # Get unique match ids and their winners
        match_ids = df[id_column].unique()
        match_winners = []
        match_data = []
        
        # Process each match
        for match_id in match_ids:
            match_df = match_groups.get_group(match_id)
            
            # Extract features for this match
            features = match_df[feature_columns].values
            match_data.append(features)
            
            # Get the winner for this match (should be consistent across all rows)
            winner = match_df[target_column].iloc[0]
            match_winners.append(winner)
        
        # Find the maximum sequence length
        max_length = max(len(match) for match in match_data)
        n_features = len(feature_columns)
        n_matches = len(match_data)

        # Initialize arrays
        X = np.zeros((n_matches, max_length, n_features))
        y = np.zeros((n_matches, max_length, 1))

        # Process each match
        for i, (match, winner) in enumerate(
            zip(match_data, match_winners, strict=False)
        ):
            match_length = len(match)

            # Pad sequences to max_length (pad with zeros at the end)
            X[i, :match_length, :] = match

            # Create target sequence - winner probability at each timestamp
            # The true winner is known throughout the game
            y[i, :match_length, 0] = winner

        # Normalize features
        # Reshape for scaling
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_matches, max_length, n_features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=match_winners,
        )

        return X_train, X_test, y_train, y_test

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> None:
        """Train the LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level

        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        # Train model
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Input features

        Returns:
            Predicted probabilities for each timestamp

        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        id_column: str = "filehash",
    ) -> dict[str, np.ndarray]:
        """Make predictions on new data from a DataFrame.

        Args:
            df: DataFrame containing match data with id column to separate matches
            feature_columns: List of column names to use as features
            id_column: Name of the column that identifies different matches

        Returns:
            Dictionary mapping match_id to predicted probabilities for each timestamp

        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Group data by match id
        match_groups = df.groupby(id_column)
        
        # Get unique match ids
        match_ids = df[id_column].unique()
        match_data = []
        
        # Process each match
        for match_id in match_ids:
            match_df = match_groups.get_group(match_id)
            
            # Extract features for this match
            features = match_df[feature_columns].values
            match_data.append(features)
        
        # Find the maximum sequence length from training (stored in scaler)
        # We'll use the same max_length as was used during training
        # For now, let's use the current max length from the data
        max_length = max(len(match) for match in match_data)
        n_features = len(feature_columns)
        n_matches = len(match_data)

        # Initialize array
        X = np.zeros((n_matches, max_length, n_features))

        # Process each match
        for i, match in enumerate(match_data):
            match_length = len(match)
            # Pad sequences to max_length (pad with zeros at the end)
            X[i, :match_length, :] = match

        # Normalize features using the existing scaler
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X = X_scaled.reshape(n_matches, max_length, n_features)

        # Make predictions
        predictions = self.model.predict(X)
        
        # Return as dictionary mapping match_id to predictions
        result = {}
        for i, match_id in enumerate(match_ids):
            result[match_id] = predictions[i]
            
        return result

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on test data.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics

        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Calculate metrics across all timestamps
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )

        # Calculate F1 score
        f1_score = (
            2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        )

        return {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1_score": f1_score,
        }

    def plot_training_history(self) -> None:
        """Plot training history."""
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss
        axes[0, 0].plot(self.history.history["loss"], label="Training Loss")
        if "val_loss" in self.history.history:
            axes[0, 0].plot(self.history.history["val_loss"], label="Validation Loss")
        axes[0, 0].set_title("Model Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()

        # Accuracy
        axes[0, 1].plot(self.history.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in self.history.history:
            axes[0, 1].plot(
                self.history.history["val_accuracy"], label="Validation Accuracy"
            )
        axes[0, 1].set_title("Model Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].legend()

        # Precision
        axes[1, 0].plot(self.history.history["precision"], label="Training Precision")
        if "val_precision" in self.history.history:
            axes[1, 0].plot(
                self.history.history["val_precision"], label="Validation Precision"
            )
        axes[1, 0].set_title("Model Precision")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].legend()

        # Recall
        axes[1, 1].plot(self.history.history["recall"], label="Training Recall")
        if "val_recall" in self.history.history:
            axes[1, 1].plot(
                self.history.history["val_recall"], label="Validation Recall"
            )
        axes[1, 1].set_title("Model Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save(filepath)

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        self.model = tf.keras.models.load_model(filepath)
