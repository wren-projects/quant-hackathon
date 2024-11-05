# handles predicting results
import os

import numpy as np
import xgboost as xgb
from sklearn import metrics, model_selection


class Ai:
    """Class for training and predicting."""

    model: xgb.XGBClassifier

    def __init__(self, model: xgb.XGBClassifier):
        """Create a new Model from a XGBClassifier."""
        self.model = model

    @staticmethod
    def untrained() -> "Ai":
        """Get model type."""
        return Ai(xgb.XGBClassifier())

    @staticmethod
    def load_from_file(path: os.PathLike) -> "Ai":
        """Load model from given file path."""
        return Ai(xgb.XGBClassifier.load_model(path))

    def train(self, train_matrix: np.ndarray) -> None:
        """Return trained model."""
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            train_matrix[:, :-1],
            train_matrix[:, -1],
            test_size=0.1,
            random_state=6,
        )
        self.model.fit(x_train, y_train)
        probabilities = self.model.predict_proba(x_val)
        predictions = self.model.predict(x_val)
        prob = [probabilities[i][pred] for i, pred in enumerate(predictions)]
        print("Accuracy:", metrics.accuracy_score(y_val, predictions))
        print("Average confidence:", sum(prob) / len(prob))

    def get_probabilities(self, data_matrix: np.ndarray) -> np.ndarray:
        """Get probabilities for match outcome [home_loss, home_win]."""
        return self.model.predict_proba(data_matrix)

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)
