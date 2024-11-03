# handles predicting results
import os
from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb
from data import Data
from sklearn import metrics, model_selection


class Ai:
    """Class for training and predicting."""

    def __init__(self, train_new_model: Literal[True], model_path: str, data: Data):
        self.data = data
        if train_new_model or not os.path.exists(model_path):
            self.model = self.train_model()
            self.save_model()
        else:
            self.model = self.load_model_from_file(model_path)

    def train_model(self) -> xgb.XGBClassifier:
        """Return trained model."""
        train_matrix = self.data.get_train_matrix()
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            train_matrix[:, :-1], train_matrix[:, -1], test_size=0.1, random_state=2
        )
        model = xgb.XGBClassifier()
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_val)
        predictions = model.predict(x_val)
        prob = [probabilities[a][pred] for a, pred in enumerate(predictions)]
        print("Accuracy:", metrics.accuracy_score(y_val, predictions))
        print("Average confidence:", sum(prob) / len(prob))
        return model

    def get_probabilities(self, new_matches: pd.DataFrame) -> np.ndarray:
        """Get probabilities for match outcome [home_loss, home_win]."""
        x = [self.data.get_match_array(row) for _, row in new_matches.iterrows()]
        return self.model.predict_proba(x)

    def save_model(self) -> None:
        """Save ML model."""
        self.model.save_model("model.json")

    def load_model_from_file(self, path) -> xgb.XGBClassifier:
        """Load model from given file path."""
        return xgb.XGBClassifier.load_model(path)
