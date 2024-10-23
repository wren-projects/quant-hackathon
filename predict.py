# handles predicting results
from analysis import Data

import numpy as np
import xgboost as xgb
from sklearn import model_selection
import os


class Ai:
    def __init__(self, train_new_model, model_path, data: Data):
        self.model = None
        self.data = data
        if train_new_model or not os.path.exists(model_path):
            self.model = self.train_model()
            self.save_model()
        else:
            self.model = self.load_model_from_file(model_path)

    def train_model(self):
        """returns trained model"""
        train_matrix = self.data.get_train_matrix()
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            train_matrix[:, :-1], train_matrix[:, -1], test_size=0.1, random_state=2
        )

        model = xgb.XGBClassifier()

        model.fit(x_train, y_train)

        return model
    
    def save_model():
        pass

    def load_model_from_file(path):
        pass
