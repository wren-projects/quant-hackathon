# handles predicting results
from analysis import Data

import numpy as np
import xgboost as xgb
from sklearn import model_selection


def train_model(data: Data):
    """returns trained model"""
    train_matrix = data.get_train_matrix()
    x_train, x_val, y_train, y_val = model_selection.train_test_split(
        train_matrix[:, :-1], train_matrix[:, -1], test_size=0.1, random_state=2
    )

    model = xgb.XGBClassifier()

    model.fit(x_train, y_train)

    return model

