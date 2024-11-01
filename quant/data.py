# extracts information from dataset
import numpy as np
import pandas as pd


class Data:
    # all arrays and matrixes needs to be in the same order and of the same size
    def __init__(self):
        self.data = self.load_data("")

    def load_data(self, data_path) -> str:
        """Format data to desired format."""
        return ""

    def get_train_matrix(self) -> np.ndarray:
        """Get train matrix, match outcome is in the last column."""
        """train_matrix format
            [[home_parametr, away_parametr,..., match_parametr, match_outcome],
             [home_parametr, away_parametr,..., match_parametr, match_outcome],
             ...
             [home_parametr, away_parametr,..., match_parametr, match_outcome]]
        """
        array_list = [
            self.get_home_avg_points(),
            self.get_away_avg_points(),
            self.get_match_outcomes(),
        ]
        # funcion returns colums of the final train_matrix format, needs to be transposed
        train_matrix = np.stack(array_list).T
        return train_matrix

    def get_match_array(self, match: pd.DataFrame) -> np.ndarray:
        """Return array for specific match, used for predicting."""
        pass

    def add_new_matches_outcome(self, new_matches):
        """Add new information to existing dataset."""
        pass

    def get_home_avg_points() -> np.ndarray:
        pass

    def get_away_avg_points() -> np.ndarray:
        pass

    def get_match_outcomes() -> np.ndarray:
        # 1 - home_win, 0 - home_loss
        pass
