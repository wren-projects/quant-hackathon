# extracts information from dataset
import os

import numpy as np
import pandas as pd


class Data:
    """Class for working with data."""

    # all arrays and matrixes needs to be in the same order and of the same siz

    def load_csv_into_pd(self, data_path: os.PathLike) -> pd.DataFrame:
        """Format data to desired format."""
        return pd.read_csv(data_path)

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

    def get_match_array(self, match: pd.Series) -> np.ndarray:
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
