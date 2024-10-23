# extracts information from dataset
import numpy as np


class Data:
    # all arrays and matrixes needs to be in the same order and of the same size
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        """formats data to desired format"""
        data = ""
        return data

    def get_train_matrix(self) -> np.ndarray:
        """gets train matrix, match outcome is in the last column (1 - home win, 0 - home loose)"""
        """train_matrix format
            [[home_parametr, away_parametr,..., match_parametr, match_outcome],
             [home_parametr, away_parametr,..., match_parametr, match_outcome],
             ...
             [home_parametr, away_parametr,..., match_parametr, match_outcome]]
        """
        features = [self.get_home_avg_points, self.get_away_avg_points, self.get_match_outcomes]
        array_list = [func() for func in features]
        # funcions returns colums of the final train_matrix format, needs to be transposed
        train_matrix = np.stack(array_list).T
        return train_matrix

    def get_match_array(self, team_home, team_away) -> np.ndarray:
        """returns array for specific match, used for predicting"""
        pass
    
    
    

    def add_new_match_outcome(new_match):
        """adds new information to existing dataset"""
        pass

    def get_home_avg_points() -> np.ndarray:
        pass

    def get_away_avg_points() -> np.ndarray:
        pass

    def get_match_outcomes() -> np.ndarray:
        # 1 - home_win, 0 - home_loss
        pass
