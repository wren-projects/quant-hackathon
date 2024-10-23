#extracts information from dataset

import numpy as np

class Data():
    def __init__(self, data_path):
        self.data = self.load_data(data_path)


    def load_data(self, data_path):
        """formats data to desired format"""
        data = ""
        return data
    
    def get_train_matrix(self) -> np.array:
        "gets train matrix, match outcome is in the last column"

        features = [self.get_home_avg_points(), self.get_away_avg_points(), self.get_match_outcomes]

        matrix = [func() for func in features]

        return np.stack(matrix).T
    
    






    
    def get_home_avg_points() -> np.array:
        pass

    def get_away_avg_points() -> np.array:
        pass

    def get_match_outcomes() -> np.array:
        pass

        
