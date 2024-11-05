import numpy as np
import pandas as pd


class CustomQueue:
    """Serve as custom version of queue."""

    def __init__(self, n: int) -> None:
        """Initialize queue."""
        self.size: int = n
        self.values: np.array = np.zeros((n, 1))
        self.__curent_oldest: int = 0

    def put(self, value: float) -> None:
        """Put new value in queue."""
        self.values[self.__curent_oldest % self.size] = value
        self.__curent_oldest += 1

    def get_q_avr(self) -> np.array:
        """Return avrage array of each feature."""
        if self.__curent_oldest == 0:
            return 0.0
        return np.sum(self.values) / min(self.size, self.__curent_oldest)


class TeamData:
    """Hold data of one team."""

    def __init__(self, id: int):
        """Init datastucture."""
        self.id = id
        self.date_last_mach: pd.Timestamp = pd.to_datetime("1975-11-06")
        self.home_points_last_n: CustomQueue = CustomQueue(10)
        self.away_points_last_n: CustomQueue = CustomQueue(10)

    def get_days_scince_last_mach(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach."""
        delta = today - self.date_last_mach
        return delta.days

    def update_data(self, data_line: pd.DataFrame) -> None:
        """Update team data based on dato from one mach (one line of data)."""
        self.date_last_mach = data_line["Date"]
        # TODO update last n based on home_away

    def get_data_vector(self, home_away: int) -> np.array:
        """Return complete data vector for given team."""
        if home_away == 0:
            output_points = self.home_points_last_n.get_q_avr()
        else:
            output_points = self.away_points_last_n.get_q_avr()
        return np.array([self.get_days_scince_last_mach, output_points])


"""
data = TeamData(1)
time = pd.Timestamp("1975-11-20")
print(data.get_days_scince_last_mach(time))
"""
