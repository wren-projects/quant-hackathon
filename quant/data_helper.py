from enum import IntEnum

import numpy as np
import pandas as pd


class Team(IntEnum):
    """Enum discerning teams playing home or away."""

    Home = 0
    Away = 1


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
    """Hold data of one team, both as home and away."""

    def __init__(self, team_id: int):
        """Init datastucture."""
        self.id: int = team_id
        self.date_last_mach: pd.Timestamp = pd.to_datetime("1975-11-06")
        self.home_points_last_n: CustomQueue = CustomQueue(10)
        self.away_points_last_n: CustomQueue = CustomQueue(10)

    def _get_days_scince_last_mach(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach."""
        delta = today - self.date_last_mach
        return delta.days

    def update_data(self, data_line: pd.Series, home_away: Team) -> None:
        """Update team data based on data from one mach."""
        self.date_last_mach = pd.to_datetime(data_line["Date"])
        if home_away == Team.Home:
            self.home_points_last_n.put(data_line["HSC"])
        else:
            self.away_points_last_n.put(data_line["ASC"])

    def get_data_vector(self, home_away: Team, date: pd.Timestamp) -> np.array:
        """
        Return complete data vector for given team.

        Return vector:[
        days scine last mach,
        avr points in lasnt n matches as H/A
        ]
        """
        if home_away == Team.Home:
            output_points = self.home_points_last_n.get_q_avr()
        else:
            output_points = self.away_points_last_n.get_q_avr()
        return np.array([self._get_days_scince_last_mach(date), output_points])


"""
data = TeamData(1)
time = pd.Timestamp("1975-11-20")
print(data._get_days_scince_last_mach(time))
"""
