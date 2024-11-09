from enum import IntEnum

import numpy as np
import pandas as pd

from quant.types import Match


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

    def get_q_avr(self) -> float:
        """Return average array of each feature."""
        if self.__curent_oldest == 0:
            return 0.0

        return (
            np.sum(self.values) / min(self.size, self.__curent_oldest)
            if self.__curent_oldest
            else 0.0
        )


class TeamData:
    """Hold data of one team, both as home and away."""

    N_SHORT = 8
    N_LONG = 20

    COLUMNS = 2

    def __init__(self, team_id: int) -> None:
        """Init datastucture."""
        self.id: int = team_id
        self.date_last_mach: pd.Timestamp = pd.to_datetime("1975-11-06")
        self.home_points_last_n: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.away_points_last_n: CustomQueue = CustomQueue(TeamData.N_SHORT)

    def _get_days_since_last_mach(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach."""
        return (today - self.date_last_mach).days

    def update(self, match: Match, played_as: Team) -> None:
        """Update team data based on data from one mach."""
        self.date_last_mach = pd.to_datetime(match.Date)

        if played_as == Team.Home:
            self.home_points_last_n.put(match.HSC)
        else:
            self.away_points_last_n.put(match.ASC)

    def get_data_vector(self, played_as: Team, date: pd.Timestamp) -> np.ndarray:
        """
        Return complete data vector for given team.

        Return vector:[
        days scine last mach,
        avr points in lasnt n matches as H/A
        ]
        """
        points = (
            self.home_points_last_n
            if played_as == Team.Home
            else self.away_points_last_n
        )

        output_points = points.get_q_avr()

        last_date = self._get_days_since_last_mach(date)

        return np.array([last_date, output_points])
