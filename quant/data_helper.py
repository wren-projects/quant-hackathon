import numpy as np
import pandas as pd


class TeamData:
    """Hold data of one team."""

    def __init__(self, id: int):
        """Init datastucture."""
        self.id = id
        self.date_last_mach: pd.Timestamp = pd.to_datetime("1975-11-06")

    def get_days_scince_last_mach(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach and update last mach to today."""
        delta = today - self.date_last_mach
        self.date_last_mach = today
        return delta.days


data = TeamData(1)
time = pd.Timestamp("1975-11-20")
print(data.get_days_scince_last_mach(time))


teams_last_ten: dict = {}


class CustomQueue:
    """Serve as custom version of queue."""

    def __init__(self, n: int, feature_count: int) -> None:
        """Initialize queue."""
        self.size: int = n
        self.feature_count: int = feature_count
        self.values: np.matrix = np.zeros((n, feature_count))
        self.__curent_oldest: int = 0

    def put(self, value: np.array) -> None:
        """Put new value in queue."""
        self.values[self.__curent_oldest % self.size, :] = value
        self.__curent_oldest += 1

    def get_q_avr(self) -> np.array:
        """Return avrage array of each feature."""
        if self.__curent_oldest == 0:
            return np.zeros((1, self.feature_count))
        return self.values.mean(0)


class TeamsLastN:
    """Contain all features needed to calculate avreages of last n matches."""

    def __init__(self, n: int) -> None:
        """Initialize."""
        self.teams_last_n_values: dict[int, CustomQueue] = {}
        self.n: int = n

    def add_value(self, team_id: int, value: np.array) -> None:
        """Add featuer values to dataset for given team."""
        if team_id not in self.teams_last_n_values:
            self.teams_last_n_values[team_id] = CustomQueue(self.n, len(value))
        self.teams_last_n_values[team_id].put(value)

    def get_avr(self, team_id: int) -> np.array:
        """"""
        return self.teams_last_n_values[team_id].get_q_avr()


"""
team = TeamsLastN(10, 2)
for i in range(100):
    team.add_value(0, [i, 3 * i])
    print(team.get_avr(0))
"""
"""

# for i in range(100):
q = Custom_Queue(10)
for i in range(100):
    q.put(i)
    print(q.get_avr())
"""
