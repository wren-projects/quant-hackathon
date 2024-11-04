import numpy as np

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
    def __init__(self, n: int, feature_count: int) -> None:
        self.teams_last_n_values: dict[int, CustomQueue] = {}
        self.n: int = n
        self.feature_count = feature_count

    def add_value(self, team_id: int, value: np.array) -> None:
        if team_id not in self.teams_last_n_values:
            self.teams_last_n_values[team_id] = CustomQueue(self.n, self.feature_count)
        self.teams_last_n_values[team_id].put(value)

    def get_avr(self, team_id: int) -> np.array:
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
