# extracts information from dataset
from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd

from quant.data_helper import Team, TeamData
from quant.types import Match


class GamePlace(IntEnum):
    """Enum for game place."""

    Home = 0
    Away = 1
    Neutral = 2


class Data:
    """Class for working with data."""

    def __init__(self, data: pd.DataFrame) -> None:
        """Create Data from csv file."""
        self.data = data
        self.teams_data: dict[int, TeamData] = {}

    def _get_match_array(self, match: Match) -> np.ndarray:
        """
        Return array for specific match and update team data.

        Based on matches that happend so far.
        Used for making training matrix.
        """
        h_id: int = match.HID
        a_id: int = match.AID
        date: pd.Timestamp = pd.to_datetime(match.Date)

        home_team = self.teams_data.setdefault(h_id, TeamData(h_id))
        away_team = self.teams_data.setdefault(a_id, TeamData(a_id))

        output: np.ndarray = np.concatenate(
            (
                home_team.get_data_vector(Team.Home, date),
                away_team.get_data_vector(Team.Away, date),
            )
        )

        home_team.update(match, Team.Home)
        away_team.update(match, Team.Away)

        return output

    def get_train_matrix(self) -> np.ndarray:
        """Create train matrix from the current data."""
        train_matrix: np.ndarray = np.empty((len(self.data), 2 * TeamData.COLUMNS))
        for match in (Match(*row) for row in self.data.itertuples(index=True)):
            train_matrix[match.Index] = self._get_match_array(match)
            print(f"\r{match.Index}", end="")

        return train_matrix


if __name__ == "__main__":
    dataframe = pd.read_csv("quant/datasets/games.csv")

    # model = Elo()

    data = Data(dataframe)
    data.get_train_matrix()
