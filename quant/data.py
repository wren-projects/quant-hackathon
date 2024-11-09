# extracts information from dataset
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quant.types import Match

if TYPE_CHECKING:
    import os

from quant.data_helper import Team, TeamData


class GamePlace(IntEnum):
    """Enum for game place."""

    Home = 0
    Away = 1
    Neutral = 2


'''
def average_points(team: Team, dataframe: pd.DataFrame, n: int) -> np.ndarray:
    """
    Calculate the average points.

    For each match, calculate the average of the team's points in their n last
    games.
    """
    teams_data = TeamsLastN(10)
    for i in range(len(dataframe)):
        home_id = int(dataframe.iloc[i]["HID"])

    print()


def avg_points_at(
    team: Team, place: GamePlace, dataframe: pd.DataFrame, n: int
) -> np.ndarray:
    """
    Calculate the average points of the given team at given place.

    For each match, calculate the average of the team's points in their n last
    matches in given place.
    """
'''


def match_outcomes(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Aggregate the match outcomes.

    For each match, output 1 if home team won, and 0 if home team lost.
    """


class Data:
    """Class for working with data."""

    # all arrays and matrixes needs to be in the same order and of the same size

    def __init__(self, data: pd.DataFrame) -> None:
        """Create Data from csv file."""
        self.data = data
        self.train_matrix: np.ndarray = np.empty((len(self.data), 2 * TeamData.COLUMNS))
        self.teams_data: dict[int, TeamData] = {}

    '''
    def add_new_matches_outcome(self, new_matches) -> None:
        """Add new information to existing dataset."""

    '''

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
                # TODO result??
            )
        )

        home_team.update(match, Team.Home)
        away_team.update(match, Team.Away)

        return output

    def get_train_matrix(self) -> np.ndarray:
        """Create train matrix from the current data."""
        """train_matrix format
                [[home_parametr, away_parametr,..., match_parametr, match_outcome],
                [home_parametr, away_parametr,..., match_parametr, match_outcome],
                ...
                [home_parametr, away_parametr,..., match_parametr, match_outcome]]
        """
        for match in (Match(*row) for row in self.data.itertuples(index=True)):
            self.train_matrix[match.Index] = self._get_match_array(match)
            print(f"\r{match.Index}", end="")

        return self.train_matrix


# sampleData = Data(pd.read_csv("quant/datasets/games.csv"))
# print(sampleData.get_train_matrix())


"""
data = pd.read_csv("quant/datasets/games.csv")
average_points(0, data, 10)
"""
