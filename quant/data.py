# extracts information from dataset
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import os

from data_helper import Team, TeamData


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

    # data: pd.DataFrame
    # train_matrix: np.ndarray

    def __init__(self, data_path: os.PathLike) -> None:
        """Create Data from csv file."""
        self.data: pd.DataFrame = pd.read_csv(data_path)
        self.train_matrix: np.ndarray = np.empty(self.data.shape)
        self.teams_data: dict[int, TeamData] = {}

    def add_new_matches_outcome(self, new_matches) -> None:
        """Add new information to existing dataset."""

    def _get_match_array(self, match_data: pd.Series) -> np.array:
        """
        Return array for specific match and update team data.

        Based on matches that happend so far.
        Used for making training matrix.
        """
        h_id: int = match_data["HID"]
        a_id: int = match_data["AID"]
        date: pd.Timestamp = match_data["Date"]

        if h_id not in self.teams_data:
            self.teams_data[h_id] = TeamData(h_id)
        if a_id not in self.teams_data:
            self.teams_data[h_id] = TeamData(a_id)

        self.teams_data[h_id].get_data_vector(Team.Home)

        output: np.array = np.concatenate(
            (
                self.teams_data[h_id].get_data_vector(Team.Home, date),
                self.teams_data[a_id].get_data_vector(Team.Away, date),
                # TODO result??
            )
        )

        self.teams_data[h_id].update_data(match_data)
        self.teams_data[a_id].update_data(match_data)

        return output

    def get_train_matrix(self) -> np.ndarray:
        """Create train matrix from the current data."""
        """train_matrix format
                [[home_parametr, away_parametr,..., match_parametr, match_outcome],
                [home_parametr, away_parametr,..., match_parametr, match_outcome],
                ...
                [home_parametr, away_parametr,..., match_parametr, match_outcome]]
        """
        for index, line in self.data.iterrows():
            self.train_matrix[index] = self._get_match_array(line)

        return self.train_matrix

        """
        columns: list[np.ndarray] = [
            average_points(Team.Home, self.data, relevant_matches),
            average_points(Team.Away, self.data, relevant_matches),
            avg_points_at(Team.Home, GamePlace.Home, self.data, relevant_matches),
            avg_points_at(Team.Away, GamePlace.Away, self.data, relevant_matches),
            match_outcomes(self.data),
        ]
        return np.column_stack(columns)
        """


"""
# sampleData = Data("quant/dataset/games.csv")
# sampleData.get_train_matrix()


data = pd.read_csv("quant/datasets/games.csv")
average_points(0, data, 10)
"""
