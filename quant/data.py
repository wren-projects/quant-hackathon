# extracts information from dataset
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import os


class Team(IntEnum):
    """Enum discerning teams playing home or away."""

    Home = 0
    Away = 1


class GamePlace(IntEnum):
    """Enum for game place."""

    Home = 0
    Away = 1
    Neutral = 2


def average_points(team: Team, dataframe: pd.DataFrame, n: int) -> np.ndarray:
    """
    Calculate the average points.

    For each match, calculate the average of the team's points in their n last
    games.
    """


def avg_points_at(
    team: Team, place: GamePlace, dataframe: pd.DataFrame, n: int
) -> np.ndarray:
    """
    Calculate the average points of the given team at given place.

    For each match, calculate the average of the team's points in their n last
    matches in given place.
    """


def match_outcomes(dataframe: pd.DataFrame) -> np.ndarray:
    """
    Aggregate the match outcomes.

    For each match, output 1 if home team won, and 0 if home team lost.
    """


class Data:
    """Class for working with data."""

    # all arrays and matrixes needs to be in the same order and of the same size

    data: pd.DataFrame
    train_matrix: np.ndarray

    def __init__(self, data_path: os.PathLike) -> None:
        """Create Data from csv file."""
        self.data = pd.read_csv(data_path)

    def add_new_matches_outcome(self, new_matches) -> None:
        """Add new information to existing dataset."""

    def get_match_array(self, match: pd.Series) -> np.ndarray:
        """Return array for specific match, used for predicting."""

    def get_train_matrix(self, relevant_matches: int) -> np.ndarray:
        """Create train matrix from the current data."""
        """train_matrix format
                [[home_parametr, away_parametr,..., match_parametr, match_outcome],
                [home_parametr, away_parametr,..., match_parametr, match_outcome],
                ...
                [home_parametr, away_parametr,..., match_parametr, match_outcome]]
        """
        columns: list[np.ndarray] = [
            average_points(Team.Home, self.data, relevant_matches),
            average_points(Team.Away, self.data, relevant_matches),
            avg_points_at(Team.Home, GamePlace.Home, self.data, relevant_matches),
            avg_points_at(Team.Away, GamePlace.Away, self.data, relevant_matches),
            match_outcomes(self.data),
        ]
        return np.column_stack(columns)
