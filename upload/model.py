from __future__ import annotations

import math
from collections import namedtuple
from itertools import product
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics, model_selection

if TYPE_CHECKING:
    import os

TeamID: TypeAlias = int

Match = namedtuple(
    "Match",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "H",
        "A",
        "HSC",
        "ASC",
        "HFGM",
        "AFGM",
        "HFGA",
        "AFGA",
        "HFG3M",
        "AFG3M",
        "HFG3A",
        "AFG3A",
        "HFTM",
        "AFTM",
        "HFTA",
        "AFTA",
        "HORB",
        "AORB",
        "HDRB",
        "ADRB",
        "HRB",
        "ARB",
        "HAST",
        "AAST",
        "HSTL",
        "ASTL",
        "HBLK",
        "ABLK",
        "HTOV",
        "ATOV",
        "HPF",
        "APF",
    ],
)

Opp = namedtuple(
    "Opp",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "BetH",
        "BetA",
    ],
)

Summary = namedtuple(
    "Summary",
    [
        "Bankroll",
        "Date",
        "Min_bet",
        "Max_bet",
    ],
)


class RankingModel(Protocol):
    """Ranking model interface."""

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        raise NotImplementedError

    def rankings(self) -> dict[TeamID, float]:
        """Return normalized rankings."""
        raise NotImplementedError


class Player:
    """Handles betting strateggy."""

    def get_expected_profit(
        self, probability: float, ratio: float, proportion: float
    ) -> float:
        """Calculate the expected profit for given parametrs."""
        return (probability * ratio - 1) * proportion

    def get_variance_of_profit(
        self, probability: float, ratio: float, proportion: float
    ) -> float:
        """Calculate the variance of profit for given parameters."""
        return (1 - probability) * probability * (proportion**2) * (ratio**2)

    def sharpe_ratio(self, total_profit: float, total_var: float) -> float:
        """Return total sharpe ratio."""
        return total_profit / math.sqrt(total_var) if total_var > 0 else float("inf")

    def max_function(
        self, proportions: np.ndarray, probabilities: np.ndarray, ratios: np.ndarray
    ) -> float:
        """We are trying to minimize this function for sharpe ratio."""
        total_profit = 0
        total_var = 0

        for prob_row, ratios_row, proportion in zip(probabilities, ratios, proportions):
            print(prob_row, ratios_row, proportion)
            for probability, ratio in zip(prob_row, ratios_row):
                if len(prob_row) != 2 or len(ratios) != len(probabilities):
                    print(ratios, probabilities, proportions)
                    print("min_funciton, wrong format of probs")

                total_profit += self.get_expected_profit(probability, ratio, proportion)
                total_var += self.get_variance_of_profit(probability, ratio, proportion)

        return self.sharpe_ratio(total_profit, total_var)

    def get_bet_proportions(
        self,
        probabilities: np.ndarray,
        active_matches: pd.DataFrame,
        summary: Summary,
        steps: int,
    ) -> np.ndarray:
        """
        Return proportions of the budget to bet on given probs.

        Args:
            probabilities: 2d numpy array of probabilities.
            active_matches: DataFrame with active matches.
            summary: Summary of the game state.
            steps: number of steps to discretize the budget.

        Returns:
            2d numpy array of proportions with shape (num_bets, 2).

        """
        num_bets = probabilities.shape[0] * probabilities.shape[1]

        possible_ranges: list[np.ndarray] = [
            np.linspace(summary.Min_bet, summary.Max_bet, steps) / summary.Bankroll
            for _ in range(num_bets)
        ]

        ratios = np.array(active_matches[["OddsH", "OddsA"]])

        best_sharpe = -np.inf
        best_allocation = None
        for props in product(*possible_ranges):
            print(f"{props = }")
            sharpe = self.max_function(np.array(props), probabilities, ratios)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_allocation = props

        return np.array(best_allocation).reshape(probabilities.shape)

    def get_betting_strategy(
        self,
        probabilities: np.ndarray,
        active_matches: pd.DataFrame,
        summary: Summary,
    ) -> list:
        """Return absolute cash numbers and on what to bet in 2d list."""
        return (
            self.get_bet_proportions(probabilities, active_matches, summary, 10)
            * summary.Bankroll
        )


class TeamElo:
    """
    Class for keeping track of a team's Elo.

    Attributes:
        opponents: Sum of Elo ratings of opponents
        games: Number of games
        wins: Number of wins
        rating: Current Elo rating

    """

    K: int = 32
    opponents: int
    games: int
    wins: int
    rating: int

    def __init__(self) -> None:
        """Initialize TeamElo."""
        self.opponents = 0
        self.games = 0
        self.wins = 0
        self.rating = 1200

    def adjust(self, opponent: int, win: int) -> None:
        """
        Adjust Elo rating based on one match.

        Args:
            opponent: Elo rating of the other team
            win: 1 for win, 0 for loss

        """
        self.opponents += opponent
        self.games += 1
        self.wins += 1 if win else 0

        expected = 1 / (1 + 10 ** ((opponent - self.rating) / 400))

        self.rating += int(self.K * (win - expected))

    def __str__(self) -> str:
        """Create a string representation of the team's Elo."""
        return (
            f"{self.rating:>4} ({self.opponents:>7}, {self.games:>4}, "
            f"{self.wins:>4}, {self.wins / self.games * 100:>6.2f}%)"
        )


class Elo(RankingModel):
    """Class for the Elo ranking model."""

    teams: dict[int, TeamElo]

    def __init__(self) -> None:
        """Initialize Elo model."""
        self.teams = {}

    def __str__(self) -> str:
        """Create a string representation of the model."""
        return "Team  Elo Opponents Games  Wins  WinRate\n" + "\n".join(
            f" {team:>2}: {elo}"
            for team, elo in sorted(
                self.teams.items(), key=lambda item: -item[1].rating
            )
        )

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        home_team = self.teams.setdefault(match.HID, TeamElo())
        away_team = self.teams.setdefault(match.AID, TeamElo())

        home_elo = home_team.rating
        away_elo = away_team.rating

        home_team.adjust(away_elo, match.H)
        away_team.adjust(home_elo, match.A)

    def rankings(self) -> dict[int, float]:
        """Return normalized rankings."""
        max_elo = max(elo.rating for elo in self.teams.values())
        return {team: teamElo.rating / max_elo for team, teamElo in self.teams.items()}


class Model:
    """Main class."""

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()
        self.player = Player()
        self.ai = Ai()
        self.trained = False

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples()):
            self.elo.add_match(match)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opps: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        games_increment = inc[0]

        if not self.trained:
            self.train_ai(games_increment)
            self.trained = True
        else:
            self.update_models(games_increment)

        summary = Summary(*summ.iloc[0])

        upcoming_games: pd.DataFrame = opps[opps["Date"] == summary.Date]

        data_matrix = self.create_data_matrix(upcoming_games)

        probabilities = self.ai.get_probabilities(data_matrix)
        bets = self.player.get_betting_strategy(probabilities, opps, summary)

        new_bets = pd.DataFrame(
            data=bets,
            columns=pd.Index(["BetH", "BetA"], dtype="str"),
            index=upcoming_games.index,
        )
        return new_bets.reindex(opps.index, fill_value=0)

    def create_data_matrix(self, upcoming_games: pd.DataFrame) -> np.ndarray:
        """Get matches to predict outcome for."""
        data_matrix = np.ndarray([upcoming_games.shape[0], 4])

        upcoming_games = upcoming_games.reset_index(drop=True)

        for match in upcoming_games.itertuples(index=True):
            print(match)
            print(Opp(*match))

        for match in (Opp(*row) for row in upcoming_games.itertuples(index=True)):
            home_elo = self.elo.teams[match.HID].rating
            away_elo = self.elo.teams[match.AID].rating

            data_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
            ]

        return data_matrix

    def train_ai(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        train_matrix = np.ndarray([dataframe.shape[0], 5])

        for match in (Match(*x) for x in dataframe.itertuples()):
            self.elo.add_match(match)

            home_elo = self.elo.teams[match.HID].rating
            away_elo = self.elo.teams[match.AID].rating

            train_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
                match.H,
            ]

        self.ai.train(train_matrix)


# Merging predict.py
# handles predicting results


class Ai:
    """Class for training and predicting."""

    model: xgb.XGBClassifier

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
        self.model = xgb.XGBClassifier()

    def train(self, train_matrix: np.ndarray) -> None:
        """Return trained model."""
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            train_matrix[:, :-1],
            train_matrix[:, -1],
            test_size=0.1,
            random_state=6,
        )
        self.model.fit(x_train, y_train)
        probabilities = self.model.predict_proba(x_val)
        predictions = self.model.predict(x_val)
        prob = [probabilities[i][pred] for i, pred in enumerate(predictions)]
        print("Accuracy:", metrics.accuracy_score(y_val, predictions))
        print("Average confidence:", sum(prob) / len(prob))

    def get_probabilities(self, data_matrix: np.ndarray) -> np.ndarray:
        """Get probabilities for match outcome [home_loss, home_win]."""
        return self.model.predict_proba(data_matrix)

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)
