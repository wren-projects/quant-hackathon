from __future__ import annotations

import math
from collections import namedtuple
from enum import IntEnum
from itertools import product
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
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

    def min_function(
        self, proportions: np.ndarray, probabilities: np.ndarray, ratios: np.ndarray
    ) -> float:
        """We are trying to minimize this function for sharpe ratio."""
        total_profit = 0
        total_var = 0
        for i in range(len(probabilities)):
            for j in range(len(probabilities[i])):
                probability = probabilities[i][
                    j
                ]  # First column is for win, second column is for loss
                ratio = ratios[i][j]  # Use the ratio corresponding to the win scenario
                # Access flattened array index
                prop_of_budget = proportions[i * len(probabilities[i]) + j]
                total_profit += self.get_expected_profit(
                    probability, ratio, prop_of_budget
                )
                total_var += self.get_variance_of_profit(
                    probability, ratio, prop_of_budget
                )
        """
        for prob_row, ratios_row, proportion in zip(probabilities, ratios, proportions):
            print(prob_row, ratios_row, proportion)
            for probability, ratio in zip(prob_row, ratios_row):
            """

        return -self.sharpe_ratio(total_profit, total_var)

    def get_bet_proportions(
        self,
        probabilities: np.ndarray,
        active_matches: pd.DataFrame,
        summary: Summary,
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
            # print(f"{props = }")
            sharpe = self.max_function(np.array(props), probabilities, ratios)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_allocation = props
        """
        ratios = np.array(active_matches[["OddsH", "OddsA"]])
        print(ratios)
        initial_props = np.full_like(probabilities, 0.01, dtype=float)

        # Constraint: sum of all props <= 1 (global budget constraint for entire 2D array)
        cons = [
            {"type": "ineq", "fun": lambda props: 1.0 - sum(props)}
        ]  # Global budget constraint

        # Bounds: Each proportion must be between 0 and 1
        bounds = [
            (0, (summary.Max_bet / summary.Bankroll))
            for _ in range(probabilities.shape[0] * probabilities.shape[1])
        ]

        # Flatten the props for optimization and define the bounds
        initial_props_flat = initial_props.flatten()
        # Objective function minimization
        result = minimize(
            self.min_function,
            initial_props_flat,
            args=(probabilities, ratios),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
        )
        print(np.array(result.x).reshape(probabilities.shape))
        return np.array(result.x).reshape(probabilities.shape)

    def get_betting_strategy(
        self,
        probabilities: np.ndarray,
        active_matches: pd.DataFrame,
        summary: Summary,
    ) -> list:
        """Return absolute cash numbers and on what to bet in 2d list."""
        proportions: list[float] = (
            self.get_bet_proportions(probabilities, active_matches, summary)
            * summary.Bankroll
        )
        return np.array(proportions).round(decimals=0)


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
        self.data = Data()

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples()):
            self.elo.add_match(match)
            self.data.add_match(match)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opps: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        print("new round")
        games_increment = inc[0]

        if not self.trained:
            self.train_ai(games_increment)
            self.trained = True
        else:
            self.update_models(games_increment)

        summary = Summary(*summ.iloc[0])

        upcoming_games: pd.DataFrame = opps[opps["Date"] == summary.Date]
        if upcoming_games.shape[0] != 0:

            data_matrix = self.create_data_matrix(upcoming_games)

            probabilities = self.ai.get_probabilities(data_matrix)
            # probabilities = probabilities * 0.5 + 0.25
            print(probabilities)
            bets = self.player.get_betting_strategy(
                probabilities, upcoming_games, summary
            )

        else:
            bets = []
        new_bets = pd.DataFrame(
            data=bets,
            columns=pd.Index(["BetH", "BetA"], dtype="str"),
            index=upcoming_games.index,
        )

        print(bets)
        new_bets = new_bets.reindex(opps.index, fill_value=0)
        # print(new_bets)
        return new_bets

    def create_data_matrix(self, upcoming_games: pd.DataFrame) -> np.ndarray:
        """Get matches to predict outcome for."""
        data_matrix = np.ndarray([upcoming_games.shape[0], 4])

        upcoming_games = upcoming_games.reset_index(drop=True)

        """"
        for match in upcoming_games.itertuples(index=True):
            print(match)
            print(Opp(*match))
        """

        for match in (Opp(*row) for row in upcoming_games.itertuples(index=True)):
            home_elo = self.elo.teams[match.HID].rating
            away_elo = self.elo.teams[match.AID].rating

            data_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
                *self.data.get_match_array(match),
            ]

        return data_matrix

    def train_ai(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        train_matrix = np.ndarray([dataframe.shape[0], 5])

        for match in (Match(*x) for x in dataframe.itertuples()):
            self.elo.add_match(match)

            home_id: TeamID = match.HID
            away_id: TeamID = match.AID

            home_elo = self.elo.teams[home_id].rating
            away_elo = self.elo.teams[away_id].rating

            train_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
                match.H,
                *self.data.get_match_array(match),
            ]

            self.data.add_match(match)

        self.ai.train(train_matrix)


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
        return np.array(self.model.predict_proba(data_matrix))[:, ::-1]

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)


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

    def __init__(self, team_id: TeamID) -> None:
        """Init datastucture."""
        self.id: TeamID = team_id
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


class Data:
    """Class for working with data."""

    def __init__(self) -> None:
        """Create Data from csv file."""
        self.teams: dict[TeamID, TeamData] = {}

    def add_match(self, match: Match) -> None:
        """Update team data based on data from one mach."""
        self.teams.setdefault(match.HID, TeamData(match.HID)).update(match, Team.Home)
        self.teams.setdefault(match.AID, TeamData(match.AID)).update(match, Team.Away)

    def team_data(self, team_id: TeamID) -> TeamData:
        """Return the TeamData for given team."""
        return self.teams[team_id]

    def get_match_array(self, match: Match) -> np.ndarray:
        """Get array for match."""
        home_team = self.teams.setdefault(match.HID, TeamData(match.HID))
        away_team = self.teams.setdefault(match.AID, TeamData(match.AID))

        date: pd.Timestamp = pd.to_datetime(match.Date)

        return np.array(
            [
                *home_team.get_data_vector(Team.Home, date),
                *away_team.get_data_vector(Team.Away, date),
            ]
        )
