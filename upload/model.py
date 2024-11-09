from __future__ import annotations

import math
from collections import namedtuple
from enum import IntEnum
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Protocol, TypeAlias

import matplotlib.pyplot as plt
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


def plot_bad_predictions(Y_true, probabilities):
    # Ensure input is a numpy array for consistent processing
    Y_true = np.array(Y_true)
    probabilities = np.array(probabilities)

    # Create a DataFrame for analysis
    df = pd.DataFrame({"Y_true": Y_true, "probabilities": probabilities})

    # Define bins for grouping probabilities by tens
    bins = np.arange(0, 1.05, 0.05)
    labels = [f"{b:.2f}-{b+0.05:.02f}" for b in bins[:-1]]

    # Categorize probabilities into bins
    df["probability_group"] = pd.cut(
        df["probabilities"], bins=bins, labels=labels, include_lowest=True
    )

    # Calculate total instances and bad predictions
    total_instances = df.groupby("probability_group").size()
    bad_predictions = df.groupby("probability_group").apply(
        lambda x: (
            (
                (x["Y_true"] == 1) & (x["probabilities"] < 0.5)
            ).sum()  # True class 1, predicted low
            + (
                (x["Y_true"] == 0) & (x["probabilities"] >= 0.5)
            ).sum()  # True class 0, predicted high
        )
    )

    # Calculate the probability of being badly classified
    bad_classification_probability = bad_predictions / total_instances

    # Prepare data for plotting
    bad_classification_probability = bad_classification_probability.reset_index(
        name="bad_classification_prob"
    )
    bad_classification_probability["total_count"] = (
        total_instances.values
    )  # Add total counts

    # Plotting
    # Plotting
    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        bad_classification_probability["probability_group"],
        bad_classification_probability["bad_classification_prob"],
        color="orange",
    )

    plt.xlabel("Probability Groups (5% Intervals)")
    plt.ylabel("Probability of Being Badly Classified")
    plt.title("Probability of Bad Classification by Probability Group (Grouped by 5%)")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    # Adding counts on top of bars
    for bar, count in zip(bars, bad_classification_probability["total_count"]):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    plt.tight_layout()
    plt.show()
    # Creating a DataFrame for correlation calculation
    data = pd.DataFrame({"Y_true": Y_true, "probabilities": probabilities})
    # Calculating the correlation
    correlation = data.corr().iloc[0, 1]
    print(f"corelation: {correlation}")


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

    K: int = 400
    A: int = 200
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

        expected = 1 / (1 + 10 ** ((opponent - self.rating) / self.A))

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

    def team_rating(self, team_id: int) -> float:
        """Return Elo rating of a team."""
        return self.teams.setdefault(team_id, TeamElo()).rating


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
            self.train_ai_reg(games_increment)
            self.trained = True
        else:
            self.update_models(games_increment)

        summary = Summary(*summ.iloc[0])

        upcoming_games: pd.DataFrame = opps[opps["Date"] == summary.Date]

        if upcoming_games.shape[0] != 0:
            data_matrix = self.create_data_matrix(upcoming_games)

            probabilities = self.ai.get_probabilities_reg(data_matrix)
            # probabilities = probabilities * 0.5 + 0.25
            print(probabilities)

            bets = self.player.get_betting_strategy(
                probabilities, upcoming_games, summary
            )

            """bets = self.put_max_bet(probabilities, upcoming_games, summary)"""

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

    def put_max_bet(
        self, probabilities: np.ndarray, upcoming_games: Match, summary: Summary
    ) -> np.ndarray:
        ratio_cut_off = 1.2
        budget = summary.Bankroll / 2
        binary_bets = (probabilities - 0.3).round(decimals=0)
        ratios = deepcopy(np.array(upcoming_games[["OddsH", "OddsA"]]))
        print(ratios)
        for i in range(len(ratios)):
            for j in range(2):
                if ratios[i][j] > ratio_cut_off:
                    ratios[i][j] = 1
                else:
                    ratios[i][j] = 0
        print(ratios)
        binary_bets = binary_bets * ratios
        num_of_bets = np.count_nonzero(binary_bets)
        bet = min(budget / num_of_bets, summary.Max_bet)
        bets = binary_bets * bet
        return bets

    def create_data_matrix(self, upcoming_games: pd.DataFrame) -> np.ndarray:
        """Get matches to predict outcome for."""
        data_matrix = np.ndarray([upcoming_games.shape[0], 2])

        upcoming_games = upcoming_games.reset_index(drop=True)

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
            home_elo = self.elo.team_rating(match.HID)
            away_elo = self.elo.team_rating(match.AID)

            train_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
                match.H,
                *self.data.get_match_array(match),
            ]

            self.data.add_match(match)
            self.elo.add_match(match)

        self.ai.train(train_matrix)

    def train_ai_reg(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        train_matrix = np.ndarray([dataframe.shape[0], 3])

        for match in (Match(*x) for x in dataframe.itertuples()):
            home_elo = self.elo.team_rating(match.HID)
            away_elo = self.elo.team_rating(match.AID)

            train_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
                match.HSC - match.ASC,
            ]

            self.elo.add_match(match)

        self.ai.train_reg(train_matrix)


def calculate_elo_accuracy(data: list[list[int]]) -> float:
    """Calculate the accuracy of ELO predictions."""
    correct_predictions = 0
    total_games = len(data)
    games = np.array(data)[:, :-1]
    outcomes = np.array(data)[:, -1].clip(0, 1).round(decimals=0)
    for i in range(len(data)):
        elo_home = games[i][0]
        elo_away = games[i][1]
        outcome = outcomes[i]

        # Predict home win if home ELO is greater than away ELO
        predicted_outcome = 1 if elo_home > elo_away else 0

        # Compare predicted outcome with actual outcome
        if predicted_outcome == outcome:
            correct_predictions += 1

    # Calculate accuracy as a percentage
    return correct_predictions / total_games


class Ai:
    """Class for training and predicting."""

    model: xgb.XGBRegressor

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
        self.model = xgb.XGBClassifier()

    def train(self, train_matrix: np.ndarray) -> None:
        """Return trained model."""
        len_t = len(train_matrix)
        print(f"len of training data {len_t}")
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            train_matrix[:-600, :-1],
            train_matrix[:-600, -1],
            test_size=0.1,
            random_state=6,
        )
        self.model.fit(x_train, y_train)
        probabilities = self.model.predict_proba(train_matrix[-600:, :-1])
        predictions = self.model.predict(train_matrix[-600:, :-1])
        prob = [probabilities[i][pred] for i, pred in enumerate(predictions)]
        print("Accuracy:", metrics.accuracy_score(train_matrix[-600:, -1], predictions))

        fpr, fnr, _ = metrics.det_curve(train_matrix[-600:, -1], probabilities[:, -1])
        print("Average confidence:", sum(prob) / len(prob))
        plot_bad_predictions(train_matrix[-600:, -1], probabilities[:, -1])

        cm = metrics.confusion_matrix(
            train_matrix[-600:, -1], predictions, labels=self.model.classes_
        )
        disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=self.model.classes_
        )
        disp.plot()
        plt.show()

    def train_reg(self, train_matrix: np.ndarray) -> None:
        """Return trained model."""
        len_t = len(train_matrix)
        print(f"len of training data {len_t}")
        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            train_matrix[:-600, :-1],
            train_matrix[:-600, -1],
            test_size=0.01,
            random_state=6,
        )
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.005,
            n_estimators=500,
            max_depth=5,
        )

        """
        param_grid = {
        'learning_rate': [0.005],
        'n_estimators': [ 500, 800, 1000],
        'max_depth': [3, 4 , 5, 6, 7],
        }

        grid_search = model_selection.GridSearchCV(estimator=self.model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

        grid_search.fit(x_train, y_train)

        print("Best Parameters:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)
        """

        self.model.fit(x_train, y_train)
        predictions = self.model.predict(train_matrix[-600:, :-1])
        print(
            "RMSE:",
            math.sqrt(metrics.mean_squared_error(train_matrix[-600:, -1], predictions)),
        )
        print("MAE:", metrics.mean_absolute_error(train_matrix[-600:, -1], predictions))
        probabilities = self.calculate_probabilities(predictions)
        plot_bad_predictions(
            train_matrix[-600:, -1].clip(0, 1).round(decimals=0), probabilities[:, -1]
        )
        cm = metrics.confusion_matrix(
            train_matrix[-600:, -1].clip(0, 1).round(decimals=0),
            probabilities[:, -1].round(decimals=0),
            labels=[0, 1],
        )
        disp = metrics.ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=[0, 1]
        )
        disp.plot()

        print(
            "just home: ",
            train_matrix[-600:, -1].clip(0, 1).round(decimals=0).sum() / 600,
        )
        print("ELO accuracy:", calculate_elo_accuracy(train_matrix[-600:, :]))

        plt.show()

    def get_probabilities(self, data_matrix: np.ndarray) -> np.ndarray:
        """Get probabilities for match outcome [home_loss, home_win]."""
        return np.array(self.model.predict_proba(data_matrix))[:, ::-1]

    def get_probabilities_reg(self, data_matrix: np.ndarray) -> np.ndarray:
        """Get probabilities for match outcome [home_loss, home_win]."""
        predicted_dif_score = np.array(self.model.predict(data_matrix))
        return self.calculate_probabilities(predicted_dif_score)[:, ::-1]

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)

    def home_team_win_probability(self, point_difference: float) -> float:
        slope = 0.8  # range optimal 0.1 to 1. liked 0.3 and 0.5 (maybe 1)
        return 1 / (1 + np.exp(-slope * point_difference))

    def calculate_probabilities(self, score_differences: np.ndarray) -> np.ndarray:
        """Calculate the probabilities of home and away teams winning based on score differences."""
        probabilities = np.zeros((len(score_differences), 2))
        for i, diff in enumerate(score_differences):
            home_prob = self.home_team_win_probability(diff)
            away_prob = 1 - home_prob
            probabilities[i] = [away_prob, home_prob]
        return probabilities


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
