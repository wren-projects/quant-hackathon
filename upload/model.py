from __future__ import annotations
import sys
import math
from collections import namedtuple
from copy import deepcopy
from enum import IntEnum
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


def plot_bad_predictions(Y_true, probabilities) -> None:
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
        #print(ratios)
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
        #print(np.array(result.x).reshape(probabilities.shape))
        return np.array(result.x).reshape(probabilities.shape)

    def get_betting_strategy(
        self,
        probabilities: np.ndarray,
        active_matches: pd.DataFrame,
        summary: Summary,
    ) -> np.ndarray:
        """Return absolute cash numbers and on what to bet in 2d list."""
        proportions: list[float] = (
            self.get_bet_proportions(probabilities, active_matches, summary)
            * summary.Bankroll
        )
        return np.array(proportions).round(decimals=0)


class EloTeam:
    """
    Class for keeping track of a team's Elo.

    Attributes:
        opponents: Sum of Elo ratings of opponents
        games: Number of games
        wins: Number of wins
        rating: Current Elo rating

    """

    K: int = 40
    base: int = 160
    opponents: int
    games: int
    wins: int
    rating: float

    def __init__(self) -> None:
        """Initialize TeamElo."""
        self.games = 0
        self.wins = 0
        self.rating = 1000

    def adjust(self, opponentElo: float, win: int) -> None:
        """
        Adjust Elo rating based on one match.

        Args:
            opponent: Elo rating of the other team
            win: 1 for win, 0 for loss

        """
        self.games += 1
        self.wins += 1 if win else 0
        
        expected = self.predict(opponentElo)

        self.rating += self.K * (win - expected)

    def predict(self, opponentElo: float):
        d = opponentElo - self.rating

        A = self.base**(d/400)
        expected = 1/(1+A)
        return expected
    

    def __str__(self) -> str:
        """Create a string representation of the team's Elo."""
        return (
            f"{self.rating:>4} ({self.games:>4}, "
            f"{self.wins:>4}, {self.wins / self.games * 100:>6.2f}%)"
        )


class Elo(RankingModel):
    """Class for the Elo ranking model."""

    elo_home_by_season: dict[int,dict[int, EloTeam]]
    elo_away_by_season: dict[int,dict[int, EloTeam]]
    def __init__(self) -> None:
        """Initialize Elo model."""
        self.elo_home_by_season = {}
        self.elo_away_by_season = {}

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
        season = match.Season
        if season not in self.elo_home_by_season:
            self.elo_home_by_season[season] = {}
            self.elo_away_by_season[season] = {}
        elo_home = self.elo_home_by_season[season]
        elo_away = self.elo_away_by_season[season]


        if match.HID not in elo_home:
            elo_home[match.HID] = EloTeam()

        if match.AID not in elo_away:
            elo_away[match.AID] = EloTeam()

        home_elo = elo_home[match.HID]
        away_elo = elo_away[match.AID]

        # memorize elo values before they change
        home_elo_value = home_elo.rating
        away_elo_value = away_elo.rating

        home_elo.adjust(away_elo_value, match.H)
        away_elo.adjust(home_elo_value, match.A)
        
    def predict(self, match: Opp) -> float|None:
        """
        Predicts how the match might go.
        Float from 0 to 1 = chance of H to win
        None means no data
        """
        season = match.Season
        if season not in self.elo_home_by_season:
            self.elo_home_by_season[season] = {}
            self.elo_away_by_season[season] = {}

        elo_home = self.elo_home_by_season[season]
        elo_away = self.elo_away_by_season[season]

        if match.HID not in elo_home:
            elo_home[match.HID] = EloTeam()

        if match.AID not in elo_away:
            elo_away[match.AID] = EloTeam()

        home_elo = elo_home[match.HID]
        away_elo = elo_away[match.AID]
        if home_elo.games < 10 or away_elo.games < 10:
            return None
        return home_elo.predict(away_elo.rating)

        

class Model:
    """Main class."""

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples()):
            self.elo.add_match(match)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opportunities: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        #print("new round")
        self.update_models(inc[0])


        summary = Summary(*summ.iloc[0])
        maxBet = summary.Max_bet
        bankroll = summary.Bankroll
        
        opportunities["ELO"] = opportunities.apply(lambda row: self.elo.predict(row),axis=1)
        opportunities["EV"] = opportunities["ELO"]*opportunities["OddsH"]
        toBetOn = opportunities[opportunities["ELO"] is not None and opportunities["EV"] >= 2]
        toBetOn = toBetOn.sort_values(by=["EV"], ascending=False)

        toBetOn["BetH"] = maxBet

        # not enough money
        if toBetOn["BetH"].sum() > bankroll:
            rows_to_discard = int((toBetOn["BetH"].sum()-bankroll)//maxBet)
            if rows_to_discard > 0:
                toBetOn = toBetOn[:-rows_to_discard]
            toBetOn.loc[toBetOn.index[-1],"BetH"] -= (toBetOn["BetH"].sum()-bankroll)%maxBet


        toBetOn["BetA"] = 0
        return toBetOn