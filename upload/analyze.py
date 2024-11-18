from __future__ import annotations

import functools
import sys
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copiedopenskill import PlackettLuce
from openskill.models import BradleyTerryFull, BradleyTerryPart, ThurstoneMostellerFull

pd.options.mode.chained_assignment = None

sys.path.append(".")


data = []

bets = [[0, 0], [0, 0]]

K = 40


class PageRank:
    """Class for the page rank model."""

    def __init__(
        self, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100
    ) -> None:
        """
        Initialize the BasketballPageRank class.

        Args:
        - alpha: Damping factor for PageRank.
        - tol: Convergence tolerance.
        - max_iter: Maximum number of iterations.

        """
        self.alpha: float = alpha
        self.tol: float = tol
        self.max_iter: int = max_iter
        self.teams: dict = {}
        self.games: list = []

    def add_match(self, home_id: int, away_id: int, result: bool) -> None:
        """
        Add a match to the dataset.

        Args:
        - match: line with data

        """
        if home_id not in self.teams:
            self.teams[home_id] = len(self.teams)
        if away_id not in self.teams:
            self.teams[away_id] = len(self.teams)

        # FIXME???
        if result:
            self.games.append((away_id, home_id))
        else:
            self.games.append((home_id, away_id))

    def _calculate_ratings(self) -> dict:
        """
        Calculate the PageRank ratings for all teams.

        Returns:
        - ranks: Dictionary mapping team IDs to their PageRank scores.
        Here is where the magic happen.

        """
        n = len(self.teams)
        if n == 0:
            return {}

        # team_index = {team: idx for team, idx in self.teams.items()}
        team_index = dict(self.teams.items())

        # Build adjacency matrix
        m = np.zeros((n, n))
        for loser, winner in self.games:
            m[team_index[winner], team_index[loser]] += 1

        # Normalize the matrix
        for i in range(n):
            if m[i].sum() > 0:
                m[i] /= m[i].sum()
            else:
                m[i] = 1 / n  # Handle dangling nodes

        # PageRank algorithm
        rank = np.ones(n) / n
        for _ in range(self.max_iter):
            new_rank = self.alpha * m.T @ rank + (1 - self.alpha) / n
            if np.linalg.norm(new_rank - rank, ord=1) < self.tol:
                break
            rank = new_rank

        # Map scores to teams
        return {team: rank[team_index[team]] for team in self.teams}

    def team_rating(self, team_id1: int, team_id2: int) -> tuple:
        """
        Get the rating of one or two teams.

        Args:
        - team_id1: ID of the first team.
        - team_id2: ID of the second team.

        Returns:
        - vecstor of ratings

        """
        ratings = self._calculate_ratings()
        return (ratings.get(team_id1, None), ratings.get(team_id2, None))

    def predict(self, home_id: int, away_id: int) -> float:
        """
        Get the ratio of winning two teams.

        Args:
        - team_id1: ID of the first team.
        - team_id2: ID of the second team.

        Returns:
        - ratio of winning(home 0, away 100)

        """
        ratings = self._calculate_ratings()
        if home_id not in ratings or away_id not in ratings:
            return 0.5
        ratio: float = ratings.get(away_id, None) / (
            ratings.get(home_id, None) + ratings.get(away_id, None)
        )
        return ratio


def pagerank_model(season: pd.DataFrame) -> list[float]:
    """Predicts the season using pageran."""
    model = PageRank()

    results = []
    # count number of matches in this season
    for _, match in season.iterrows():
        home_id = match["HID"]
        away_id = match["AID"]
        home_score = match["H"]

        expected = model.predict(home_id, away_id)
        model.add_match(home_id, away_id, home_score)

        results.append(expected)

    return results


def openskill_model(
    season: pd.DataFrame,
    modelType: Type[BradleyTerryFull | BradleyTerryPart | PlackettLuce],
) -> list[float]:
    """Predicts the season using openskill."""
    model = modelType()

    x = season.groupby("AID").groups.keys()
    rating_database = {i: [model.mu, model.sigma] for i in x}

    results = []
    # count number of matches in this season
    for _, match in season.iterrows():
        home_id = match["HID"]
        away_id = match["AID"]
        home_score = match["H"]

        elo_home = [
            model.rating(
                rating_database[home_id][0],
                rating_database[home_id][1],
            )
        ]
        elo_away = [
            model.rating(rating_database[away_id][0], rating_database[away_id][1])
        ]

        expected = model.predict_win([elo_home, elo_away])[0]
        # adjusts elo
        if home_score:
            [[new_elo_home], [new_elo_away]] = model.rate([elo_home, elo_away])
        else:
            [[new_elo_away], [new_elo_home]] = model.rate([elo_away, elo_home])

        rating_database[home_id] = [new_elo_home.mu, new_elo_home.sigma]
        rating_database[away_id] = [new_elo_away.mu, new_elo_away.sigma]

        results.append(expected)

    return results


def elo_model(season: pd.DataFrame) -> list[float]:
    """Predicts the season using elo."""

    # calculates odds
    def helper(elo_home: float, elo_away: float) -> float:
        d = elo_away - elo_home
        # d = max(min(d,800), -800)
        a = 160 ** ((d) / 400)
        return 1 / (1 + a)

    # setups everyone's elo to 1000
    x = season.groupby("AID").groups.keys()
    home_rating_database = {i: 1000 for i in x}
    away_rating_database = {i: 1000 for i in x}

    results = []
    # count number of matches in this season
    for _, match in season.iterrows():
        home_id = match["HID"]
        away_id = match["AID"]
        home_score = match["H"]
        away_score = match["A"]

        elo_home = home_rating_database[home_id]
        elo_away = away_rating_database[away_id]

        # calculates the odds of home to win
        expected = helper(elo_home, elo_away)

        # adjusts elo
        home_rating_database[home_id] += (home_score - expected) * K
        away_rating_database[away_id] += (away_score - (1 - expected)) * K
        results.append(expected)

    return results


# load everything

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])


@functools.cache
def add_predictions_to_season(
    function: Callable[[pd.DataFrame], list[float]],
    matches_to_discard_each_season: int = 64,
) -> pd.DataFrame:
    """Add models prediction as a new column into games (called "PRED")."""
    games["EARLY"] = 0

    x = []
    for _, season in games.groupby("Season"):
        season["PRED"] = function(season)
        games.loc[season.index[:matches_to_discard_each_season], "EARLY"] = 1
        x += function(season)
    games["PRED"] = x
    return games[games["EARLY"] == 0]


def try_bets(
    model: Callable[[pd.DataFrame], list[float]],
    bettings_strategy_home: tuple[float, float] = [0, 2],
    bettings_strategy_away: tuple[float, float] = [0, 2],
    discard_per_season: int = 64,
    print_output: bool = True,
) -> tuple[float, float]:
    """
    Simulate betting.

    Bettings strategy format is (offset, treshold). Where offset is the
    value that is subtracted from predicted winrate and treshold is EV
    treshold above which a bet is made.
    """
    games_with_predictions = add_predictions_to_season(
        model, matches_to_discard_each_season=discard_per_season
    )

    games_with_predictions["BH"] = games_with_predictions.apply(
        lambda row: (row["PRED"] - bettings_strategy_home[0]) * row["OddsH"]
        > bettings_strategy_home[1],
        axis=1,
    )
    games_with_predictions["BA"] = games_with_predictions.apply(
        lambda row: (1 - row["PRED"] - bettings_strategy_away[0]) * row["OddsA"]
        > bettings_strategy_away[1],
        axis=1,
    )

    print_output and print(
        "bets Home/Away",
        games_with_predictions["BH"].sum(),
        games_with_predictions["BA"].sum(),
    )

    profit_home = (
        games_with_predictions["BH"]
        * games_with_predictions["H"]
        * games_with_predictions["OddsH"]
        - games_with_predictions["BH"]
    ).sum()

    profit_away = (
        games_with_predictions["BA"]
        * games_with_predictions["A"]
        * games_with_predictions["OddsA"]
        - games_with_predictions["BA"]
    ).sum()

    print_output and print("Profit Home/Away", profit_home, profit_away)

    bets_home = games_with_predictions["BH"].sum() * 100
    winrate_home = (
        (games_with_predictions["BH"] * games_with_predictions["H"]).sum() / bets_home
        if bets_home != 0
        else 0
    )

    bets_away = games_with_predictions["BA"].sum() * 100
    winrate_away = (
        (games_with_predictions["BA"] * games_with_predictions["A"]).sum() / bets_away
        if bets_away != 0
        else 0
    )
    print_output and print("Winrate Home/Away", winrate_home, winrate_away)
    return profit_home, profit_away


def analyze_rating(
    model: Callable[[pd.DataFrame], list[float]],
    discard_per_season: int = 64,
    axis: None | any = None,
) -> None:
    """Analyzes model's rating capabilities."""
    data = add_predictions_to_season(
        model, matches_to_discard_each_season=discard_per_season
    )
    data = data[data["EARLY"] == 0]
    predictions_accuracy = np.array([np.zeros(2)] * 100)
    prediction_count = np.array([np.zeros(1)] * 100)
    for _, match in data.iterrows():
        predictions_accuracy[min(99, int(match["PRED"] * 100))][0] += match["H"]
        predictions_accuracy[min(99, int(match["PRED"] * 100))][1] += 1
        prediction_count[min(99, int(match["PRED"] * 100))] += 1

    winrate = predictions_accuracy[:, 0] / predictions_accuracy[:, 1]

    # filter out all NaNs
    mask = np.logical_not(np.isnan(winrate))
    actual = winrate[mask] * 100
    expected = np.arange(100)[mask]

    print(np.corrcoef(actual, expected))
    dif = actual - expected
    print(np.var(dif))
    print(sum(dif) / len(dif))

    if axis is None:
        figure, my_axis = plt.subplots(2, 1)
    else:
        my_axis = axis

    my_axis[0].plot(expected, expected)
    my_axis[0].plot(expected, actual)
    my_axis[0].set_title("Accuracy of guesses")
    my_axis[1].plot(range(100), prediction_count)
    my_axis[1].set_title("Number of guesses")
    if axis is None:
        plt.show()


def adjust_model(
    model: Callable[[pd.DataFrame], list[float]], a: float = 1, b: float = 0
) -> Callable[[pd.DataFrame], list[float]]:
    """
    Adjust a model that outputs a line of the form y=ax+b into a line of the form y=x.

    Still respects 0 as min and 1 as max values.
    """

    def helper(data: pd.DataFrame) -> list[float]:
        output = np.array(model(data))
        x = (output - b) / a + b
        x[x >= 1] = 0.99
        x[x <= 0] = 0.01
        return list(x)

    return helper


def openskill_with_specific_model(
    model: type[BradleyTerryFull | BradleyTerryPart | PlackettLuce],
) -> Callable[[pd.DataFrame], list[float]]:
    """Return openskill_model that uses the selected prediction model."""
    return lambda season: openskill_model(season, model)


# analyzeOpenSKill()
# plt.plot(range(100), range(100))
# print(bets)
# analyze_rating(pageran_model)
def main_analyze() -> None:
    models = [
        elo_model,
        openskill_with_specific_model(PlackettLuce),
        openskill_with_specific_model(BradleyTerryFull),
        pagerank_model,
    ]
    name = [
        "elo",
        "ThurstoneMostellerFull",
        "ThurstoneMostellerPart",
        "BradleyTerryPart",
        "BradleyTerryFull",
        "PageRank",
    ]
    plots, axis = plt.subplots(len(models), 2)
    for i, model in enumerate(models):
        analyze_rating(
            model=model,
            axis=axis[i],
            discard_per_season=128,
        )
        axis[i][0].set_title(axis[i][0].get_title() + name[i])
    plt.show()


def main_bets() -> None:
    models = [openskill_with_specific_model(PlackettLuce)]
    for i, model in enumerate(models):
        print(model)
        max_away = -10000
        for j in range(0, 20, 4):
            for threshold in range(120, 250, 25):
                discard = 140
                threshold /= 100
                home, away = try_bets(
                    model=model,
                    bettings_strategy_home=[100, 1],
                    bettings_strategy_away=[j, threshold],
                    discard_per_season=discard,
                    print_output=False,
                )
                if max_away < away:
                    treshold_away = threshold, j
                    max_away = away

        print("Tresholds", treshold_away)
        print("Away", max_away)


def currentBest():
    a, _ = try_bets(
        model=pagerank_model,
        bettings_strategy_home=[0, 2],
        bettings_strategy_away=[100, 1],
        discard_per_season=90,
    )

    _, b = try_bets(
        model=openskill_with_specific_model(PlackettLuce),
        bettings_strategy_home=[100, 1],
        bettings_strategy_away=[0, 2],
        discard_per_season=90,
    )
    print("Current best is", a + b)


currentBest()
# main_elo()
# main_analyze()
