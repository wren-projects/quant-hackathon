from __future__ import annotations

import sys
import math
import matplotlib.pyplot as plt
from openskill.models import BradleyTerryFull, BradleyTerryPart, PlackettLuce
import pandas as pd
import numpy as np

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
    model = PlackettLuce()

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


def openskill_model(season: pd.DataFrame) -> list[float]:
    """Predicts the season using openskill."""
    model = PlackettLuce()

    x = season.groupby("AID").groups.keys()
    rating_database = {i: model.rating(name=i) for i in x}

    results = []
    # count number of matches in this season
    for _, match in season.iterrows():
        home_id = match["HID"]
        away_id = match["AID"]
        home_score = match["H"]

        elo_home = [rating_database[home_id]]
        elo_away = [rating_database[away_id]]

        expected = model.predict_win([elo_home, elo_away])[0]
        # adjusts elo
        if home_score:
            [[new_elo_home], [new_elo_away]] = model.rate([elo_home, elo_away])
        else:
            [[new_elo_away], [new_elo_home]] = model.rate([elo_away, elo_home])

        rating_database[home_id] = new_elo_home
        rating_database[away_id] = new_elo_away

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
    print(len(home_rating_database.keys()))

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


"""def glicoModel(season: pd.DataFrame):
    C = 35
    q = math.log(10) / 1000
    # tracks accuracy
    # on i-th position is a tuple:
    # 0) how many times did home win when his predicted winrate was i%
    # 1) how many times was home's predicted winrate i%

    def get_g(RD):
        return 1 / math.sqrt(1 + 3 * q**2 * RD**2 / math.pi**2)

    # calculates odds
    def helper(ratH, ratA):
        return 1 / (1 + 10 ** (get_g(ratA[1]) * (ratH[0] - ratA[1]) / (-400)))

    def adjustRating(
        ratH: tuple[int, int],
        ratA: tuple[int, int],
        daysSinceLastGame: int,
        result: bool,
    ):
        RD = min(math.sqrt(ratH[1] ** 2 + C**2 * daysSinceLastGame), 350)
        d_squared = 1 / (
            q**2 * get_g(ratA[1]) ** 2 * helper(ratH, ratA) * (1 - helper(ratH, ratA))
        )
        newR = ratH[0] + q / (1 / RD**2 + 1 / d_squared) * get_g(ratA[1]) * (
            result - helper(ratH, ratA)
        )
        newRD = 1 / math.sqrt(1 / RD**2 + 1 / d_squared)
        return [newR, newRD]

    # setups everyone's elo to 1000
    x = season.groupby("AID").groups.keys()
    homeGlico = {i: [1500, 350] for i in x}
    awayGlico = {i: [1500, 350] for i in x}
    lastGameHome = {i: pd.Timestamp(0) for i in x}
    lastGameAway = {i: pd.Timestamp(0) for i in x}
    # count number of matches in this season
    i = 0
    for _, match in season.iterrows():
        i += 1
        hId = match["HID"]
        aId = match["AID"]
        hScore = match["H"]
        aScore = match["A"]
        date = match["Date"]
        eloH = homeGlico[hId]
        eloA = awayGlico[aId]

        # calculates the odds of home to win
        expected = helper(eloH, eloA)

        # adjusts elo
        homeGlico[hId] = adjustRating(
            eloH, eloA, (date - lastGameHome[hId]).days, hScore > aScore
        )
        awayGlico[aId] = adjustRating(
            eloA, eloH, (date - lastGameAway[aId]).days, hScore < aScore
        )
        lastGameHome[hId] = date
        lastGameAway[hId] = date

        # if elo had enough time to stabilize, adjust expected dif
        if i > 200:
            realWinner = hScore > aScore
            eloDif[int(expected * 100)][0] += realWinner
            eloDif[int(expected * 100)][1] += 1

    return eloDif
"""


# load everything

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])


def add_predictions_to_season(function: callable) -> any:
    x = []
    for i, season in games.groupby("Season"):
        x += function(season)
    games["PRED"] = x
    return games


def analyze_with_elo() -> None:
    matrix = [[0, 0], [0, 0], [0, 0]]
    predicted = pd.DataFrame()
    predicted["P"] = withElo["Elo"].map(
        lambda a: 0 if a < 0.3 else (1 if a < 0.7 else 2)
    )
    predicted["R"] = withElo["H"]
    for i, p in predicted.iterrows():
        matrix[p["P"]][p["R"]] += 1


def try_bets() -> None:
    # Home
    home = pd.DataFrame()
    profit = []
    success = []
    for i in range(100, 300, 2):
        print(i)
        i = i / 100  # noqa: PLW2901
        home["B"] = withRating.apply(
            lambda line, i=i: 1 if line["RATING"] * line["OddsH"] > i else 0, axis=1
        )
        home["Win"] = home["B"] * withRating["H"]
        home["Profit"] = home["B"] * withRating["H"] * withRating["OddsH"]
        profit.append(home["Profit"].sum() - home["B"].sum())
        success.append(home["Win"].sum() / home["B"].sum())
    print(max(profit))
    plt.plot(range(100, 300, 2), profit)
    plt.plot(range(100, 300, 2), success)
    plt.legend(["Profit", "success"])
    plt.show()
    """
    #Away
    away = pd.DataFrame()

    away["B"] = withElo.apply(lambda line: 1 if 1/line["OddsA"] < 0.75 and (1-line["Elo"]) * line["OddsA"] > 3 else 0,axis=1)
    away["Win"] = away["B"] * withElo["A"]
    away["Profit"] = away["B"] * withElo["A"] * withElo["OddsA"]

    print("bet", away["B"].sum())
    print("won", away["Win"].sum())
    print("won", away["Profit"].sum())
    """


def analyzeBets() -> None:
    x = pd.DataFrame()
    x["Elo"] = withRating["RATING"]
    x["H"] = withRating["H"]
    x["N"] = withRating["N"]
    x["POFF"] = withRating["POFF"]
    x["OddsH"] = withRating["OddsH"]
    x["OddsA"] = withRating["OddsA"]
    x["RequiredH"] = 1 / x["OddsH"]
    x["RequiredA"] = 1 / x["OddsA"]
    print(x.nsmallest(50, "RequiredH"))
    print(x.nsmallest(50, "RequiredA"))


def analyze_rating(model) -> None:
    data = add_predictions_to_season(model)
    predictions_accuracy = np.array([np.zeros(2)] * 100)
    prediction_count = np.array([np.zeros(1)] * 100)
    lastSeason = None
    count = 0
    for _, match in data.iterrows():
        count += 1
        if match["Season"] != lastSeason:
            lastSeason = match["Season"]
            count = 0
        if count < 64:
            continue
        predictions_accuracy[min(99, int(match["PRED"] * 100))][0] += match["H"]
        predictions_accuracy[min(99, int(match["PRED"] * 100))][1] += 1
        prediction_count[min(99, int(match["PRED"] * 100))] += 1

    figure, axis = plt.subplots(2, 1)
    winrate = predictions_accuracy[:, 0] / predictions_accuracy[:, 1]

    # filter out all NaNs
    mask = np.logical_not(np.isnan(winrate))
    actual = winrate[mask] * 100
    expected = np.arange(100)[mask]
    print(np.corrcoef(actual, expected))
    dif = actual - expected
    print(np.var(dif))
    axis[0].plot(expected, actual)
    axis[0].set_title("Accuracy of guesses")
    print(sum(dif) / len(dif))
    axis[1].plot(range(100), prediction_count)
    axis[1].set_title("Number of guesses")
    plt.show()


# analyzeOpenSKill()
# plt.plot(range(100), range(100))
# print(bets)
# analyze_rating(pageran_model)

analyze_rating(openskill_model)
