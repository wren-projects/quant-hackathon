from __future__ import annotations

import functools
import sys
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openskill.models import BradleyTerryFull, BradleyTerryPart, ThurstoneMostellerFull
from predictionModels.copiedopenskill import OpenSkillAPI
from predictionModels.elo import Elo
from predictionModels.pagerank import PageRank

pd.options.mode.chained_assignment = None

sys.path.append(".")


data = []

bets = [[0, 0], [0, 0]]

K = 40


def implement_model(modelClass: any, **kwargs) -> Callable[[pd.DataFrame], list[float]]:
    def helper(season: pd.DataFrame) -> list[float]:
        """Predicts the season using pageran."""
        model = modelClass(**kwargs)

        results = []
        # count number of matches in this season
        for _, match in season.iterrows():
            expected = model.predict(match)
            model.add_match(match)

            results.append(expected)

        return results

    return helper


pagerank_model = implement_model(PageRank)
openskill_model = implement_model(OpenSkillAPI)
elo_model = implement_model(Elo)


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
    my_games = games[games["Season"] > 10]

    x = []
    for i, season in my_games.groupby("Season"):
        if i < 10:
            continue

        season["PRED"] = function(season)
        my_games.loc[season.index[:matches_to_discard_each_season], "EARLY"] = 1
        x += function(season)
    my_games["PRED"] = x
    return my_games[my_games["EARLY"] == 0]


def try_bets(
    model: Callable[[pd.DataFrame], list[float]],
    bettings_strategy_home: tuple[float, float] = [0, 2],
    bettings_strategy_away: tuple[float, float] = [0, 2],
    discard_per_season: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate betting.

    Bettings strategy format is (min_certainty, treshold). Where min_certainty is the
    minimal predicted winrate and treshold is EV
    treshold above which a bet is made.
    """
    games_with_predictions = add_predictions_to_season(
        model, matches_to_discard_each_season=discard_per_season
    )

    games_with_predictions["BH"] = games_with_predictions.apply(
        lambda row: row["PRED"] >= bettings_strategy_home[0]
        and (row["PRED"] * row["OddsH"]) > bettings_strategy_home[1],
        axis=1,
    )

    games_with_predictions["BA"] = games_with_predictions.apply(
        lambda row: (1 - row["PRED"]) >= bettings_strategy_away[0]
        and (1 - row["PRED"]) * row["OddsA"] > bettings_strategy_away[1],
        axis=1,
    )

    games_with_predictions["WONH"] = (
        games_with_predictions["BH"] * games_with_predictions["H"]
    )
    games_with_predictions["WONA"] = (
        games_with_predictions["BA"] * games_with_predictions["A"]
    )
    games_with_predictions["HPROFIT"] = (
        games_with_predictions["WONH"] * games_with_predictions["OddsH"]
        - games_with_predictions["BH"]
    )
    games_with_predictions["APROFIT"] = (
        games_with_predictions["WONA"] * games_with_predictions["OddsA"]
        - games_with_predictions["BA"]
    )
    """print(
        "Total bets",
        games_with_predictions["BA"].sum() + games_with_predictions["BH"].sum(),
    )"""
    by_season = games_with_predictions.groupby("Season")

    # print(by_season["WONH"].sum().values / by_season["BH"].sum().values)
    # print(by_season["WONA"].sum().values / by_season["BA"].sum().values)
    return by_season["HPROFIT"].sum(), by_season["APROFIT"].sum()


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


# analyzeOpenSKill()
# plt.plot(range(100), range(100))
# print(bets)
# analyze_rating(pageran_model)
def main_analyze() -> None:
    models = [openskill_model, elo_model(base=10), elo_model(base=160)]
    name = [
        "open skill",
        "elo B = 10",
        "elo B= 160",
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
    models = [openskill_model]
    for i, model in enumerate(models):
        print(model)
        max_home = -10000
        for EV in range(100, 150, 5):
            EV /= 100
            for threshhold in range(0, 90, 10):
                threshhold /= 100
                discard = 140
                home, away = try_bets(
                    model=model,
                    bettings_strategy_home=[threshhold, EV, 0.2],
                    bettings_strategy_away=[2, 1],
                    discard_per_season=discard,
                )
                home = home.sum()
                if max_home < home:
                    treshold_away = EV, threshhold
                    max_home = home

        print("Tresholds", treshold_away)
        print("Away", max_home)


def currentBest():
    _, a = try_bets(
        openskill_model,
        bettings_strategy_home=[2, 0],
        bettings_strategy_away=[0.66, 2],
        discard_per_season=140,
    )
    b, _ = try_bets(
        elo_model,
        bettings_strategy_home=[0, 2],
        bettings_strategy_away=[2, 1],
        discard_per_season=60,
    )

    plot = plt.subplot()
    print(a.values)
    plot.plot(a.index, a.values)
    plot.plot(b.index, b.values)
    plot.legend(["away", "home"])
    plt.show()
    # openRating [0.66, 2], discard 140
    """
    a, _ = try_bets(
        model=pagerank_model,
        bettings_strategy_home=[0, 1.45],
        bettings_strategy_away=[100, 1],
        discard_per_season=60,
    )
    """
    print(a.sum(), b.sum())
    # print("Current best is", a, b)


currentBest()
# main_analyze()
# main_bets()
