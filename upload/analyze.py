from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

sys.path.append(".")


data = []

bets = [[0, 0], [0, 0]]

K = 40


def eloAnalyze(season: pd.DataFrame):
    # tracks accuracy
    # on i-th position is a tuple:
    # 0) how many times did home win when his predicted winrate was i%
    # 1) how many times was home's predicted winrate i%

    eloDif = np.array([np.zeros(2) for i in range(100)])

    # calculates odds
    def helper(eloH, eloA):
        d = eloA - eloH
        # d = max(min(d,800), -800)
        A = 160 ** (d / 400)
        return 1 / (1 + A)

    # setups everyone's elo to 1000
    x = season.groupby("AID").groups.keys()
    homeElo = {i: 1000 for i in x}
    awayElo = {i: 1000 for i in x}
    print(len(homeElo.keys()))
    # count number of matches in this season
    i = 0
    for _, match in season.iterrows():
        i += 1
        hId = match["HID"]
        aId = match["AID"]
        hScore = match["H"]
        aScore = match["A"]

        eloH = homeElo[hId]
        eloA = awayElo[aId]

        # calculates the odds of home to win
        expected = helper(eloH, eloA)

        # adjusts elo
        homeElo[hId] += (hScore - expected) * K
        awayElo[aId] += (aScore - (1 - expected)) * K

        # if elo had enough time to stabilize, adjust expected dif
        if i > 400:
            realWinner = hScore > aScore
            eloDif[int(expected * 100)][0] += realWinner
            eloDif[int(expected * 100)][1] += 1

    return eloDif


def glicoAnalyze(season: pd.DataFrame):
    C = 35
    q = math.log(10) / 1000
    # tracks accuracy
    # on i-th position is a tuple:
    # 0) how many times did home win when his predicted winrate was i%
    # 1) how many times was home's predicted winrate i%

    eloDif = np.array([np.zeros(2) for i in range(100)])

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


##################
# IGNORE FOR NOW #
##################
def assignElo(season: pd.DataFrame):
    def helper(eloH, eloA):
        d = eloA - eloH
        # d = max(min(d,800), -800)
        A = 160 ** ((d) / 400)
        return 1 / (1 + A)

    x = season.groupby("AID").groups.keys()
    homeElo = {i: 1000 for i in x}
    awayElo = {i: 1000 for i in x}
    c = 0
    x = []
    for i, match in season.iterrows():
        c += 1
        hId = match["HID"]
        aId = match["AID"]
        match["OddsH"]
        match["OddsA"]
        hScore = match["H"]
        aScore = match["A"]

        eloH = homeElo[hId]
        eloA = awayElo[aId]

        expected = helper(eloH, eloA)

        homeElo[hId] += (hScore - expected) * K
        awayElo[aId] += (aScore - (1 - expected)) * K
        if i > 200:
            x.append(expected)
        else:
            x.append(None)

    season["Elo"] = x
    return season


# load everything

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])


withElo = pd.read_csv("./data/withelo.csv", index_col=0)
withElo["Date"] = pd.to_datetime(games["Date"])
withElo["Open"] = pd.to_datetime(games["Open"])


def runSeason(function):
    x = None
    for _i, season in games.groupby("Season"):
        # print("SEASON", i)
        if x is None:
            x = function(season)
        elif type(x) == pd.DataFrame:
            x = pd.concat([x, function(season)])
        else:
            x += function(season)
    return x


# RELEVANT MAIN
def analyzeElo() -> None:
    eloDif = runSeason(eloAnalyze)
    a = eloDif[:, 0]
    b = eloDif[:, 1]
    winrate = a / b

    # filter out all NaNs
    mask = np.logical_not(np.isnan(winrate))
    actual = winrate[mask] * 100
    expected = np.arange(100)[mask]

    print(np.corrcoef(actual, expected))
    dif = actual - expected
    print(np.var(dif))
    print(sum(dif) / len(dif))
    print(dict(zip(expected, actual)))


def analyzeWithElo() -> None:
    matrix = [[0, 0], [0, 0], [0, 0]]
    predicted = pd.DataFrame()
    predicted["P"] = withElo["Elo"].map(
        lambda a: 0 if a < 0.3 else (1 if a < 0.7 else 2)
    )
    predicted["R"] = withElo["H"]
    for _i, p in predicted.iterrows():
        matrix[p["P"]][p["R"]] += 1


def tryBets() -> None:
    # Home
    home = pd.DataFrame()

    # home["B"] = withElo.apply(lambda line: 1 if 1/line["OddsH"] < 0.25 and line["Elo"] * line["OddsH"] > 2 else 0,axis=1)
    for i in range(100, 300, 1):
        i = i / 100
        home["B"] = withElo.apply(
            lambda line: 1 if line["Elo"] * line["OddsH"] > i else 0, axis=1
        )
        home["Win"] = home["B"] * withElo["H"]
        home["Profit"] = home["B"] * withElo["H"] * withElo["OddsH"]
        print(
            i,
            f'won {home["Win"].sum()}/{home["B"].sum()} [{home["Profit"].sum()}] = {home["Profit"].sum() - home["B"].sum()}',
        )

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
    x["Elo"] = withElo["Elo"]
    x["H"] = withElo["H"]
    x["N"] = withElo["N"]
    x["POFF"] = withElo["POFF"]
    x["OddsH"] = withElo["OddsH"]
    x["OddsA"] = withElo["OddsA"]
    x["RequiredH"] = 1 / x["OddsH"]
    x["RequiredA"] = 1 / x["OddsA"]
    print(x.nsmallest(50, "RequiredH"))
    print(x.nsmallest(50, "RequiredA"))


def anaylzeGlico() -> None:
    eloDif = runSeason(glicoAnalyze)
    a = eloDif[:, 0]
    b = eloDif[:, 1]
    winrate = a / b

    # filter out all NaNs
    mask = np.logical_not(np.isnan(winrate))
    actual = winrate[mask] * 100
    expected = np.arange(100)[mask]

    print(np.corrcoef(actual, expected))
    dif = actual - expected
    print(np.var(dif))
    print(sum(dif) / len(dif))
    print(dict(zip(expected, actual)))


analyzeElo()
