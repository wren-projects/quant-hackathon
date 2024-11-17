import sys
import math
import matplotlib.pyplot as plt
from openskill.models import BradleyTerryPart
import pandas as pd
import numpy as np

sys.path.append(".")


data = []

bets = [[0, 0], [0, 0]]

K = 40


openSkillRating = {}


def openSkillAnalyze(season: pd.DataFrame):
    # tracks accuracy
    # on i-th position is a tuple:
    # 0) how many times did home win when his predicted winrate was i%
    # 1) how many times was home's predicted winrate i%
    global openSkillRating

    predictionDif = np.array([np.zeros(2) for i in range(100)])
    predictionCount = np.array([np.zeros(1) for i in range(100)])
    model = BradleyTerryPart()

    x = season.groupby("AID").groups.keys()
    rating = {i: model.rating(name=i) for i in x}
    # rating.update(openSkillRating)

    # for i in rating:
    #    rating[i].sigma = model.sigma
    # count number of matches in this season
    i = 0
    for _, match in season.iterrows():
        i += 1
        hId = match["HID"]
        aId = match["AID"]
        hScore = match["H"]
        aScore = match["A"]

        eloH = [rating[hId]]
        # eloA = [awayElo[aId]]
        eloA = [rating[aId]]

        # calculates the odds of home to win
        expected = model.predict_win([eloH, eloA])[0]
        # adjusts elo
        if hScore:
            [[newHome], [newAway]] = model.rate([eloH, eloA])
        else:
            [[newAway], [newHome]] = model.rate([eloA, eloH])

        rating[hId] = newHome
        rating[aId] = newAway

        # if elo had enough time to stabilize, adjust expected dif
        realWinner = hScore > aScore
        predictionDif[min(99, int(expected * 100))][0] += realWinner
        predictionDif[min(99, int(expected * 100))][1] += 1
        predictionCount[min(99, int(expected * 100))] += 1

        # bet home
        if expected > 0.7:
            bets[0][aScore] += 1
        if expected < 0.3:
            bets[1][aScore] += 1

    openSkillRating.update(rating)
    # return predictionCount
    return predictionDif


def eloAnalyze(season: pd.DataFrame):
    # tracks accuracy
    # on i-th position is a tuple:
    # 0) how many times did home win when his predicted winrate was i%
    # 1) how many times was home's predicted winrate i%

    predictionCount = np.array([np.zeros(1) for i in range(100)])
    eloDif = np.array([np.zeros(2) for i in range(100)])

    # calculates odds
    def helper(eloH, eloA):
        d = eloA - eloH
        # d = max(min(d,800), -800)
        A = 160 ** ((d) / 400)
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
            predictionCount[int(expected * 100)] += 1

    return predictionCount


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
def assignOpen(season: pd.DataFrame):
    global openSkillRating

    predictionDif = np.array([np.zeros(2) for i in range(100)])
    predictionCount = np.array([np.zeros(1) for i in range(100)])
    model = BradleyTerryPart()

    x = season.groupby("AID").groups.keys()
    rating = {i: model.rating(name=i) for i in x}
    rating.update(openSkillRating)

    for i in rating:
        rating[i].sigma += model.sigma / 18

    i = 0
    x = []
    for _, match in season.iterrows():
        i += 1
        hId = match["HID"]
        aId = match["AID"]
        hScore = match["H"]
        aScore = match["A"]

        eloH = [rating[hId]]
        # eloA = [awayElo[aId]]
        eloA = [rating[aId]]

        # calculates the odds of home to win
        expected = model.predict_win([eloH, eloA])[0]
        # adjusts elo
        if hScore:
            [[newHome], [newAway]] = model.rate([eloH, eloA])
        else:
            [[newAway], [newHome]] = model.rate([eloA, eloH])

        rating[hId] = newHome
        rating[aId] = newAway
        x.append(expected)

    season["RATING"] = x
    return season


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
withElo["RATING"] = withElo["Elo"]
withRating = pd.read_csv("./data/withRating.csv", index_col=0)
withRating["Date"] = pd.to_datetime(games["Date"])
withRating["Open"] = pd.to_datetime(games["Open"])


def runSeason(function: callable) -> any:
    x = None
    for i, season in games.groupby("Season"):
        # print("SEASON", i)
        if x is None:
            x = function(season)
        elif type(x) is pd.DataFrame:
            x = pd.concat([x, function(season)])
        else:
            x += function(season)
    return x


# RELEVANT MAIN
def analyzeElo() -> None:
    eloDif = runSeason(eloAnalyze)
    plt.plot(range(100), eloDif)
    return
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


def analyzeOpenSKill() -> None:
    eloDif = runSeason(openSkillAnalyze)
    # plt.plot(range(100), eloDif)
    # plt.show()
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
    plt.plot(range(100), actual)
    print(sum(dif) / len(dif))
    # print(dict(zip(expected, actual)))


# analyzeOpenSKill()
# plt.plot(range(100), range(100))
# print(bets)
try_bets()
runSeason(assignOpen).to_csv("data/withRating.csv")
