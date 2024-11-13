import sys

import pandas as pd
import numpy as np
sys.path.append(".")

from environment import Environment
from model import Model




bets = [[0,0],[0,0]]
    
K = 20

def eloAnalyze(season: pd.DataFrame):
    # tracks accuracy
    # on i-th position is a tuple:
    # 0) how many times did home win when his predicted winrate was i%
    # 1) how many times was home's predicted winrate i%

    eloDif = np.array([np.zeros(2) for i in range(100)])

    # calculates odds
    def helper(eloH, eloA):
        d = eloA-eloH
        #d = max(min(d,800), -800)
        A = 160**((d)/400)
        return 1/(1+A)
    
    # setups everyone's elo to 1000
    x = season.groupby("AID").groups.keys()
    homeElo = {i: 1000 for i in x}
    awayElo = {i: 1000 for i in x}

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
        expected = helper(eloH,eloA)

        # adjusts elo
        homeElo[hId] += (hScore-expected) * K
        awayElo[aId] += (aScore-(1-expected)) * K

        
        # if elo had enough time to stabilize, adjust expected dif
        if i > 400:
            realWinner = hScore > aScore
            eloDif[int(expected*100)][0] += realWinner
            eloDif[int(expected*100)][1] += 1

    return eloDif



##################
# IGNORE FOR NOW #
##################
def eloUse(season: pd.DataFrame):
    guess = [[0,0],[0,0]]
    bets = [0,0]
    def helper(eloH, eloA):
        d = eloA-eloH
        #d = max(min(d,800), -800)
        A = 160**((d)/400)
        return 1/(1+A)
    
    x = season.groupby("AID").groups.keys()
    homeElo = {i: 1000 for i in x}
    awayElo = {i: 1000 for i in x}
    #awayElo = homeElo
    c = 0
    for i, match in season.iterrows():
        c += 1
        hId = match["HID"]
        aId = match["AID"]
        betsH = match["OddsH"]
        betsA = match["OddsA"]
        hScore = match["H"]
        aScore = match["A"]

        eloH = homeElo[hId]
        eloA = awayElo[aId]

        expected = helper(eloH,eloA)

        homeElo[hId] += (hScore-expected) * K
        awayElo[aId] += (aScore-(1-expected)) * K

        realWinner = hScore > aScore
        #print(expected*100)
        if c > 200:
            guess[expected>0.5][realWinner] += 1

            assert not (expected * betsH > 3 and (1-expected) * betsA > 3)
            # bet home
            if expected * betsH > 3:
                bets[0] += betsH*hScore - 1
            
            # bet home
            if (1-expected) * betsA > 3:
                bets[1] += betsA*aScore - 1
            
    print(bets)
    return bets
        



# load everything

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])



def analyze():
    x = np.zeros([100, 2])
    for i, season in games.groupby("Season"):
        #print("SEASON", i)
        x += eloAnalyze(season)
    return x

def test():
    x = np.zeros([2])
    for i, season in games.groupby("Season"):
        #print("SEASON", i)
        x += eloUse(season)
    return x




# RELEVANT MAIN
def analyzeElo():
    eloDif = analyze()
    a = eloDif[:,0]
    b = eloDif[:,1]
    winrate = a/b

    # filter out all NaNs
    mask = np.logical_not(np.isnan(winrate))
    actual = winrate[mask]*100
    expected = np.arange(100)[mask]

    print(np.corrcoef(actual, expected))
    dif = actual-expected
    print(np.var(dif))
    print(sum(dif)/len(dif))
    print(dict(zip(expected, actual)))



def testElo():
    cor = test()
    print(cor)

analyzeElo()