import sys

import pandas as pd
import numpy as np
sys.path.append(".")

from environment import Environment
from model import Model




bets = [[0,0],[0,0]]
guess = [[0,0],[0,0]]
def analyzeForSeason(season: pd.DataFrame, support_col: str):
    def helper(scores, sup, home = None):
        if scores[1] < 10:
            return None
        return scores[0]/scores[1] #+ sup[0]/sup[1]
    
    tie = 0
    avgScoresHome = {}
    avgScoresAway = {}
    avgSupHome = {}
    avgSupAway = {}
    for i, match in season.iterrows():
        skip = False
        hId = match["HID"]
        aId = match["AID"]
        hScore = match["HSC"]
        aScore = match["ASC"]
        hSup = match["H"+support_col]
        aSup = match["A"+support_col]
        betsH = match["OddsH"]
        betsA = match["OddsA"]

        if hId not in avgScoresHome:
            avgScoresHome[hId] = [0,0]
            avgSupHome[hId] = [0, 0]
            skip = True
        if aId not in avgScoresAway:
            avgScoresAway[aId] = [0,0]
            avgSupAway[aId] = [0, 0]
            skip = True


        if hScore == aScore:
            tie += 1


        if not skip:
            predHome = helper(avgScoresHome[hId], avgSupHome[hId])
            predAway = helper(avgScoresAway[aId], avgSupAway[aId])
            if predHome is not None and predAway is not None and abs(predHome-predAway) > 25:
                predictedWinner = helper(avgScoresHome[hId], avgSupHome[hId], True) >= helper(avgScoresAway[aId], avgSupAway[aId], False)
                realWinner = hScore > aScore
                guess[predictedWinner][realWinner] += 1
                bets.append((predictedWinner and realWinner) * betsH + \
                    ((not predictedWinner) and (not realWinner)) * betsA)
            else:
                bets.append(1)

        avgScoresHome[hId][0] += hScore
        avgScoresHome[hId][1] += 1
        avgSupHome[hId][0] += hSup
        avgSupHome[hId][1] += 1


        avgScoresAway[aId][0] += aScore
        avgScoresAway[aId][1] += 1
        avgSupAway[aId][0] += aSup
        avgSupAway[aId][1] += 1
    #print(guess)
    #print(tie)
        
    return

    # grouped by home id
    homeIds = season[1].groupby("HID")
    homeScores = homeIds["HSC"].aggregate(["count","mean","std"]).rename(columns={"HID": "ID", "mean": "home_m", "std": "home_s"})
    homeWins = homeIds["H"].sum()
    homeGames = homeIds["H"].count()
    homeScores["H_wins"] = homeWins/homeGames
    print(homeScores)




    awayIds = season[1].groupby("AID")
    awayWins = awayIds["A"].sum()
    awayGames = awayIds["A"].count()
    awayScores = awayIds["ASC"].aggregate(["count","mean","std"]).rename(columns={"AID": "ID", "mean": "away_m", "std": "away_s"})
    awayScores["A_wins"] = awayWins/awayGames
    
    print(pd.merge(homeScores, awayScores, left_on="HID", right_on="AID",how="left"))

    
K = 20

def eloSeason(season: pd.DataFrame, base):
    eloDif = np.array([np.zeros(2) for i in range(100)])
    def helper(eloH, eloA):
        d = eloA-eloH
        #d = max(min(d,800), -800)
        A = base**((d)/400)
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
        if c > 400:
            eloDif[int(expected*100)][0] += realWinner
            eloDif[int(expected*100)][1] += 1

    return eloDif
        
        
        





games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])


def main(i):
    x = np.zeros([100, 2])
    for i, season in games.groupby("Season"):
        #print("SEASON", i)
        x += eloSeason(season, i)
    return x


for i, col in enumerate(games.columns):
    if i < 13 or i%2 == 0:
        continue
    ...
    #main(col[1:])
eloDif = main(i)
#print(bets)
#print(guess)
#print(bets)
#print(sum(bets[0])+sum(bets[1]))
#print(games["HID"].count())
x = [i/j if j!= 0 else None for i,j in eloDif]

# filter out only where x is not None
a,b = map(np.array,zip(*filter(lambda a: a[0] is not None and a[1] is not None, zip(range(100),x))))
#print(dict(zip(a,b)))
dif = a-100*b
print()


print(sum(dif*dif)/len(dif))
print(sum(dif)/len(a))
print(np.corrcoef(a,b))