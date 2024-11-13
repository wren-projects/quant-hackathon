import sys

import pandas as pd
import numpy as np
sys.path.append(".")

from environment import Environment
from model import Model




bets = [[0,0],[0,0]]
guess = [[0,0],[0,0]]
    
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