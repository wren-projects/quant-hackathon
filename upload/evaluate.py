import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(".")

from environment import Environment

# from model import Model
from x import Model

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])


year = 98
env = Environment(
    games,
    players,
    Model(),
    init_bankroll=1000,
    min_bet=5,
    max_bet=100,
    start_date=pd.Timestamp("19" + str(year) + "-10-01"),
    end_date=pd.Timestamp("19" + str(year + 1) + "-10-01"),
)

evaluation = env.run()

print(f"year: 19{year}-19{year+1}")
print(f"Final bankroll: {env.bankroll:.2f}")


history = env.get_history()
print(history["Bankroll"].min())
plt.plot(history.index, history["Bankroll"])
plt.show()
