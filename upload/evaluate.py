import sys

import pandas as pd

sys.path.append(".")

from environment import Environment
from model import Model

games = pd.read_csv("./data/games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv("./data/players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])

env = Environment(
    games,
    players,
    Model(),
    init_bankroll=1000,
    min_bet=5,
    max_bet=100,
    start_date=pd.Timestamp("1985-11-11"),
)

evaluation = env.run()

print()
print(f"Final bankroll: {env.bankroll:.2f}")

history = env.get_history()
