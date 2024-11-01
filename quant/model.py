import os

import numpy as np

from quant.bet import Player
from quant.data import Data
from quant.predict import Ai
import pandas as pd

class Model:
    def __init__(self) -> None:
        self.seen_matches = set()
        self.player = Player(1000)
        self.data = Data()
        self.ai = Ai(True, "model.json", self.data)

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame])-> pd.DataFrame:
        self.data.add_new_matches_outcome(inc)
        new_matches = self.get_new_matches(opps)
        probabbilities = self.ai.get_probabilities(new_matches)
        
        self.player.update_budget(summary.iloc[0]["Bankroll"])
        min_bet = summary.iloc[0]["Min_bet"]
        max_bet = summary.iloc[0]["Max_bet"]
        
        bets = self.player.get_betting_strategy(probabbilities, opps)
        
        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
        return bets
    
    def get_new_matches(self, opps: pd.DataFrame) -> pd.DataFrame:
        new_opps = opps[~opps["ID"]].isin(self.seen_matches)
        self.seen_matches.update(opps["ID"])
        return new_opps
        
        