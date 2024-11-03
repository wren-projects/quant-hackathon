import os

import numpy as np
import pandas as pd

from quant.bet import Player
from quant.data import Data
from quant.predict import Ai


class Model:
    """Main class."""

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.player = Player()
        self.data = Data()
        self.ai = Ai(True, "model.json", self.data)

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        self.data.add_new_matches_outcome(inc)
        new_matches = self.get_new_matches(opps, summary)
        probabbilities = self.ai.get_probabilities(new_matches)
        bets = self.player.get_betting_strategy(probabbilities, opps, summary)
        new_bets = pd.DataFrame(
            data=bets, columns=["BetH", "BetA"], index=new_matches.index
        )
        return new_bets.reindex(opps.index, fill_value=0)

    def get_new_matches(
        self, opps: pd.DataFrame, summary: pd.DataFrame
    ) -> pd.DataFrame:
        """Get matches to predict outcome for."""
        """
        new_opps = opps[~opps["ID"]].isin(self.seen_matches)
        self.seen_matches.update(opps["ID"])
        """
        return opps[opps["Date"] == summary.iloc[0]["Date"]]
