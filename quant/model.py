from __future__ import annotations

import numpy as np
import pandas as pd

from quant.bet import Player
from quant.data import Data
from quant.models.Elo import Elo
from quant.predict import Ai
from quant.types import Match


class Model:
    """Main class."""

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()
        self.player = Player()
        self.data = Data()
        self.ai = Ai.untrained()
        self.trained = False

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples(index=False)):
            self.elo.add_match(match)

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        games_increment = inc[0]

        if not self.trained:
            self.train_ai(games_increment)
            self.trained = True
        else:
            self.update_models(games_increment)

        upcoming_games: pd.DataFrame = opps[opps["Date"] == summary.iloc[0]["Date"]]

        data_matrix = self.create_data_matrix(upcoming_games)

        probabilities = self.ai.get_probabilities(data_matrix)
        bets = self.player.get_betting_strategy(probabilities, opps, summary)

        new_bets = pd.DataFrame(
            data=bets,
            columns=pd.Index(["BetH", "BetA"], dtype="str"),
            index=upcoming_games.index,
        )
        return new_bets.reindex(opps.index, fill_value=0)

    def create_data_matrix(self, upcoming_games: pd.DataFrame) -> np.ndarray:
        """Get matches to predict outcome for."""
        data_matrix = np.ndarray([upcoming_games.shape[0], 4])
        for match in (Match(*row) for row in upcoming_games.itertuples(index=False)):
            home_elo = self.elo.teams[match.HID].rating
            away_elo = self.elo.teams[match.AID].rating

            data_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
            ]

        return data_matrix

    def train_ai(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        train_matrix = np.ndarray([dataframe.shape[0], 5])

        for match in (Match(*x) for x in dataframe.itertuples()):
            self.elo.add_match(match)

            home_elo = self.elo.teams[match.HID].rating
            away_elo = self.elo.teams[match.AID].rating

            train_matrix[match.Index] = [
                home_elo,
                away_elo,
                match.OddsH,
                match.OddsA,
                match.H,
            ]

        self.ai.train(train_matrix)
