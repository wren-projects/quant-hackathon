from __future__ import annotations

import sys
from collections import namedtuple
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import pandas as pd
from predictionModels.copiedopenskill import OpenSkillAPI
from predictionModels.elo import Elo
from predictionModels.pagerank import PageRank

pd.options.mode.chained_assignment = None

if TYPE_CHECKING:
    import os

TeamID: TypeAlias = int

Match = namedtuple(
    "Match",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "H",
        "A",
        "HSC",
        "ASC",
        "HFGM",
        "AFGM",
        "HFGA",
        "AFGA",
        "HFG3M",
        "AFG3M",
        "HFG3A",
        "AFG3A",
        "HFTM",
        "AFTM",
        "HFTA",
        "AFTA",
        "HORB",
        "AORB",
        "HDRB",
        "ADRB",
        "HRB",
        "ARB",
        "HAST",
        "AAST",
        "HSTL",
        "ASTL",
        "HBLK",
        "ABLK",
        "HTOV",
        "ATOV",
        "HPF",
        "APF",
    ],
)

Opp = namedtuple(
    "Opp",
    [
        "Index",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "OddsH",
        "OddsA",
        "BetH",
        "BetA",
    ],
)

Summary = namedtuple(
    "Summary",
    [
        "Bankroll",
        "Date",
        "Min_bet",
        "Max_bet",
    ],
)


class Model:
    """Main class."""

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.open_skill = OpenSkillAPI()
        # self.page_rank = PageRank()
        self.elo = Elo()

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for i, row in games_increment.iterrows():
            self.open_skill.add_match(row)
            self.elo.add_match(row)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opportunities: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        self.update_models(inc[0])

        summary = Summary(*summ.iloc[0])
        max_bet = summary.Max_bet
        min_bet = summary.Min_bet

        bankroll = summary.Bankroll
        if bankroll < summary.Min_bet or max_bet == 0 or len(opportunities) == 0:
            return pd.DataFrame()

        # bankroll /= 8
        # bankroll = max(min_bet, bankroll / 100)
        opportunities["BET_AWAY"] = opportunities.apply(
            self.open_skill.should_bet_away, axis=1
        )

        opportunities["BET_HOME"] = opportunities.apply(self.elo.should_bet, axis=1)

        assert opportunities[
            np.logical_and(opportunities["BET_AWAY"], opportunities["BET_HOME"])
        ].empty

        bet_on_away = opportunities[opportunities["BET_AWAY"]]
        bet_on_home = opportunities[opportunities["BET_HOME"]]

        if bet_on_home.shape[0] + bet_on_away.shape[0] == 0:
            return pd.DataFrame()

        # bankroll /= 8
        bet_size = min(
            max_bet,
            max(bankroll / (bet_on_home.shape[0] + bet_on_away.shape[0]), min_bet),
        )
        # bet_size = min_bet
        # assert bet_on_home["N"].sum() == 0
        # assert bet_on_home["POFF"].sum() == 0
        # assert bet_on_away["N"].sum() == 0
        # assert bet_on_away["POFF"].sum() == 0
        bet_on_home["BetH"] = bet_size
        bet_on_away["BetA"] = bet_size

        bets = pd.DataFrame()
        bets["BetH"] = bet_on_home["BetH"]
        bets["BetA"] = bet_on_away["BetA"]
        bets = bets.fillna(0)
        return bets
