from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Protocol, TypeAlias

import numpy as np
import pandas as pd
import sys
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


class EloTeam:
    """
    Class for keeping track of a team's Elo.

    Attributes:
        opponents: Sum of Elo ratings of opponents
        games: Number of games
        wins: Number of wins
        rating: Current Elo rating

    """

    K: int = 40
    base: int = 160
    opponents: int
    games: int
    wins: int
    rating: float

    def __init__(self) -> None:
        """Initialize TeamElo."""
        self.games = 0
        self.wins = 0
        self.rating = 1000

    def adjust(self, opponentElo: float, win: int) -> None:
        """
        Adjust Elo rating based on one match.

        Args:
            opponent: Elo rating of the other team
            win: 1 for win, 0 for loss

        """
        self.games += 1
        self.wins += 1 if win else 0
        
        expected = self.predict(opponentElo)

        self.rating += self.K * (win - expected)

    def predict(self, opponentElo: float):
        d = opponentElo - self.rating

        A = self.base**(d/400)
        expected = 1/(1+A)
        return expected
    

    def __str__(self) -> str:
        """Create a string representation of the team's Elo."""
        return (
            f"{self.rating:>4} ({self.games:>4}, "
            f"{self.wins:>4}, {self.wins / self.games * 100:>6.2f}%)"
        )


class Elo:
    """Class for the Elo ranking model."""

    elo_home_by_season: dict[int,dict[int, EloTeam]]
    elo_away_by_season: dict[int,dict[int, EloTeam]]
    def __init__(self) -> None:
        """Initialize Elo model."""
        self.elo_home_by_season = {}
        self.elo_away_by_season = {}

    def __str__(self) -> str:
        """Create a string representation of the model."""
        return "Team  Elo Opponents Games  Wins  WinRate\n" + "\n".join(
            f" {team:>2}: {elo}"
            for team, elo in sorted(
                self.teams.items(), key=lambda item: -item[1].rating
            )
        )

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        season = match.Season
        if season not in self.elo_home_by_season:
            self.elo_home_by_season[season] = {}
            self.elo_away_by_season[season] = {}
        elo_home = self.elo_home_by_season[season]
        elo_away = self.elo_away_by_season[season]


        if match.HID not in elo_home:
            elo_home[match.HID] = EloTeam()

        if match.AID not in elo_away:
            elo_away[match.AID] = EloTeam()

        home_elo = elo_home[match.HID]
        away_elo = elo_away[match.AID]

        # memorize elo values before they change
        home_elo_value = home_elo.rating
        away_elo_value = away_elo.rating

        home_elo.adjust(away_elo_value, match.H)
        away_elo.adjust(home_elo_value, match.A)
        
    def predict(self, match: Opp) -> float|None:
        """
        Predicts how the match might go.
        Float from 0 to 1 = chance of H to win
        None means no data
        """
        season = match.Season
        if season not in self.elo_home_by_season:
            self.elo_home_by_season[season] = {}
            self.elo_away_by_season[season] = {}

        elo_home = self.elo_home_by_season[season]
        elo_away = self.elo_away_by_season[season]

        if match.HID not in elo_home:
            elo_home[match.HID] = EloTeam()

        if match.AID not in elo_away:
            elo_away[match.AID] = EloTeam()

        home_elo = elo_home[match.HID]
        away_elo = elo_away[match.AID]
        if home_elo.games < 10 or away_elo.games < 10:
            return None
        return home_elo.predict(away_elo.rating)

        

class Model:
    """Main class."""

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples()):
            self.elo.add_match(match)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opportunities: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        #print("new round")
        self.update_models(inc[0])


        summary = Summary(*summ.iloc[0])
        maxBet = summary.Max_bet
        bankroll = summary.Bankroll
        if bankroll < summary.Min_bet or maxBet == 0 or len(opportunities) == 0:
            return pd.DataFrame()
        opportunities["ELO"] = opportunities.apply(self.elo.predict,axis=1)
        opportunities["EV"] = opportunities["ELO"]*opportunities["OddsH"]
        toBetOn = opportunities[opportunities["ELO"] is not None and opportunities["EV"] >= 2]
        toBetOn = toBetOn.sort_values(by=["EV"], ascending=False)

        toBetOn["BetH"] = maxBet

        # not enough money
        if toBetOn["BetH"].sum() > bankroll:
            rows_to_discard = int((toBetOn["BetH"].sum()-bankroll)//maxBet)
            if rows_to_discard > 0:
                toBetOn = toBetOn[:-rows_to_discard]
            toBetOn.loc[toBetOn.index[-1],"BetH"] -= (toBetOn["BetH"].sum()-bankroll)%maxBet


        toBetOn["BetA"] = 0
        return toBetOn