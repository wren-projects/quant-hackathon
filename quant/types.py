from __future__ import annotations

from collections import namedtuple
from typing import Protocol

Team = int

Match = namedtuple(
    "Match",
    [
        "Index",
        "GameID",
        "Season",
        "Date",
        "HID",
        "AID",
        "N",
        "POFF",
        "Open",
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


class RankingModel(Protocol):
    """Ranking model interface."""

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        raise NotImplementedError

    def rankings(self) -> dict[Team, float]:
        """Return normalized rankings."""
        raise NotImplementedError
