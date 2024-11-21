import numpy as np


class PageRank:
    """Class for the page rank model."""

    def __init__(
        self, alpha: float = 0.85, tol: float = 1e-6, max_iter: int = 100
    ) -> None:
        """
        Initialize the BasketballPageRank class.

        Args:
        - alpha: Damping factor for PageRank.
        - tol: Convergence tolerance.
        - max_iter: Maximum number of iterations.

        """
        self.alpha: float = alpha
        self.tol: float = tol
        self.max_iter: int = max_iter
        self.teams: dict = {}
        self.games: list = []
        self.last_season = None

    def add_match(self, match) -> None:
        """
        Add a match to the dataset.

        Args:
        - match: line with data

        """
        season = match["Season"]
        if self.last_season != season:
            self.last_season = season
            self.games = []
        home_id = match["HID"]
        away_id = match["AID"]
        result = match["H"]
        if home_id not in self.teams:
            self.teams[home_id] = len(self.teams)
        if away_id not in self.teams:
            self.teams[away_id] = len(self.teams)

        if result:
            self.games.append((away_id, home_id))
        else:
            self.games.append((home_id, away_id))

    def _calculate_ratings(self) -> dict:
        """
        Calculate the PageRank ratings for all teams.

        Returns:
        - ranks: Dictionary mapping team IDs to their PageRank scores.
        Here is where the magic happen.

        """
        n = len(self.teams)
        if n == 0:
            return {}

        # team_index = {team: idx for team, idx in self.teams.items()}
        team_index = dict(self.teams.items())

        # Build adjacency matrix
        m = np.zeros((n, n))
        for loser, winner in self.games:
            m[team_index[winner], team_index[loser]] += 1

        # Normalize the matrix
        for i in range(n):
            if m[i].sum() > 0:
                m[i] /= m[i].sum()
            else:
                m[i] = 1 / n  # Handle dangling nodes

        # PageRank algorithm
        rank = np.ones(n) / n
        for _ in range(self.max_iter):
            new_rank = self.alpha * m.T @ rank + (1 - self.alpha) / n
            if np.linalg.norm(new_rank - rank, ord=1) < self.tol:
                break
            rank = new_rank

        # Map scores to teams
        return {team: rank[team_index[team]] for team in self.teams}

    def team_rating(self, match) -> tuple:
        """
        Get the rating of one or two teams.

        Args:
        - team_id1: ID of the first team.
        - team_id2: ID of the second team.

        Returns:
        - vecstor of ratings

        """
        home_id = match["HID"]
        away_id = match["AID"]
        ratings = self._calculate_ratings()
        return (ratings.get(home_id, None), ratings.get(away_id, None))

    def predict(self, match) -> float:
        """
        Get the ratio of winning two teams.

        Args:
        - team_id1: ID of the first team.
        - team_id2: ID of the second team.

        Returns:
        - ratio of winning(home 0, away 100)

        """
        home_id = match["HID"]
        away_id = match["AID"]
        ratings = self._calculate_ratings()
        if home_id not in ratings or away_id not in ratings:
            return 0.5
        ratio: float = ratings.get(away_id, None) / (
            ratings.get(home_id, None) + ratings.get(away_id, None)
        )
        return ratio

    def ready(self):
        return len(self.games) >= 60

    def should_bet(self, match):
        PRED = self.predict(match)
        return self.ready() and PRED >= 0.16 and PRED * match["OddsH"] > 1.45
