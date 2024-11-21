class Elo:
    """Class for the page rank model."""

    def __init__(self, K=80, base=160) -> None:
        """
        Initialize the BasketballPageRank class.

        Args:
        - alpha: Damping factor for PageRank.
        - tol: Convergence tolerance.
        - max_iter: Maximum number of iterations.

        """
        self.base = base
        self.K = K
        self.last_season = None
        self.home_rating_database = {}
        self.away_rating_database = {}
        self.games = 0
        # print(K)

    def _get_odds(self, elo_home, elo_away):
        d = elo_away - elo_home
        # d = max(min(d,800), -800)
        a = self.base ** ((d) / 400)
        return 1 / (1 + a)

    def add_match(self, match) -> None:
        """
        Add a match to the dataset.

        Args:
        - match: line with data

        """
        season = match["Season"]
        if self.last_season != season:
            self.last_season = season
            self.home_rating_database = {}
            self.away_rating_database = {}
            self.games = 0
        self.games += 1

        home_id = match["HID"]
        away_id = match["AID"]
        result = match["H"]

        rating_home = self.home_rating_database.get(home_id, 1000)
        rating_away = self.away_rating_database.get(away_id, 1000)

        P = self._get_odds(rating_home, rating_away)
        rating_home += self.K * (result - P)
        # 1 - result - 1 + p = p - result
        rating_away += self.K * (P - result)

        self.home_rating_database[home_id] = rating_home
        self.away_rating_database[away_id] = rating_away

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

        rating_home = self.home_rating_database.get(home_id, 1000)
        rating_away = self.away_rating_database.get(away_id, 1000)

        P = self._get_odds(rating_home, rating_away)
        return P

    def ready(self):
        return self.games >= 60

    def should_bet(self, match):
        PRED = self.predict(match)
        return self.ready() and PRED >= 0 and PRED * match["OddsH"] > 1.95
