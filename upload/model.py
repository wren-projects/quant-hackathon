from __future__ import annotations

import math
from collections import namedtuple
from enum import IntEnum
from itertools import chain, product, repeat, starmap
from operator import add
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn import metrics, model_selection

if TYPE_CHECKING:
    import os

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
    defaults=(None,) * 32,
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


def match_to_opp(match: Match) -> Opp:
    """
    Convert Match to Opp.

    Fills Bets with 0.
    """
    return Opp(
        Index=match.Index,
        Season=match.Season,
        Date=match.Date,
        HID=match.HID,
        AID=match.AID,
        N=match.N,
        POFF=match.POFF,
        OddsH=match.OddsH,
        OddsA=match.OddsA,
        BetH=0,
        BetA=0,
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


class Team(IntEnum):
    """Enum discerning teams playing home or away."""

    Home = 0
    Away = 1


class CustomQueue:
    """Serve as custom version of queue."""

    def __init__(self, n: int) -> None:
        """Initialize queue."""
        self.size: int = n
        self.values: np.array = np.zeros((n, 1))
        self.__curent_oldest: int = 0

    def put(self, value: float) -> None:
        """Put new value in queue."""
        self.values[self.__curent_oldest % self.size] = value
        self.__curent_oldest += 1

    def get_q_avr(self) -> float:
        """Return average array of each feature."""
        if self.__curent_oldest == 0:
            return 0.0

        return np.sum(self.values) / min(self.size, self.__curent_oldest)


class TeamData:
    """Hold data of one team, both as home and away."""

    N_SHORT = 5
    N_LONG = 30

    BASE_COLUMNS: tuple[str, ...] = (
        "WR",
        "WRH",
        "WRA",
        "PSA",
        "PSAH",
        "PSAA",
        "PLTA",
        "PLTAH",
        "PLTAA",
        "PD",
        "PDH",
        "PDA",
    )

    TEAM_COLUMNS: tuple[str, ...] = (
        "DSLM",
        *starmap(add, product(BASE_COLUMNS, ["_S", "_L"])),
    )

    # HACK: Python's scopes are weird, so we have to work around them with the
    # extra repeat iterator
    COLUMNS: tuple[tuple[str, ...], ...] = tuple(
        tuple(starmap(add, product(team_prefix, tc)))
        for team_prefix, tc in zip([["H_"], ["A_"]], repeat(TEAM_COLUMNS))
    )

    MATCH_COLUMNS: tuple[str, ...] = tuple(chain.from_iterable(COLUMNS))

    def __init__(self, team_id: TeamID) -> None:
        """Init datastucture."""
        self.id: TeamID = team_id
        self.date_last_match: pd.Timestamp = pd.to_datetime("1977-11-10")

        # short averages
        self.win_rate_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.win_rate_home_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.win_rate_away_S: CustomQueue = CustomQueue(TeamData.N_SHORT)

        self.points_scored_average_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_scored_average_home_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_scored_average_away_S: CustomQueue = CustomQueue(TeamData.N_SHORT)

        self.points_lost_to_x_average_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_lost_to_x_average_home_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )
        self.points_lost_to_x_average_away_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )

        self.points_diference_average_S: CustomQueue = CustomQueue(TeamData.N_SHORT)
        self.points_diference_average_home_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )
        self.points_diference_average_away_S: CustomQueue = CustomQueue(
            TeamData.N_SHORT
        )

        # long averages
        self.win_rate_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.win_rate_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.win_rate_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

        self.points_scored_average_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_scored_average_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_scored_average_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

        self.points_lost_to_x_average_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_lost_to_x_average_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_lost_to_x_average_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

        self.points_diference_average_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_diference_average_home_L: CustomQueue = CustomQueue(TeamData.N_LONG)
        self.points_diference_average_away_L: CustomQueue = CustomQueue(TeamData.N_LONG)

    def _get_days_since_last_mach(self, today: pd.Timestamp) -> int:
        """Return number of days scince last mach."""
        return (today - self.date_last_match).days

    def update(self, match: Match, played_as: Team) -> None:
        """Update team data based on data from one mach."""
        self.date_last_match = pd.to_datetime(match.Date)

        win = match.H if played_as == Team.Home else match.A
        points = match.HSC if played_as == Team.Home else match.ASC
        points_lost_to = match.ASC if played_as == Team.Home else match.HSC
        point_diference = points - points_lost_to

        self.win_rate_S.put(win)
        self.win_rate_L.put(win)
        self.points_scored_average_S.put(points)
        self.points_scored_average_L.put(points)
        self.points_lost_to_x_average_S.put(points_lost_to)
        self.points_lost_to_x_average_L.put(points_lost_to)
        self.points_diference_average_S.put(point_diference)
        self.points_diference_average_L.put(point_diference)

        if played_as == Team.Home:
            self.win_rate_home_S.put(win)
            self.win_rate_home_L.put(win)
            self.points_scored_average_home_S.put(points)
            self.points_scored_average_home_L.put(points)
            self.points_lost_to_x_average_home_S.put(points_lost_to)
            self.points_lost_to_x_average_home_L.put(points_lost_to)
            self.points_diference_average_home_S.put(point_diference)
            self.points_diference_average_home_L.put(point_diference)
        else:
            self.win_rate_away_S.put(win)
            self.win_rate_away_L.put(win)
            self.points_scored_average_away_S.put(points)
            self.points_scored_average_away_L.put(points)
            self.points_lost_to_x_average_away_S.put(points_lost_to)
            self.points_lost_to_x_average_away_L.put(points_lost_to)
            self.points_diference_average_away_S.put(point_diference)
            self.points_diference_average_away_L.put(point_diference)

    def get_data_series(self, date: pd.Timestamp, team: Team) -> pd.Series:
        """Return complete data vector for given team."""
        return pd.Series(
            [
                self._get_days_since_last_mach(date),
                self.win_rate_S.get_q_avr(),
                self.win_rate_L.get_q_avr(),
                self.win_rate_home_S.get_q_avr(),
                self.win_rate_home_L.get_q_avr(),
                self.win_rate_away_S.get_q_avr(),
                self.win_rate_away_L.get_q_avr(),
                self.points_scored_average_S.get_q_avr(),
                self.points_scored_average_L.get_q_avr(),
                self.points_scored_average_home_S.get_q_avr(),
                self.points_scored_average_away_L.get_q_avr(),
                self.points_scored_average_home_L.get_q_avr(),
                self.points_scored_average_away_S.get_q_avr(),
                self.points_lost_to_x_average_S.get_q_avr(),
                self.points_lost_to_x_average_L.get_q_avr(),
                self.points_lost_to_x_average_home_S.get_q_avr(),
                self.points_lost_to_x_average_home_L.get_q_avr(),
                self.points_lost_to_x_average_away_S.get_q_avr(),
                self.points_lost_to_x_average_away_L.get_q_avr(),
                self.points_diference_average_S.get_q_avr(),
                self.points_diference_average_L.get_q_avr(),
                self.points_diference_average_home_S.get_q_avr(),
                self.points_diference_average_home_L.get_q_avr(),
                self.points_diference_average_away_S.get_q_avr(),
                self.points_diference_average_away_L.get_q_avr(),
            ],
            index=pd.Index(self.COLUMNS[team]),
        )


class Data:
    """Class for working with data."""

    def __init__(self) -> None:
        """Create Data from csv file."""
        self.teams: dict[TeamID, TeamData] = {}

    def add_match(self, match: Match) -> None:
        """Update team data based on data from one mach."""
        self.teams.setdefault(match.HID, TeamData(match.HID)).update(match, Team.Home)
        self.teams.setdefault(match.AID, TeamData(match.AID)).update(match, Team.Away)

    def team_data(self, team_id: TeamID) -> TeamData:
        """Return the TeamData for given team."""
        return self.teams[team_id]

    def get_match_parameters(self, match: Opp) -> pd.Series:
        """Get array for match."""
        home_team = self.teams.setdefault(match.HID, TeamData(match.HID))
        away_team = self.teams.setdefault(match.AID, TeamData(match.AID))

        date: pd.Timestamp = pd.to_datetime(match.Date)

        return pd.concat(
            [
                home_team.get_data_series(date, Team.Home),
                away_team.get_data_series(date, Team.Away),
            ]
        )


class Player:
    """Handles betting strateggy."""

    def get_expected_profit(
        self, probability: float, ratio: float, proportion: float
    ) -> float:
        """Calculate the expected profit for given parametrs."""
        return (probability * ratio - 1) * proportion

    def get_variance_of_profit(
        self, probability: float, ratio: float, proportion: float
    ) -> float:
        """Calculate the variance of profit for given parameters."""
        return (1 - probability) * probability * (proportion**2) * (ratio**2)

    def sharpe_ratio(self, total_profit: float, total_var: float) -> float:
        """Return total sharpe ratio."""
        return total_profit / math.sqrt(total_var) if total_var > 0 else float("inf")

    def min_function(
        self, proportions: np.ndarray, probabilities: np.ndarray, ratios: np.ndarray
    ) -> float:
        """We are trying to minimize this function for sharpe ratio."""
        total_profit = 0
        total_var = 0
        for i in range(len(probabilities)):
            for j in range(len(probabilities[i])):
                probability = probabilities[i][
                    j
                ]  # First column is for win, second column is for loss
                ratio = ratios[i][j]  # Use the ratio corresponding to the win scenario
                # Access flattened array index
                prop_of_budget = proportions[i * len(probabilities[i]) + j]
                total_profit += self.get_expected_profit(
                    probability, ratio, prop_of_budget
                )
                total_var += self.get_variance_of_profit(
                    probability, ratio, prop_of_budget
                )
        return -self.sharpe_ratio(total_profit, total_var)

    def get_bet_proportions(
        self,
        probabilities: np.ndarray,
        active_matches: pd.DataFrame,
        summary: Summary,
    ) -> np.ndarray:
        """Get bet proportind thru Sharp ratio. Probabilities: 2d array."""
        ratios = np.array(active_matches[["OddsH", "OddsA"]])
        initial_props = np.full_like(probabilities, 0.01, dtype=float)

        # Constraint: sum of all props <= 1
        # (global budget constraint for entire 2D array)
        cons = [
            {"type": "ineq", "fun": lambda props: 1.0 - sum(props)}
        ]  # Global budget constraint

        # Bounds: Each proportion must be between 0 and 1
        bounds = [
            (0, (summary.Max_bet / summary.Bankroll))
            for _ in range(probabilities.shape[0] * probabilities.shape[1])
        ]

        # Flatten the props for optimization and define the bounds
        initial_props_flat = initial_props.flatten()
        # Objective function minimization
        result = minimize(
            self.min_function,
            initial_props_flat,
            args=(probabilities, ratios),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-6},
        )
        return np.array(result.x).reshape(probabilities.shape)

    def get_betting_strategy(
        self,
        probabilities: pd.DataFrame,
        active_matches: pd.DataFrame,
        summary: Summary,
    ) -> np.ndarray:
        """Return absolute cash numbers and on what to bet in 2d list."""
        proportions: list[float] = (
            self.get_bet_proportions(probabilities.to_numpy(), active_matches, summary)
            * summary.Bankroll
        )
        return np.array(proportions).round(decimals=0)


class RankingModel(Protocol):
    """Ranking model interface."""

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the model."""
        raise NotImplementedError


class TeamElo:
    """
    Class for keeping track of a team's Elo.

    Attributes:
        opponents: Sum of Elo ratings of opponents
        games: Number of games
        wins: Number of wins
        rating: Current Elo rating

    """

    A: int = 400
    K: int = 40
    BASE: int = 160
    opponents: int
    games: int
    wins: int
    rating: float

    def __init__(self) -> None:
        """Initialize TeamElo."""
        self.games = 0
        self.wins = 0
        self.rating = 1000

    def adjust(self, opponent_elo: float, win: int) -> None:
        """
        Adjust Elo rating based on one match.

        Args:
            opponent_elo: Elo rating of the other team
            win: 1 for win, 0 for loss

        """
        self.games += 1
        self.wins += 1 if win else 0

        expected = self.predict(opponent_elo)

        self.rating += self.K * (win - expected)

    def predict(self, opponent_elo: float) -> float:
        """
        Predict outcome of a match with opponent of given Elo.

        Args:
            opponent_elo: Elo of the opponent

        Returns:
            Probability of winning (0..1)

        """
        d = opponent_elo - self.rating
        return 1 / (1 + self.BASE ** (d / self.A))

    def __str__(self) -> str:
        """Create a string representation of the team's Elo."""
        return (
            f"{self.rating:>4} ({self.games:>4}, "
            f"{self.wins:>4}, {self.wins / self.games * 100:>6.2f}%)"
        )


class Elo(RankingModel):
    """Class for the Elo ranking model."""

    teams: dict[int, TeamElo]

    def __init__(self) -> None:
        """Initialize Elo model."""
        self.teams = {}

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
        home_team = self.teams.setdefault(match.HID, TeamElo())
        away_team = self.teams.setdefault(match.AID, TeamElo())

        home_elo = home_team.rating
        away_elo = away_team.rating

        home_team.adjust(away_elo, match.H)
        away_team.adjust(home_elo, match.A)

    def rankings(self) -> dict[int, float]:
        """Return normalized rankings."""
        max_elo = max(elo.rating for elo in self.teams.values())
        return {team: teamElo.rating / max_elo for team, teamElo in self.teams.items()}

    def team_rating(self, team_id: int) -> float:
        """Return Elo rating of a team."""
        return self.teams.setdefault(team_id, TeamElo()).rating

    def reset(self) -> None:
        """Reset the model."""
        self.teams = {}


class EloByLocation(RankingModel):
    """Class for the Elo ranking model."""

    teams_home: dict[int, TeamElo]
    teams_away: dict[int, TeamElo]

    def __init__(self) -> None:
        """Initialize Elo model."""
        self.teams_home = {}
        self.teams_away = {}

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        home_elo = self.teams_home.setdefault(match.HID, TeamElo())
        away_elo = self.teams_away.setdefault(match.AID, TeamElo())

        # memorize elo values before they change
        home_elo_value = home_elo.rating
        away_elo_value = away_elo.rating

        home_elo.adjust(away_elo_value, match.H)
        away_elo.adjust(home_elo_value, match.A)

    def predict(self, match: Opp) -> float | None:
        """
        Predicts how the match might go.

        Float from 0 to 1 = chance of H to win
        None means no data
        """
        home_elo = self.teams_home.setdefault(match.HID, TeamElo())
        away_elo = self.teams_away.setdefault(match.AID, TeamElo())

        played_enough = home_elo.games >= 10 and away_elo.games >= 10
        return 100 * home_elo.predict(away_elo.rating) if played_enough else None

    def reset(self) -> None:
        """Reset the model."""
        self.teams_home.clear()
        self.teams_away.clear()


class Model:
    """Main class."""

    TRAIN_SIZE: int = 4000
    FIRST_TRAIN_MOD: int = 1

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()
        self.elo_by_location = EloByLocation()
        self.player = Player()
        self.ai = Ai()
        self.trained = False
        self.data = Data()
        self.season_number: int = 0
        self.budget: int = 0
        self.old_matches: pd.DataFrame = pd.DataFrame()
        self.old_outcomes: pd.Series = pd.Series()
        self.last_retrain = 0

    def update_models(self, games_increment: pd.DataFrame) -> None:
        """Update models."""
        for match in (Match(*row) for row in games_increment.itertuples()):
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)
            self.data.add_match(match)

    def place_bets(
        self,
        summ: pd.DataFrame,
        opps: pd.DataFrame,
        inc: tuple[pd.DataFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """Run main function."""
        games_increment = inc[0]
        summary = Summary(*summ.iloc[0])

        if not self.trained:
            train_size = self.TRAIN_SIZE * self.FIRST_TRAIN_MOD
            print(
                f"Initial training on {games_increment[-train_size :].shape[0]}"
                f" matches with bankroll {summary.Bankroll}"
            )
            self.train_ai_reg(cast(pd.DataFrame, games_increment[-train_size:]))
        elif games_increment.shape[0] > 0:
            increment_season = int(games_increment.iloc[0]["Season"])
            # if self.season_number != increment_season:
            #    self.elo.reset()
            #    self.elo_by_location.reset()
            #    self.season_number = increment_season

            self.old_matches = pd.concat(
                [
                    self.old_matches.iloc[-self.TRAIN_SIZE :],
                    self.create_dataframe(games_increment),
                ],
            )

            self.old_outcomes = cast(
                pd.Series,
                pd.concat(
                    [
                        self.old_outcomes.iloc[-self.TRAIN_SIZE :],
                        games_increment.HSC - games_increment.ASC,
                    ],
                ),
            )

            month = pd.to_datetime(summary.Date).month
            if self.last_retrain != month:
                print(
                    f"{summary.Date}: retraining on {self.old_matches.shape[0]}"
                    f" matches with bankroll {summary.Bankroll}"
                )
                self.ai.train_reg(self.old_matches, self.old_outcomes)
                self.last_retrain = month
                self.budget = summary.Bankroll

            self.update_models(games_increment)

        active_matches = cast(pd.DataFrame, opps[opps["Date"] == summary.Date])

        if active_matches.shape[0] == 0 or summary.Bankroll < (self.budget * 0.9):
            return pd.DataFrame(
                data=0,
                index=np.arange(active_matches.shape[0]),
                columns=pd.Index(["BetH", "BetA"], dtype="str"),
            )

        dataframe = self.create_dataframe(active_matches)
        probabilities = self.ai.get_probabilities_reg(dataframe)
        bets = self.player.get_betting_strategy(probabilities, active_matches, summary)

        new_bets = pd.DataFrame(
            data=bets,
            columns=pd.Index(["BetH", "BetA"], dtype="str"),
            index=active_matches.index,
        )

        return new_bets.reindex(opps.index, fill_value=0)

    RANKING_COLUMNS: tuple[str, ...] = (
        "HomeElo",
        "AwayElo",
        "EloByLocation",
    )
    MATCH_PARAMETERS = len(TeamData.COLUMNS) + len(RANKING_COLUMNS)
    TRAINING_DATA_COLUMNS: tuple[str, ...] = (*RANKING_COLUMNS, *TeamData.MATCH_COLUMNS)

    def create_dataframe(self, active_matches: pd.DataFrame) -> pd.DataFrame:
        """Get matches to predict outcome for."""
        return cast(
            pd.DataFrame,
            active_matches.apply(
                lambda x: self.get_match_parameters(match_to_opp(Match(0, *x))),
                axis=1,
            ),
        )

    def get_match_parameters(self, match: Opp) -> pd.Series:
        """Get parameters for given match."""
        home_elo = self.elo.team_rating(match.HID)
        away_elo = self.elo.team_rating(match.AID)
        elo_by_location_prediction = self.elo_by_location.predict(match)

        rankings = pd.Series(
            [
                home_elo,
                away_elo,
                elo_by_location_prediction,
            ],
            index=self.RANKING_COLUMNS,
        )

        data_parameters = self.data.get_match_parameters(match)

        return pd.concat([rankings, data_parameters], axis=0)

    def train_ai(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        training_data = []
        outcomes_list = []

        for match in (Match(*x) for x in dataframe.itertuples()):
            match_parameters = self.get_match_parameters(match_to_opp(match))

            training_data.append(match_parameters)
            outcomes_list.append(match.H)

            self.data.add_match(match)
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)

        training_dataframe = pd.DataFrame(
            training_data, columns=pd.Index(self.TRAINING_DATA_COLUMNS)
        )

        outcomes = pd.Series(outcomes_list)

        self.old_matches = training_dataframe
        self.old_outcomes = outcomes

        self.ai.train(training_dataframe, outcomes)
        self.trained = True

    def train_ai_reg(self, dataframe: pd.DataFrame) -> None:
        """Train AI."""
        training_data = []
        outcomes_list = []

        for match in (Match(*x) for x in dataframe.itertuples()):
            match_parameters = self.get_match_parameters(match_to_opp(match))

            training_data.append(match_parameters)
            outcomes_list.append(match.HSC - match.ASC)

            self.data.add_match(match)
            self.elo.add_match(match)
            self.elo_by_location.add_match(match)

        training_dataframe = pd.DataFrame(
            training_data, columns=pd.Index(self.TRAINING_DATA_COLUMNS)
        )

        outcomes = pd.Series(outcomes_list)

        self.old_matches = training_dataframe
        self.old_outcomes = outcomes

        self.ai.train_reg(training_dataframe, outcomes)
        self.trained = True


def calculate_elo_accuracy(data: list[list[int]]) -> float:
    """Calculate the accuracy of ELO predictions."""
    correct_predictions = 0
    total_games = len(data)
    games = np.array(data)[:, :-1]
    outcomes = np.array(data)[:, -1].clip(0, 1).round(decimals=0)
    for i in range(len(data)):
        elo_home = games[i][0]
        elo_away = games[i][1]
        outcome = outcomes[i]

        # Predict home win if home ELO is greater than away ELO
        predicted_outcome = 1 if elo_home > elo_away else 0

        # Compare predicted outcome with actual outcome
        if predicted_outcome == outcome:
            correct_predictions += 1

    # Calculate accuracy as a percentage
    return correct_predictions / total_games


class Ai:
    """Class for training and predicting."""

    model: xgb.XGBRegressor | xgb.XGBClassifier

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
        self.initialized = False

    def train(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""
        if not self.initialized:
            self.model = xgb.XGBClassifier()
            self.initialized = True

        self.model = self.model.fit(training_dataframe, outcomes)

    def train_reg(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror", max_depth=10, n_estimators=1000
        )
        self.initialized = True
        print(training_dataframe.columns)

        x_train, x_val, y_train, y_val = model_selection.train_test_split(
            training_dataframe.to_numpy(),
            outcomes.to_numpy(),
            test_size=0.01,
            random_state=2,
            shuffle=True,
        )
        print(x_train.shape)
        self.model.fit(x_train, y_train)
        print("MAE:", metrics.mean_absolute_error(y_val, self.model.predict(x_val)))
        print(self.model.feature_importances_)

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        return self.model.predict_proba(dataframe.to_numpy())

    def get_probabilities_reg(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        predicted_score_differences = self.model.predict(dataframe)
        return self.calculate_probabilities(predicted_score_differences)

    def save_model(self, path: os.PathLike) -> None:
        """Save ML model."""
        self.model.save_model(path)

    def home_team_win_probability(self, score_difference: float) -> float:
        """Calculate the probability of home team winning based on score difference."""
        slope = 0.8  # range optimal 0.1 to 1. liked 0.3 and 0.5 (maybe 1)
        return 1 / (1 + np.exp(-slope * score_difference))

    def calculate_probabilities(self, score_differences: np.ndarray) -> pd.DataFrame:
        """Calculate the probabilities of teams winning based on score differences."""
        probabilities = []

        for score_difference in score_differences:
            home_prob = self.home_team_win_probability(score_difference)
            away_prob = 1 - home_prob
            probabilities.append((home_prob, away_prob))

        return pd.DataFrame(probabilities, columns=pd.Index(["WinHome", "WinAway"]))
