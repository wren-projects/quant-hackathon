from __future__ import annotations

import copy
import itertools
import math
import sys
from collections import namedtuple
from enum import IntEnum
from itertools import chain, product, repeat, starmap, zip_longest
from operator import add
from statistics import NormalDist
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    cast,
)

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn import metrics, model_selection, neural_network

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

__all__: List[str] = ["PlackettLuce", "PlackettLuceRating"]


class OpenSkillAPI:
    def __init__(self):
        self.rating_database = {}
        self.model = PlackettLuce()

    def _get_players(
        self, home_id: any, away_id: any
    ) -> tuple[list[PlackettLuceRating], list[PlackettLuceRating]]:
        home_rating = self.rating_database.get(
            home_id, (self.model.mu, self.model.sigma)
        )
        away_rating = self.rating_database.get(
            away_id, (self.model.mu, self.model.sigma)
        )

        home_player = [self.model.rating(home_rating[0], home_rating[1])]
        away_player = [self.model.rating(away_rating[0], away_rating[1])]
        return home_player, away_player

    def add_match(self, match: Match) -> None:
        home_id = match.HID
        away_id = match.AID
        home_score = match.H
        home_player, away_player = self._get_players(home_id, away_id)

        if home_score:
            [[new_home_player], [new_away_player]] = self.model.rate(
                [home_player, away_player]
            )
        else:
            [[new_away_player], [new_home_player]] = self.model.rate(
                [away_player, home_player]
            )

        self.rating_database[home_id] = (new_home_player.mu, new_home_player.sigma)
        self.rating_database[away_id] = (new_away_player.mu, new_away_player.sigma)

    def predict(self, match: Match) -> float:
        """Predict the result of the given match and return the change home wins (on a scale 0-1)."""
        home_id = match.HID
        away_id = match.AID
        home_player, away_player = self._get_players(home_id, away_id)
        return self.model.predict_win([home_player, away_player])[0]


"""
Common functions for all models.
"""

from typing import Any, List


def _unary_minus(number: float) -> float:
    """
    Takes value of a number and makes it negative.

    :param number: A number to convert.
    :return: Converted number.
    """
    return -number


def _matrix_transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    """
    Transpose a matrix.

    :param matrix: A matrix in the form of a list of lists.
    :return: A transposed matrix.
    """
    return [list(row) for row in zip(*matrix)]


def _normalize(
    vector: List[float], target_minimum: float, target_maximum: float
) -> List[float]:
    """
    Normalizes a vector to a target range of values.

    :param vector: A vector to normalize.
    :param target_minimum: Minimum value to scale the values between.
    :param target_maximum: Maximum value to scale the values between.
    :return: Normalized vector.
    """
    if len(vector) == 1:
        return [target_maximum]

    source_minimum = min(vector)
    source_maximum = max(vector)
    source_range = source_maximum - source_minimum
    target_range = target_maximum - target_minimum

    if source_range == 0:
        source_range = 0.0001

    scaled_vector = [
        ((((value - source_minimum) / source_range) * target_range) + target_minimum)
        for value in vector
    ]

    return scaled_vector


"""
Common functions for the Weng-Lin models.
"""


_normal = NormalDist()


def _unwind(tenet: List[float], objects: List[Any]) -> Tuple[List[Any], List[float]]:
    """
    Retain the stochastic tenet of a sort to revert original sort order.

    :param tenet: A list of tenets for each object in the list.

    :param objects: A list of teams to sort.
    :return: Ordered objects and their tenets.
    """

    def _pick_zeroth_index(item: Tuple[float, Any]) -> float:
        """
        Returns the first item in a list.

        :param item: A list of objects.
        :return: The first item in the list.
        """
        return item[0]

    def _sorter(
        objects_to_sort: List[Any],
    ) -> Tuple[List[Any], List[float]]:
        """
        Sorts a list of objects based on a tenet.

        :param objects_to_sort: A list of objects to sort.
        :return: A tuple of the sorted objects and their tenets.
        """
        matrix = [[tenet[i], [x, i]] for i, x in enumerate(objects_to_sort)]
        unsorted_matrix = _matrix_transpose(matrix)
        if unsorted_matrix:
            zipped_matrix = list(zip(unsorted_matrix[0], unsorted_matrix[1]))
            zipped_matrix.sort(key=_pick_zeroth_index)
            sorted_matrix = [x for _, x in zipped_matrix]
            return [x for x, _ in sorted_matrix], [x for _, x in sorted_matrix]
        else:
            return [], []

    return _sorter(objects) if isinstance(objects, list) else _sorter


def phi_major(x: float) -> float:
    """
    Normal cumulative distribution function.

    :param x: A number.
    :return: A number.
    """
    return _normal.cdf(x)


def phi_major_inverse(x: float) -> float:
    """
    Normal inverse cumulative distribution function.

    :param x: A number.
    :return: A number.
    """
    return _normal.inv_cdf(x)


def phi_minor(x: float) -> float:
    """
    Normal probability density function.

    :param x: A number.
    :return: A number.
    """
    return _normal.pdf(x)


def v(x: float, t: float) -> float:
    """
    The function :math:`V` as defined in :cite:t:`JMLR:v12:weng11a`

    :param x: A number.
    :param t: A number.
    :return: A number.
    """
    xt = x - t
    denominator = phi_major(xt)
    return (
        -xt if (denominator < sys.float_info.epsilon) else phi_minor(xt) / denominator
    )


def w(x: float, t: float) -> float:
    """
    The function :math:`W` as defined in :cite:t:`JMLR:v12:weng11a`

    :param x: A number.
    :param t: A number.
    :return: A number.
    """
    xt = x - t
    denominator = phi_major(xt)
    if denominator < sys.float_info.epsilon:
        return 1 if (x < 0) else 0
    return v(x, t) * (v(x, t) + xt)


def vt(x: float, t: float) -> float:
    r"""
    The function :math:`\tilde{V}` as defined in :cite:t:`JMLR:v12:weng11a`

    :param x: A number.
    :param t: A number.
    :return: A number.
    """
    xx = abs(x)
    b = phi_major(t - xx) - phi_major(-t - xx)
    if b < 1e-5:
        if x < 0:
            return -x - t
        return -x + t
    a = phi_minor(-t - xx) - phi_minor(t - xx)
    return (-a if x < 0 else a) / b


def wt(x: float, t: float) -> float:
    r"""
    The function :math:`\tilde{W}` as defined in :cite:t:`JMLR:v12:weng11a`

    :param x: A number.
    :param t: A number.
    :return: A number.
    """
    xx = abs(x)
    b = phi_major(t - xx) - phi_major(-t - xx)
    if b < sys.float_info.epsilon:
        return 1.0
    return ((t - xx) * phi_minor(t - xx) + (t + xx) * phi_minor(-t - xx)) / b + vt(
        x, t
    ) * vt(x, t)


def _ladder_pairs(teams: List[Any]) -> List[List[Any]]:
    """
    Returns a list of pairs of ranks that are adjacent in the ladder.

    :param teams: A list of teams.
    :return: A list of pairs of teams that are adjacent in the ladder.
    """
    left: List[Any] = [None]
    left.extend(teams[:-1])
    right: List[Any] = list(teams[1:])
    right.append(None)
    zipped_lr = zip_longest(left, right)
    result = []
    for _left, _right in zipped_lr:
        if _left and _right:
            result.append([_left, _right])
        elif _left and not _right:
            result.append([_left])
        elif not _left and _right:
            result.append([_right])
        else:
            result.append([])
    return result


class PlackettLuceRating:
    """
    Plackett-Luce player rating data.

    This object is returned by the :code:`PlackettLuce.rating` method.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        name: Optional[str] = None,
    ):
        r"""
        :param mu: Represents the initial belief about the skill of
                   a player before any matches have been played. Known
                   mostly as the mean of the Guassian prior distribution.

                   *Represented by:* :math:`\mu`

        :param sigma: Standard deviation of the prior distribution of player.

                      *Represented by:* :math:`\sigma = \frac{\mu}{z}`
                      where :math:`z` is an integer that represents the
                      variance of the skill of a player.

        :param name: Optional name for the player.
        """

        # Player Information
        self.id = 0
        self.name: Optional[str] = name

        self.mu: float = mu
        self.sigma: float = sigma

    def __repr__(self) -> str:
        return f"PlackettLuceRating(mu={self.mu}, sigma={self.sigma})"

    def __str__(self) -> str:
        if self.name:
            return (
                f"Plackett-Luce Player Data: \n\n"
                f"id: {self.id}\n"
                f"name: {self.name}\n"
                f"mu: {self.mu}\n"
                f"sigma: {self.sigma}\n"
            )
        else:
            return (
                f"Plackett-Luce Player Data: \n\n"
                f"id: {self.id}\n"
                f"mu: {self.mu}\n"
                f"sigma: {self.sigma}\n"
            )

    def __hash__(self) -> int:
        return hash((self.id, self.mu, self.sigma))

    def __deepcopy__(self, memodict: Dict[Any, Any] = {}) -> "PlackettLuceRating":
        plr = PlackettLuceRating(self.mu, self.sigma, self.name)
        plr.id = self.id
        return plr

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PlackettLuceRating):
            if self.mu == other.mu and self.sigma == other.sigma:
                return True
            else:
                return False
        else:
            return NotImplemented

    def __lt__(self, other: "PlackettLuceRating") -> bool:
        if isinstance(other, PlackettLuceRating):
            if self.ordinal() < other.ordinal():
                return True
            else:
                return False
        else:
            raise ValueError(
                "You can only compare PlackettLuceRating objects with each other."
            )

    def __gt__(self, other: "PlackettLuceRating") -> bool:
        if isinstance(other, PlackettLuceRating):
            if self.ordinal() > other.ordinal():
                return True
            else:
                return False
        else:
            raise ValueError(
                "You can only compare PlackettLuceRating objects with each other."
            )

    def __le__(self, other: "PlackettLuceRating") -> bool:
        if isinstance(other, PlackettLuceRating):
            if self.ordinal() <= other.ordinal():
                return True
            else:
                return False
        else:
            raise ValueError(
                "You can only compare PlackettLuceRating objects with each other."
            )

    def __ge__(self, other: "PlackettLuceRating") -> bool:
        if isinstance(other, PlackettLuceRating):
            if self.ordinal() >= other.ordinal():
                return True
            else:
                return False
        else:
            raise ValueError(
                "You can only compare PlackettLuceRating objects with each other."
            )

    def ordinal(self, z: float = 3.0, alpha: float = 1, target: float = 0) -> float:
        r"""
        A single scalar value that represents the player's skill where their
        true skill is 99.7% likely to be higher.

        :param z: Float that represents the number of standard deviations to subtract
              from the mean. By default, set to 3.0, which corresponds to a
              99.7% confidence interval in a normal distribution.

        :param alpha: Float scaling factor applied to the entire calculation.
                      Adjusts the overall scale of the ordinal value.
                      Defaults to 1.

        :param target: Float value used to shift the ordinal value
                       towards a specific target. The shift is adjusted by the
                       alpha scaling factor. Defaults to 0.

        :return: :math:`\alpha \cdot ((\mu - z * \sigma) + \frac{\text{target}}{\alpha})`
        """
        return alpha * ((self.mu - z * self.sigma) + (target / alpha))


class PlackettLuceTeamRating:
    """
    The collective Plackett-Luce rating of a team.
    """

    def __init__(
        self,
        mu: float,
        sigma_squared: float,
        team: Sequence[PlackettLuceRating],
        rank: int,
    ):
        r"""
        :param mu: Represents the initial belief about the collective skill of
                   a team before any matches have been played. Known
                   mostly as the mean of the Guassian prior distribution.

                   *Represented by:* :math:`\mu`

        :param sigma_squared: Standard deviation of the prior distribution of a team.

                      *Represented by:* :math:`\sigma = \frac{\mu}{z}`
                      where :math:`z` is an integer that represents the
                      variance of the skill of a player.

        :param team: A list of Weng-Lin player ratings.

        :param rank: The rank of the team within a gam
        """
        self.mu = float(mu)
        self.sigma_squared = float(sigma_squared)
        self.team = team
        self.rank = rank

    def __repr__(self) -> str:
        return (
            f"PlackettLuceTeamRating(mu={self.mu}, sigma_squared={self.sigma_squared})"
        )

    def __str__(self) -> str:
        return (
            f"PlackettLuceTeamRating Details:\n\n"
            f"mu: {self.mu}\n"
            f"sigma_squared: {self.sigma_squared}\n"
            f"rank: {self.rank}\n"
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PlackettLuceTeamRating):
            return (
                self.mu == other.mu
                and self.sigma_squared == other.sigma_squared
                and self.team == other.team
                and self.rank == other.rank
            )
        else:
            return NotImplemented

    def __hash__(self) -> int:
        return hash((self.mu, self.sigma_squared, tuple(self.team), self.rank))


def _gamma(
    c: float,
    k: int,
    mu: float,
    sigma_squared: float,
    team: Sequence[PlackettLuceRating],
    rank: int,
    weights: Optional[List[float]] = None,
) -> float:
    """
    Default gamma function for Plackett-Luce.

    :param c: The square root of the collective team sigma.

    :param k: The number of teams in the game.

    :param mu: The mean of the team's rating.

    :param sigma_squared: The variance of the team's rating.

    :param team: The team rating object.

    :param rank: The rank of the team.

    :param weights: The weights of the players in a team.

    :return: A number.
    """
    return math.sqrt(sigma_squared) / c


class PlackettLuce:
    r"""
    Algorithm 4 by :cite:t:`JMLR:v12:weng11a`

    The PlackettLuce model departs from single scalar representations of
    player performance present in simpler models. There is a vector of
    abilities for each player that captures their performance across multiple
    dimensions. The outcome of a match between multiple players depends on
    their abilities in each dimension. By introducing this multidimensional
    aspect, the Plackett-Luce model provides a richer framework for ranking
    players based on their abilities in various dimensions.
    """

    def __init__(
        self,
        mu: float = 25.0,
        sigma: float = 25.0 / 3.0,
        beta: float = 25.0 / 6.0,
        kappa: float = 0.0001,
        gamma: Callable[
            [
                float,
                int,
                float,
                float,
                Sequence[PlackettLuceRating],
                int,
                Optional[List[float]],
            ],
            float,
        ] = _gamma,
        tau: float = 25.0 / 300.0,
        limit_sigma: bool = False,
        balance: bool = False,
    ):
        r"""
        :param mu: Represents the initial belief about the skill of
                   a player before any matches have been played. Known
                   mostly as the mean of the Gaussian prior distribution.

                   *Represented by:* :math:`\mu`

        :param sigma: Standard deviation of the prior distribution of player.

                      *Represented by:* :math:`\sigma = \frac{\mu}{z}`
                      where :math:`z` is an integer that represents the
                      variance of the skill of a player.


        :param beta: Hyperparameter that determines the level of uncertainty
                     or variability present in the prior distribution of
                     ratings.

                     *Represented by:* :math:`\beta = \frac{\sigma}{2}`

        :param kappa: Arbitrary small positive real number that is used to
                      prevent the variance of the posterior distribution from
                      becoming too small or negative. It can also be thought
                      of as a regularization parameter.

                      *Represented by:* :math:`\kappa`

        :param gamma: Custom function you can pass that must contain 5
                      parameters. The function must return a float or int.

                      *Represented by:* :math:`\gamma`

        :param tau: Additive dynamics parameter that prevents sigma from
                    getting too small to increase rating change volatility.

                    *Represented by:* :math:`\tau`

        :param limit_sigma: Boolean that determines whether to restrict
                            the value of sigma from increasing.

        :param balance: Boolean that determines whether to emphasize
                        rating outliers.
        """
        # Model Parameters
        self.mu: float = float(mu)
        self.sigma: float = float(sigma)
        self.beta: float = beta
        self.kappa: float = float(kappa)
        self.gamma: Callable[
            [
                float,
                int,
                float,
                float,
                Sequence[PlackettLuceRating],
                int,
                Optional[List[float]],
            ],
            float,
        ] = gamma

        self.tau: float = float(tau)
        self.limit_sigma: bool = limit_sigma
        self.balance: bool = balance

        # Model Data Container
        self.PlackettLuceRating: Type[PlackettLuceRating] = PlackettLuceRating

    def __repr__(self) -> str:
        return f"PlackettLuce(mu={self.mu}, sigma={self.sigma})"

    def __str__(self) -> str:
        return (
            f"Plackett-Luce Model Parameters: \n\n"
            f"mu: {self.mu}\n"
            f"sigma: {self.sigma}\n"
        )

    def rating(
        self,
        mu: Optional[float] = None,
        sigma: Optional[float] = None,
        name: Optional[str] = None,
    ) -> PlackettLuceRating:
        r"""
        Returns a new rating object with your default parameters. The given
        parameters can be overridden from the defaults provided by the main
        model, but is not recommended unless you know what you are doing.

        :param mu: Represents the initial belief about the skill of
                   a player before any matches have been played. Known
                   mostly as the mean of the Gaussian prior distribution.

                   *Represented by:* :math:`\mu`

        :param sigma: Standard deviation of the prior distribution of player.

                      *Represented by:* :math:`\sigma = \frac{\mu}{z}`
                      where :math:`z` is an integer that represents the
                      variance of the skill of a player.

        :param name: Optional name for the player.

        :return: :class:`PlackettLuceRating` object
        """
        return self.PlackettLuceRating(
            mu if mu is not None else self.mu,
            sigma if sigma is not None else self.sigma,
            name,
        )

    @staticmethod
    def create_rating(
        rating: List[float], name: Optional[str] = None
    ) -> PlackettLuceRating:
        """
        Create a :class:`PlackettLuceRating` object from a list of `mu`
        and `sigma` values.

        :param rating: A list of two values where the first value is the :code:`mu`
                       and the second value is the :code:`sigma`.

        :param name: An optional name for the player.

        :return: A :class:`PlackettLuceRating` object created from the list passed in.
        """
        if isinstance(rating, PlackettLuceRating):
            raise TypeError("Argument is already a 'PlackettLuceRating' object.")
        elif len(rating) == 2 and isinstance(rating, list):
            for value in rating:
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"The {rating.__class__.__name__} contains an "
                        f"element '{value}' of type '{value.__class__.__name__}'"
                    )
            if not name:
                return PlackettLuceRating(mu=rating[0], sigma=rating[1])
            else:
                return PlackettLuceRating(mu=rating[0], sigma=rating[1], name=name)
        else:
            raise TypeError(f"Cannot accept '{rating.__class__.__name__}' type.")

    @staticmethod
    def _check_teams(teams: List[List[PlackettLuceRating]]) -> None:
        """
        Ensure teams argument is valid.
        :param teams: List of lists of PlackettLuceRating objects.
        """
        # Catch teams argument errors
        if isinstance(teams, list):
            if len(teams) < 2:
                raise ValueError(
                    f"Argument 'teams' must have at least 2 teams, not {len(teams)}."
                )

            for team in teams:
                if isinstance(team, list):
                    if len(team) < 1:
                        raise ValueError(
                            f"Argument 'teams' must have at least 1 player per team, not {len(team)}."
                        )

                    for player in team:
                        if isinstance(player, PlackettLuceRating):
                            pass
                        else:
                            raise TypeError(
                                f"Argument 'teams' must be a list of lists of 'PlackettLuceRating' objects, "
                                f"not '{player.__class__.__name__}'."
                            )
                else:
                    raise TypeError(
                        f"Argument 'teams' must be a list of lists of 'PlackettLuceRating' objects, "
                        f"not '{team.__class__.__name__}'."
                    )
        else:
            raise TypeError(
                f"Argument 'teams' must be a list of lists of 'PlackettLuceRating' objects, "
                f"not '{teams.__class__.__name__}'."
            )

    def rate(
        self,
        teams: List[List[PlackettLuceRating]],
        ranks: Optional[List[float]] = None,
        scores: Optional[List[float]] = None,
        weights: Optional[List[List[float]]] = None,
        tau: Optional[float] = None,
        limit_sigma: Optional[bool] = None,
    ) -> List[List[PlackettLuceRating]]:
        """
        Calculate the new ratings based on the given teams and parameters.

        :param teams: A list of teams where each team is a list of
                      :class:`PlackettLuceRating` objects.

        :param ranks: A list of floats where the lower values
                      represent winners.

        :param scores: A list of floats where higher values
                      represent winners.

        :param weights: A list of lists of floats, where each inner list
                        represents the contribution of each player to the
                        team's performance.

        :param tau: Additive dynamics parameter that prevents sigma from
                    getting too small to increase rating change volatility.

        :param limit_sigma: Boolean that determines whether to restrict
                            the value of sigma from increasing.

        :return: A list of teams where each team is a list of updated
                :class:`PlackettLuceRating` objects.
        """
        # Catch teams argument errors
        self._check_teams(teams)

        # Catch ranks argument errors
        if ranks:
            if isinstance(ranks, list):
                if len(ranks) != len(teams):
                    raise ValueError(
                        f"Argument 'ranks' must have the same number of elements as 'teams', "
                        f"not {len(ranks)}."
                    )

                for rank in ranks:
                    if isinstance(rank, (int, float)):
                        pass
                    else:
                        raise TypeError(
                            f"Argument 'ranks' must be a list of 'int' or 'float' values, "
                            f"not '{rank.__class__.__name__}'."
                        )
            else:
                raise TypeError(
                    f"Argument 'ranks' must be a list of 'int' or 'float' values, "
                    f"not '{ranks.__class__.__name__}'."
                )

            # Catch scores and ranks together
            if scores:
                raise ValueError(
                    "Cannot accept both 'ranks' and 'scores' arguments at the same time."
                )

        # Catch scores argument errors
        if scores:
            if isinstance(scores, list):
                if len(scores) != len(teams):
                    raise ValueError(
                        f"Argument 'scores' must have the same number of elements as 'teams', "
                        f"not {len(scores)}."
                    )

                for score in scores:
                    if isinstance(score, (int, float)):
                        pass
                    else:
                        raise TypeError(
                            f"Argument 'scores' must be a list of 'int' or 'float' values, "
                            f"not '{score.__class__.__name__}'."
                        )
            else:
                raise TypeError(
                    f"Argument 'scores' must be a list of 'int' or 'float' values, "
                    f"not '{scores.__class__.__name__}'."
                )

        # Catch weights argument errors
        if weights:
            if isinstance(weights, list):
                if len(weights) != len(teams):
                    raise ValueError(
                        f"Argument 'weights' must have the same number of elements as"
                        f" 'teams', not {len(weights)}."
                    )

                for index, team_weights in enumerate(weights):
                    if isinstance(team_weights, list):
                        if len(team_weights) != len(teams[index]):
                            raise ValueError(
                                f"Argument 'weights' must have the same number of elements"
                                f"as each team in 'teams', not {len(team_weights)}."
                            )
                        for weight in team_weights:
                            if isinstance(weight, (int, float)):
                                pass
                            else:
                                raise TypeError(
                                    f"Argument 'weights' must be a list of lists of 'float' values, "
                                    f"not '{weight.__class__.__name__}'."
                                )
                    else:
                        raise TypeError(
                            f"Argument 'weights' must be a list of lists of 'float' values, "
                            f"not '{team_weights.__class__.__name__}'."
                        )
            else:
                raise TypeError(
                    f"Argument 'weights' must be a list of lists of 'float' values, "
                    f"not '{weights.__class__.__name__}'."
                )

        # Deep Copy Teams
        original_teams = teams
        teams = copy.deepcopy(original_teams)

        # Correct Sigma With Tau
        tau = tau if tau else self.tau
        tau_squared = tau * tau
        for team_index, team in enumerate(teams):
            for player_index, player in enumerate(team):
                teams[team_index][player_index].sigma = math.sqrt(
                    player.sigma * player.sigma + tau_squared
                )

        # Convert Score to Ranks
        if not ranks and scores:
            ranks = []
            for score in scores:
                ranks.append(_unary_minus(score))

        # Normalize Weights
        if weights:
            weights = [_normalize(team_weights, 1, 2) for team_weights in weights]

        tenet = None
        if ranks:
            rank_teams_unwound = _unwind(ranks, teams)

            if weights:
                weights, _ = _unwind(ranks, weights)

            ordered_teams = rank_teams_unwound[0]
            tenet = rank_teams_unwound[1]
            teams = ordered_teams
            ranks = sorted(ranks)

        processed_result = []
        if ranks and tenet:
            result = self._compute(teams=teams, ranks=ranks, weights=weights)
            unwound_result = _unwind(tenet, result)[0]
            for item in unwound_result:
                team = []
                for player in item:
                    team.append(player)
                processed_result.append(team)
        else:
            result = self._compute(teams=teams, weights=weights)
            for item in result:
                team = []
                for player in item:
                    team.append(player)
                processed_result.append(team)

        # Possible Final Result
        final_result = processed_result

        if limit_sigma is not None:
            self.limit_sigma = limit_sigma

        if self.limit_sigma:
            final_result = []

            # Reuse processed_result
            for team_index, team in enumerate(processed_result):
                final_team = []
                for player_index, player in enumerate(team):
                    player.sigma = min(
                        player.sigma, original_teams[team_index][player_index].sigma
                    )
                    final_team.append(player)
                final_result.append(final_team)
        return final_result

    def _c(self, team_ratings: List[PlackettLuceTeamRating]) -> float:
        r"""
        Calculate the square root of the collective team sigma.

        *Represented by:*

        .. math::

           c = \Biggl(\sum_{i=1}^k (\sigma_i^2 + \beta^2) \Biggr)

        Algorithm 4: Procedure 3 in :cite:p:`JMLR:v12:weng11a`

        :param team_ratings: The whole rating of a list of teams in a game.
        :return: A number.
        """
        beta_squared = self.beta**2
        collective_team_sigma = 0.0
        for team in team_ratings:
            collective_team_sigma += team.sigma_squared + beta_squared
        return math.sqrt(collective_team_sigma)

    @staticmethod
    def _sum_q(team_ratings: List[PlackettLuceTeamRating], c: float) -> List[float]:
        r"""
        Sum up all the values of :code:`mu / c` raised to :math:`e`.

        *Represented by:*

        .. math::

           \sum_{s \in C_q} e^{\theta_s / c}, q=1, ...,k, \text{where } C_q = \{i: r(i) \geq r(q)\}

        Algorithm 4: Procedure 3 in :cite:p:`JMLR:v12:weng11a`

        :param team_ratings: The whole rating of a list of teams in a game.

        :param c: The square root of the collective team sigma.

        :return: A list of floats.
        """

        sum_q: Dict[int, float] = {}
        for i, team_i in enumerate(team_ratings):
            summed = math.exp(team_i.mu / c)
            for q, team_q in enumerate(team_ratings):
                if team_i.rank >= team_q.rank:
                    if q in sum_q:
                        sum_q[q] += summed
                    else:
                        sum_q[q] = summed
        return list(sum_q.values())

    @staticmethod
    def _a(team_ratings: List[PlackettLuceTeamRating]) -> List[int]:
        r"""
        Count the number of times a rank appears in the list of team ratings.

        *Represented by:*

        .. math::

           A_q = |\{s: r(s) = r(q)\}|, q = 1,...,k

        :param team_ratings: The whole rating of a list of teams in a game.
        :return: A list of ints.
        """
        result = list(
            map(
                lambda i: len(list(filter(lambda q: i.rank == q.rank, team_ratings))),
                team_ratings,
            )
        )
        return result

    def _compute(
        self,
        teams: Sequence[Sequence[PlackettLuceRating]],
        ranks: Optional[List[float]] = None,
        weights: Optional[List[List[float]]] = None,
    ) -> List[List[PlackettLuceRating]]:
        # Initialize Constants
        original_teams = teams
        team_ratings = self._calculate_team_ratings(teams, ranks=ranks)
        c = self._c(team_ratings)
        sum_q = self._sum_q(team_ratings, c)
        a = self._a(team_ratings)

        result = []
        for i, team_i in enumerate(team_ratings):
            omega = 0.0
            delta = 0.0
            i_mu_over_c = math.exp(team_i.mu / c)

            for q, team_q in enumerate(team_ratings):
                i_mu_over_ce_over_sum_q = i_mu_over_c / sum_q[q]
                if team_q.rank <= team_i.rank:
                    delta += (
                        i_mu_over_ce_over_sum_q * (1 - i_mu_over_ce_over_sum_q) / a[q]
                    )
                    if q == i:
                        omega += (1 - i_mu_over_ce_over_sum_q) / a[q]
                    else:
                        omega -= i_mu_over_ce_over_sum_q / a[q]

            omega *= team_i.sigma_squared / c
            delta *= team_i.sigma_squared / c**2

            if weights:
                gamma_value = self.gamma(
                    c,
                    len(team_ratings),
                    team_i.mu,
                    team_i.sigma_squared,
                    team_i.team,
                    team_i.rank,
                    weights[i],
                )
            else:
                gamma_value = self.gamma(
                    c,
                    len(team_ratings),
                    team_i.mu,
                    team_i.sigma_squared,
                    team_i.team,
                    team_i.rank,
                    None,
                )
            delta *= gamma_value

            intermediate_result_per_team = []
            for j, j_players in enumerate(team_i.team):
                if weights:
                    weight = weights[i][j]
                else:
                    weight = 1

                mu = j_players.mu
                sigma = j_players.sigma

                if omega > 0:
                    mu += (sigma**2 / team_i.sigma_squared) * omega * weight
                    sigma *= math.sqrt(
                        max(
                            1 - (sigma**2 / team_i.sigma_squared) * delta * weight,
                            self.kappa,
                        ),
                    )
                else:
                    mu += (sigma**2 / team_i.sigma_squared) * omega / weight
                    sigma *= math.sqrt(
                        max(
                            1 - (sigma**2 / team_i.sigma_squared) * delta / weight,
                            self.kappa,
                        ),
                    )

                modified_player = original_teams[i][j]
                modified_player.mu = mu
                modified_player.sigma = sigma
                intermediate_result_per_team.append(modified_player)
            result.append(intermediate_result_per_team)
        return result

    def predict_win(self, teams: List[List[PlackettLuceRating]]) -> List[float]:
        r"""
        Predict how likely a match up against teams of one or more players
        will go. This algorithm has a time complexity of
        :math:`\mathcal{0}(n^2)` where 'n' is the number of teams.

        This is a generalization of the algorithm in
        :cite:p:`Ibstedt1322103` to asymmetric n-player n-teams.

        :param teams: A list of two or more teams.
        :return: A list of odds of each team winning.
        """
        # Check Arguments
        self._check_teams(teams)

        n = len(teams)

        # 2 Player Case
        if n == 2:
            teams_ratings = self._calculate_team_ratings(teams)
            a = teams_ratings[0]
            b = teams_ratings[1]
            result = phi_major(
                (a.mu - b.mu)
                / math.sqrt(2 * self.beta**2 + a.sigma_squared + b.sigma_squared)
            )
            return [result, 1 - result]

        pairwise_probabilities = []
        for pair_a, pair_b in itertools.permutations(teams, 2):
            pair_a_subset = self._calculate_team_ratings([pair_a])
            pair_b_subset = self._calculate_team_ratings([pair_b])
            mu_a = pair_a_subset[0].mu
            sigma_a = pair_a_subset[0].sigma_squared
            mu_b = pair_b_subset[0].mu
            sigma_b = pair_b_subset[0].sigma_squared
            pairwise_probabilities.append(
                phi_major(
                    (mu_a - mu_b) / math.sqrt(2 * self.beta**2 + sigma_a + sigma_b)
                )
            )

        win_probabilities = []
        for i in range(n):
            team_win_probability = sum(
                pairwise_probabilities[j] for j in range(i * (n - 1), (i + 1) * (n - 1))
            ) / (n - 1)
            win_probabilities.append(team_win_probability)

        total_probability = sum(win_probabilities)
        return [probability / total_probability for probability in win_probabilities]

    def predict_draw(self, teams: List[List[PlackettLuceRating]]) -> float:
        r"""
        Predict how likely a match up against teams of one or more players
        will draw. This algorithm has a time complexity of
        :math:`\mathcal{0}(n^2)` where 'n' is the number of teams.

        :param teams: A list of two or more teams.
        :return: The odds of a draw.
        """
        # Check Arguments
        self._check_teams(teams)

        total_player_count = sum(len(team) for team in teams)
        draw_probability = 1 / total_player_count
        draw_margin = (
            math.sqrt(total_player_count)
            * self.beta
            * phi_major_inverse((1 + draw_probability) / 2)
        )

        pairwise_probabilities = []
        for pair_a, pair_b in itertools.combinations(teams, 2):
            pair_a_subset = self._calculate_team_ratings([pair_a])
            pair_b_subset = self._calculate_team_ratings([pair_b])
            mu_a = pair_a_subset[0].mu
            sigma_a = pair_a_subset[0].sigma_squared
            mu_b = pair_b_subset[0].mu
            sigma_b = pair_b_subset[0].sigma_squared
            pairwise_probabilities.append(
                phi_major(
                    (draw_margin - mu_a + mu_b)
                    / math.sqrt(2 * self.beta**2 + sigma_a + sigma_b)
                )
                - phi_major(
                    (mu_b - mu_a - draw_margin)
                    / math.sqrt(2 * self.beta**2 + sigma_a + sigma_b)
                )
            )

        return sum(pairwise_probabilities) / len(pairwise_probabilities)

    def predict_rank(
        self, teams: List[List[PlackettLuceRating]]
    ) -> List[Tuple[int, float]]:
        r"""
        Predict the shape of a match outcome. This algorithm has a time
        complexity of :math:`\mathcal{0}(n^2)` where 'n' is the
        number of teams.

        :param teams: A list of two or more teams.
        :return: A list of team ranks with their probabilities.
        """
        self._check_teams(teams)

        n = len(teams)
        team_ratings = self._calculate_team_ratings(teams)

        win_probabilities = []
        for i, team_i in enumerate(team_ratings):
            team_win_probability = 0.0
            for j, team_j in enumerate(team_ratings):
                if i != j:
                    team_win_probability += phi_major(
                        (team_i.mu - team_j.mu)
                        / math.sqrt(
                            2 * self.beta**2
                            + team_i.sigma_squared
                            + team_j.sigma_squared
                        )
                    )
            win_probabilities.append(team_win_probability / (n - 1))

        total_probability = sum(win_probabilities)
        normalized_probabilities = [p / total_probability for p in win_probabilities]

        sorted_teams = sorted(
            enumerate(normalized_probabilities), key=lambda x: x[1], reverse=True
        )

        ranks = [0] * n
        current_rank = 1
        for i, (team_index, _) in enumerate(sorted_teams):
            if i > 0 and sorted_teams[i][1] < sorted_teams[i - 1][1]:
                current_rank = i + 1
            ranks[team_index] = current_rank

        return list(zip(ranks, normalized_probabilities))

    def _calculate_team_ratings(
        self,
        game: Sequence[Sequence[PlackettLuceRating]],
        ranks: Optional[List[float]] = None,
        weights: Optional[List[List[float]]] = None,
    ) -> List[PlackettLuceTeamRating]:
        """
        Get the team ratings of a game.

        :param game: A list of teams, where teams are lists of
                     :class:`PlackettLuceRating` objects.

        :param ranks: A list of ranks for each team in the game.

        :param weights: A list of lists of floats, where each inner list
                        represents the contribution of each player to the
                        team's performance. The values should be normalized
                        from 0 to 1.

        :return: A list of :class:`PlackettLuceTeamRating` objects.
        """
        if ranks:
            rank = self._calculate_rankings(game, ranks)
        else:
            rank = self._calculate_rankings(game)

        result = []
        for index, team in enumerate(game):
            sorted_team = sorted(team, key=lambda p: p.ordinal(), reverse=True)
            max_ordinal = sorted_team[0].ordinal()

            mu_summed = 0.0
            sigma_squared_summed = 0.0
            for player in sorted_team:
                if self.balance:
                    ordinal_diff = max_ordinal - player.ordinal()
                    balance_weight = 1 + (ordinal_diff / (max_ordinal + self.kappa))
                else:
                    balance_weight = 1.0
                mu_summed += player.mu * balance_weight
                sigma_squared_summed += (player.sigma * balance_weight) ** 2
            result.append(
                PlackettLuceTeamRating(
                    mu_summed, sigma_squared_summed, team, rank[index]
                )
            )
        return result

    def _calculate_rankings(
        self,
        game: Sequence[Sequence[PlackettLuceRating]],
        ranks: Optional[List[float]] = None,
    ) -> List[int]:
        """
        Calculates the rankings based on the scores or ranks of the teams.

        It assigns a rank to each team based on their score, with the team with
        the highest score being ranked first.

        :param game: A list of teams, where teams are lists of
                     :class:`PlackettLuceRating` objects.

        :param ranks: A list of ranks for each team in the game.

        :return: A list of ranks for each team in the game.
        """
        if ranks:
            team_scores = []
            for index, _ in enumerate(game):
                if isinstance(ranks[index], int):
                    team_scores.append(ranks[index])
                else:
                    team_scores.append(index)
        else:
            team_scores = [i for i, _ in enumerate(game)]

        rank_output = {}
        s = 0
        for index, value in enumerate(team_scores):
            if index > 0:
                if team_scores[index - 1] < team_scores[index]:
                    s = index
            rank_output[index] = s
        return list(rank_output.values())


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


PageRankEdge = tuple[TeamID, TeamID]


class RankingModel(Protocol):
    """Ranking model interface."""

    def add_match(self, match: Match) -> None:
        """Add a match to the model."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset the model."""
        raise NotImplementedError


class PageRank(RankingModel):
    """Class for the page rank model."""

    ALPHA = 0.85
    TOL = 1e-6
    MAX_ITER = 100

    def __init__(self) -> None:
        """
        Initialize the BasketballPageRank class.

        Args:
            alpha: Damping factor for PageRank.
            tol: Convergence tolerance.
            max_iter: Maximum number of iterations.

        """
        self.teams: dict[TeamID, int] = {}
        self.games: list[PageRankEdge] = []
        self.__chached_ranks: dict[TeamID, float] | None = {}

    def add_match(self, match: Match) -> None:
        """
        Add a match to the dataset.

        Args:
             match: line with data

        """
        self.teams.setdefault(match.HID, len(self.teams))
        self.teams.setdefault(match.AID, len(self.teams))

        match_edge = (match.HID, match.AID) if match.H else (match.AID, match.HID)
        self.games.append(match_edge)
        self.__chached_ranks = None

    def ratings(self) -> dict[TeamID, float]:
        """
        Calculate the PageRank ratings for all teams.

        Here is where the magic happen.

        Returns:
            ranks: Dictionary mapping team IDs to their PageRank scores.

        """
        if not self.teams:
            return {}

        if self.__chached_ranks:
            return self.__chached_ranks

        number_of_teams = len(self.teams)

        # Build adjacency matrix
        adjacency_matrix = np.zeros((number_of_teams, number_of_teams))
        for loser, winner in self.games:
            adjacency_matrix[self.teams[winner], self.teams[loser]] += 1

        # Normalize the matrix
        for i in range(adjacency_matrix.shape[0]):
            if adjacency_matrix[i].sum() > 0:
                adjacency_matrix[i] /= adjacency_matrix[i].sum()
            else:
                adjacency_matrix[i] = 1 / number_of_teams  # Handle dangling nodes

        # PageRank algorithm
        rank = np.ones(number_of_teams) / number_of_teams
        for _ in range(self.MAX_ITER):
            new_rank = (
                self.ALPHA * adjacency_matrix.T @ rank
                + (1 - self.ALPHA) / number_of_teams
            )
            if np.linalg.norm(new_rank - rank, ord=1) < self.TOL:
                break
            rank = new_rank

        # Map scores to teams
        ratings = {team: rank[self.teams[team]] for team in self.teams}
        self.__chached_ranks = ratings
        return ratings

    def team_rating(
        self, team_id1: TeamID, team_id2: TeamID
    ) -> tuple[float | None, float | None]:
        """
        Get the rating of one or two teams.

        Args:
            team_id1: ID of the first team.
            team_id2: ID of the second team.

        Returns:
            vector of ratings

        """
        ratings = self.ratings()
        return (ratings.get(team_id1, None), ratings.get(team_id2, None))

    def team_rating_ratio(self, team_id1: TeamID, team_id2: TeamID) -> float | None:
        """
        Get the ratio of winning two teams.

        Args:
            team_id1: ID of the first team.
            team_id2: ID of the second team.

        Returns:
            ratio of winning(home 0, away 100)

        """
        ratings = self.ratings()

        rating1 = ratings.get(team_id1, None)
        rating2 = ratings.get(team_id2, None)

        if rating1 is None or rating2 is None:
            return None

        return 100 * rating2 / (rating1 + rating2)

    def reset(self) -> None:
        """Reset all data (forget all teams and matches)."""
        self.teams = {}
        self.games = []


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
            {"type": "ineq", "fun": lambda props: 0.7 - sum(props)}
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
    K: int = 100
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
        return 100 * (home_elo.predict(away_elo.rating) if played_enough else 0.5)

    def reset(self) -> None:
        """Reset the model."""
        self.teams_home.clear()
        self.teams_away.clear()


class Model:
    """Main class."""

    TRAIN_SIZE: int = 4000
    FIRST_TRAIN_MOD: int = 5

    def __init__(self) -> None:
        """Init classes."""
        self.seen_matches = set()
        self.elo = Elo()
        self.elo_by_location = EloByLocation()
        # self.pagerank = PageRank()
        self.player = Player()
        self.ai = Ai()
        self.trained = False
        self.data = Data()
        self.openskill = OpenSkillAPI()
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
            self.openskill.add_match(match)
            # self.pagerank.add_match(match)

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
            """print(
                f"Initial training on {games_increment[-train_size :].shape[0]}"
                f" matches with bankroll {summary.Bankroll}"
            )"""
            self.train_ai_reg(cast(pd.DataFrame, games_increment[-train_size:]))
        elif games_increment.shape[0] > 0:
            increment_season = int(games_increment.iloc[0]["Season"])
            if self.season_number != increment_season:
                self.elo.reset()
                self.elo_by_location.reset()
                self.season_number = increment_season

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
                """print(
                    f"{summary.Date}: retraining on {self.old_matches.shape[0]}"
                    f" matches with bankroll {summary.Bankroll}"
                )"""
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
        "OpenSkill",
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
        openskill = self.openskill.predict(match)
        # home_page, away_page = self.pagerank.team_rating(match.HID, match.AID)

        rankings = pd.Series(
            [home_elo, away_elo, elo_by_location_prediction, openskill],
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
            self.openskill.add_match(match)
            # self.pagerank.add_match(match)

        training_dataframe = pd.DataFrame(
            training_data, columns=pd.Index(self.TRAINING_DATA_COLUMNS)
        )

        outcomes = pd.Series(outcomes_list)

        self.old_matches = training_dataframe
        self.old_outcomes = outcomes

        self.ai.train_reg(training_dataframe, outcomes)
        self.trained = True


class Ai:
    """Class for training and predicting."""

    def plot_home_away(self, elo_data: np.ndarray) -> None:
        """Plot ELO ranks for home and away teams."""
        import matplotlib.pyplot as plt

        if (
            not isinstance(elo_data, np.ndarray)
            or elo_data.ndim != 2
            or elo_data.shape[1] < 2
        ):
            raise ValueError("Input.")

        # Extract home and away ELO ranks
        home_elo = elo_data[:, 0]
        away_elo = elo_data[:, 1]
        num_matches = elo_data.shape[0]

        # Plotting the data
        plt.figure(figsize=(15, 7))
        plt.scatter(
            range(num_matches),
            home_elo,
            label="Home ELO",
            color="blue",
            alpha=0.5,
            s=10,
        )
        plt.scatter(
            range(num_matches), away_elo, label="Away ELO", color="red", alpha=0.5, s=10
        )

        # Add labels and title
        plt.xlabel("Match Index")
        plt.ylabel("ELO Rank")
        plt.title("ELO Ranks for Home and Away Teams")
        plt.legend()
        plt.grid(True, alpha=0.5)

        # Show the plot
        plt.tight_layout()
        plt.show()

    def __init__(self):
        """Create a new Model from a XGBClassifier."""
        self.initialized = False
        self.model: xgb.XGBRegressor | xgb.XGBClassifier

    def train(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""
        if not self.initialized:
            self.model = xgb.XGBClassifier()
            self.initialized = True

        self.model = self.model.fit(training_dataframe, outcomes)

    def train_reg(self, training_dataframe: pd.DataFrame, outcomes: pd.Series) -> None:
        """Return trained model."""
        x, x_val, y, y_val = model_selection.train_test_split(
            training_dataframe.to_numpy(),
            outcomes.to_numpy(),
            test_size=0.05,
            shuffle=True,
            random_state=2,
        )
        self.model = neural_network.MLPRegressor(max_iter=100)
        self.model.fit(x, y)
        self.initialized = True
        print("MAE:", metrics.mean_absolute_error(y_val, self.model.predict(x_val)))

    def get_probabilities(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        return self.model.predict_proba(dataframe.to_numpy())

    def get_probabilities_reg(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Get probabilities for match outcome [home_loss, home_win]."""
        predicted_score_differences = self.model.predict(dataframe.to_numpy())
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
