# handles betting starteggy
import math
from itertools import product

import numpy as np
import pandas as pd


class Player:
    """Handles betting strateggy."""

    def get_expected_profit(
        self, prob: float, ratio: float, prop_of_budget: float
    ) -> float:
        """Get expected profit for given parametrs."""
        return (prob * ratio - 1) * prop_of_budget

    def get_variance_of_profit(
        self, prob: float, ratio: float, prop_of_budget: float
    ) -> float:
        """Get varience of profit for given parameters."""
        return (1 - prob) * prob * (prop_of_budget**2) * (ratio**2)

    def sharpe_ratio(self, total_profit: float, total_var: float) -> float:
        """Return total sharp ratio."""
        if total_var == 0:
            return np.inf
        return total_profit / math.sqrt(total_var)

    def max_function(
        self, props: np.ndarray, probs: np.ndarray, ratios: np.ndarray
    ) -> float:
        """We are trying to minimaze this function for sharp ratio."""
        total_profit = 0
        total_var = 0
        for i in range(len(probs)):
            for j in range(2):
                prob = probs[i][j]
                ratio = ratios[i][j]
                prop_of_budget = props[i * len(probs[i]) + j]
                if len(probs[i]) != 2 or len(ratios) != len(probs):
                    print("min_funciton, wrong format of probs")
                total_profit += self.get_expected_profit(prob, ratio, prop_of_budget)
                total_var += self.get_variance_of_profit(prob, ratio, prop_of_budget)
        return self.sharpe_ratio(total_profit, total_var)

    def get_bet_proportions(
        self,
        probs: np.ndarray,
        active_matches: pd.DataFrame,
        summary: pd.DataFrame,
        step: float,
    ) -> np.ndarray:
        """Return proportions of the budget to bet on speific probs, (in the same format as probs)."""
        num_bets = probs.shape[0] * probs.shape[1]
        possible_ranges = []
        for _ in range(num_bets):
            min_bound = summary.iloc[0]["Min_bet"] / summary.iloc[0]["Bankroll"]
            max_bound = summary.iloc[0]["Max_bet"] / summary.iloc[0]["Bankroll"]
            possible_ranges.append(
                [0, *list(np.arange(min_bound, max_bound + step, step))]
            )

        all_combinations = product(*possible_ranges)
        best_sharpe = -np.inf
        best_allocation = None
        for props in all_combinations:
            sharpe = self.max_function(
                np.array(props), probs, np.array(active_matches[["OddsH", "OddsA"]])
            )
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_allocation = props

        return np.array(best_allocation).reshape(probs.shape)

    def get_betting_strategy(
        self,
        probabilities: np.ndarray,
        active_matches: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> list:
        """Return absolute cash numbers and on what to bet in 2d list."""
        proportions = self.get_bet_proportions(
            probabilities, active_matches, summary, 0.001
        )
        bets = [[0] * 2 for _ in range(len(proportions))]
        for i in range(len(proportions)):
            for j in range(2):
                bets[i][j] = proportions[i][j] * summary.iloc[0]["Bankroll"]
        return bets
