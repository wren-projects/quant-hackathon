# handles betting starteggy
import math
import numpy as np
import scipy.optimize



class Player:
    def __init__(self, budget):
        self.budget = budget
        self.R = 0

    def get_expected_profit(self, prob, ratio, prop_of_budget) -> float:
        return (prob * ratio - 1) * prop_of_budget

    def get_variance_of_profit(self, prob, ratio, prop_of_budget) -> float:
        return (1 - prob) * prob * (prop_of_budget**2) * (ratio**2)

    def sharpe_ratio(self, total_profit, total_var) -> float:
        if total_var == 0:
            return np.inf
        return (total_profit - self.R) / math.sqrt(total_var)

    def min_function(self, props, probs, ratios) -> float:
        """function that we are trying to minimaze for sharp ratio"""
        total_profit = 0
        total_var = 0
        for i in range(len(probs)):
            for j in range(2):
                prob = probs[i][j]
                ratio = ratios[i][j]
                prop_of_budget = props[i * len(probs[i]) + j]
                if not len(probs[i]) == 2 or len(ratios) != len(probs):
                    print("min_funciton, wrong format of probs")
                total_profit += self.get_expected_profit(prob, ratio, prop_of_budget)
                total_var += self.get_variance_of_profit(prob, ratio, prop_of_budget)
        return -self.sharpe_ratio(total_profit, total_var)

    def get_bet_proportions(self, probs, ratios) -> np.ndarray:
        """returns proportions of the budget to bet on speific probs, (in the same format as probs)"""
        num_bets = probs.shape[0] * probs.shape[1]
        initial_props = np.full(num_bets, 1 / num_bets, dtype=float)
        # Constraint: sum of all props <= 1 (global budget constraint for entire 2D array), idk man it does something
        cons = [{"type": "ineq", "fun": lambda props: 1.0 - sum(props)}]
        bounds = [(0, 1) for _ in range(num_bets)]
        result = scipy.optimize.minimize(
            self.min_function, initial_props, args=(probs, ratios), method="SLSQP", bounds=bounds, constraints=cons
        )
        optimized_props = result.x.reshape(probs.shape)
        return optimized_props

    def get_betting_strategy(self, probabbilities, ratios) -> list:
        """returns absolute cash numbers and on what to bet in 2d list"""
        proportions = self.get_bet_proportions(probabbilities, ratios)
        bets = [[0] * 2 for _ in range(len(proportions))]
        for i in range(len(proportions)):
            for j in range(2):
                bets[i][j] = proportions[i][j] * self.budget
        return bets

    def edit_budget(self, new):
        self.budget = new
