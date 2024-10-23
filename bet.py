#handles betting starteggy
import math

class Player:
    def __init__(self, buget):
        self.buget = buget
        self.R = 0

    def get_expected_profit(self, prob, ratio, prop_of_budget):
        return (prob*ratio-1)*prop_of_budget
    
    def get_variance_of_profit(self, prob, ratio, prop_of_budget):
        return (1-prob)*prob*(prop_of_budget*prop_of_budget)*(ratio*ratio)
    
    def sharpe_ratio(self, total_profit, total_var):
        return ((total_profit-self.R)/math.sqrt(total_var))


    