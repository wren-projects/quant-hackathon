import os

import numpy as np

from quant.bet import Player
from quant.data import Data
from quant.predict import Ai

"""Unused at the moment, could be modified to creating models"""


def next_round(player: Player, ai: Ai, new_matches, ratios):
    probabbilities = ai.get_probabilities(new_matches)
    # ratios and probabilities have the same format
    bets = player.get_betting_strategy(probabbilities, ratios)
    put_bets(bets)


def put_bets(bets):
    pass


def main() -> None:
    data_path = ""
    model_path = ""
    start_buget = 1000
    model = None

    player = Player(start_buget)
    data = Data(data_path)
    ai = Ai(True, model_path, data)

    while False:
        new_bet_matches = []
        next_round(player, new_bet_matches, model)
        new_budget = 1000
        player.edit_budget(new_buget)
        new_matches = []
        data.add_new_match_outcome(new_matches)


if __name__ == "__main__":
    main()
