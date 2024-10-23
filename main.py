from analysis import Data
from bet import Player
from predict import Ai
import os


def next_round(player: Player, ai: Ai, new_matches):
    probabbilities = ai.get_predictions_for_matches(new_matches)
    bets = player.get_betting_strategy(probabbilities)
    put_bets(bets)


def put_bets():
    pass


if __name__ == "":
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
        new_buget = 1000
        player.edit_buget(new_buget)
        new_matches = []
        data.add_new_match_outcome(new_matches)
