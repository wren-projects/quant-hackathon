import sys
from pathlib import Path

import numpy as np
import pandas as pd

from quant.data import Data
from quant.predict import Ai
from quant.types import Match


def main(data_path: str, model_path: str) -> None:
    """Start testing run."""
    dataframe = pd.read_csv(data_path)

    # model = Elo()

    data = Data(dataframe)
    data.get_train_matrix()
    return

    ai = Ai.untrained()

    train_matrix = np.ndarray([dataframe.shape[0], 5])

    results = []

    for match in (Match(*x) for x in dataframe.itertuples()):
        model.add_match(match)

        home_elo = model.teams[match.HID].rating
        away_elo = model.teams[match.AID].rating

        results.append((home_elo > away_elo) == (match.H > match.A))

        train_matrix[match.Index] = [
            home_elo,
            away_elo,
            match.OddsH,
            match.OddsA,
            match.H,
        ]

    ai.train(train_matrix)

    ai.save_model(Path(model_path))

    print(sum(results) / len(results))


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "upload/data/games.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "quant/models/model.json"
    main(data_path, model_path)
