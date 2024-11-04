import sys

import pandas as pd

from quant.models.Elo import Elo
from quant.types import Match


def main(data_path: str, model_path: str) -> None:
    """Start testing run."""
    dataframe = pd.read_csv(data_path)

    model = Elo()

    for match in dataframe.itertuples(index=False, name="Match"):
        model.add_match(Match(*match))

    print(model)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "quant/datasets/games.csv"
    model_path = sys.argv[2] if len(sys.argv) > 2 else "quant/models/model.json"
    main(data_path, model_path)
