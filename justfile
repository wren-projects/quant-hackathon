bootstrap:
  poetry install

check:
  ruff check .
  pyright
  black --check --diff .

fix:
  ruff check --exit-zero --fix --unsafe-fixes .
  black .
