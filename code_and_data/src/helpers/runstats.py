#!/usr/bin/env python3

from pathlib import Path

import typer
from rich import print

from utils import read_data


def main(
        file: list[Path] = [Path() / "server"], output: Path = Path(".").absolute() / "runstats.csv"
):
    data, _ = read_data(
        file,
        failures=False,
        max_trials=0,
        verbose=True,
        add_maxs=False,
    )
    data.groupby(["Year", "Condition", "Domain"])["Rational (0)"].count().to_csv(output)
    print(data.groupby(["Year"])["Rational (0)"].count())


if __name__ == "__main__":
    typer.run(main)
