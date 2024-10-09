#!/usr/bin/env python3

from pathlib import Path

import pandas as pd
import typer
from negmas.preferences.ops import itertools
from rich import print
from utils import DTYPES

YEARS = [2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]


def main(
    src: Path,
    years: list[int] = YEARS,
    refresh: bool = True,
    ignore_empty: bool = True,
    verbose: bool = False,
):
    dst = src.absolute()
    if not src.exists():
        return
    data = pd.read_csv(src, dtype=DTYPES)
    if verbose:
        print(f"Found {len(data)} records")
    if len(data) < 1:
        if verbose:
            print(f"[red]Found {len(data)} records[/red]")
        return
    n_total = len(data)
    n_found = dict(zip(years, itertools.repeat(0)))
    for year in years:
        current = dst.parent / f"{dst.stem}{year}{dst.suffix}"
        data["year"] = data["year"].astype(int)
        x = data.loc[data.year == year, :]
        n_found[year] = len(x)
        if not ignore_empty or len(x):
            if verbose:
                print(f"Saving {len(x)} records for year {year}")
            x.to_csv(current, index=False)
        elif verbose:
            print(f"[red]No records for year {year}[/red]")
    print(n_found)
    print(f"Total: {n_total}")
    n_all = sum(list(n_found.values()))
    assert (
        n_total >= n_all
    ), f"Impossible to have {n_total=} records and total for all years of {n_all=}"
    if n_all != n_total:
        print(
            f"[red]Found {n_total} records but the total for all years considered is {n_all}. Will lose {n_total - n_all} records[/red]"
        )
    if refresh:
        current = dst.parent / f"{dst.stem}0{dst.suffix}"
        data.to_csv(current, index=False)


if __name__ == "__main__":
    typer.run(main)
