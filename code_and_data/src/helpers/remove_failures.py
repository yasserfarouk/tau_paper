# coding: utf-8
from pathlib import Path

import pandas as pd
import typer
from rich import print
from rich.progress import track

path = Path("serverclean/final/final.csv")
STRATEGIES = ["WARNegotiator", "CABNegotiator"]


def remove_failures_for(
    data: pd.DataFrame, lst: list[str], both: bool = True, path: Path | None = None
) -> pd.DataFrame:
    n = len(data)
    if both:
        cond = (
            (data.first_strategy_name.isin(lst)) & (data.second_strategy_name.isin(lst))
        ) & (~data.succeeded)
    else:
        cond = (
            (data.first_strategy_name.isin(lst)) | (data.second_strategy_name.isin(lst))
        ) & (~data.succeeded)
    m = len(data.loc[cond])
    if m <= 0:
        return data
    nofailures = data.loc[~cond, :]
    assert len(nofailures) + m == n
    print(f"Removed {m} records from {path} keeping {len(nofailures)}")
    if path is not None:
        nofailures.to_csv(path, index=False)
    return nofailures


def main(
    base: Path,
    strategy: list[str] = STRATEGIES,
    override: bool = True,
    both: bool = True,
):
    """Removes failures between the given mechanisms"""
    if base.is_dir():
        for f in base.rglob("*.csv"):
            if f.name.startswith("missing"):
                continue
            try:
                data = pd.read_csv(path)
                remove_failures_for(data, strategy, both, f if override else None)
            except:
                print(f"[red]Failed for {f}[/red]")
        return

    data = pd.read_csv(path)
    remove_failures_for(data, strategy, both, base if override else None)
