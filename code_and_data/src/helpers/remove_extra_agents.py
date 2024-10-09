#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import typer
from rich import print
from utils import (
    DTYPES,
    STRATEGY_NAME_MAP,
    TAUNATIVE,
    clean_results,
    get_all_anac_agents,
    remove_invalid_lines,
)


def remove_extra_agents(
    path: Path,
    year: int,
    remove_extra_years: bool = False,
    verbose=False,
    override=False,
    genius10=True,
    finalists=False,
):
    remove_invalid_lines(path, override=True, verbose=verbose)
    clean_results(path, verbose=verbose)
    data = pd.read_csv(path, dtype=DTYPES)  # type: ignore
    yearsfound = list(data.year.unique())
    if len(yearsfound) < 1 or len(yearsfound) > 1 or yearsfound[0] != year:
        if not remove_extra_years:
            print(
                f"[red]{path} has the following years: {yearsfound} not only {year}. Pass --remove-extra-years to remove these extra years[/red]"
            )
            return
        else:
            peryear = data.groupby(["year"]).size()
            print(
                f"{path} has the following years: {yearsfound} not only {year}\n{peryear}\nWill remove {len(data.loc[data.year!=year, :])} records of [red]wrong year[/red]"
            )
    agents = get_all_anac_agents(
        year, finalists_only=finalists, winners_only=False, genius10=genius10
    )
    if len(agents) == 0:
        raise ValueError(
            f"found No supported agents (finalists and/or genius10) for {year}"
        )
    agents = [_.split(":")[1] for _ in agents]
    agents += [STRATEGY_NAME_MAP.get(_, _) for _ in agents]
    agents = agents + TAUNATIVE
    if verbose:
        print(f"Keeping {agents} on year {year}")
    n = len(data)
    removed = data.loc[
        ~(
            (data.year == year)
            & (data.first_strategy_name.isin(agents))
            & (data.second_strategy_name.isin(agents))
        ),
        ["first_strategy_name", "second_strategy_name", "strategy_name"],
    ]
    data = data.loc[
        (data.year == year)
        & (data.first_strategy_name.isin(agents))
        & (data.second_strategy_name.isin(agents)),
        :,
    ]
    for k, v in DTYPES.items():
        if k not in data.columns or v != "category":
            continue
        data[k] = data[k].cat.remove_unused_categories()
    for k, v in DTYPES.items():
        if k not in removed.columns or v != "category":
            continue
        removed[k] = removed[k].cat.remove_unused_categories()
    m = len(data)
    if n == m:
        if verbose:
            print(
                f"[green]All OK. Found {len(data)} records in {year} for {agents} [/green]"
            )
    else:
        if override:
            data.to_csv(path, index=False)
        for col in ["first_strategy_name", "second_strategy_name", "strategy_name"]:
            print(f"{col}: {removed[col].unique()}")
        print(
            f"[yellow]Removed {n - m} records in {year} for agents other than {agents} [/yellow]"
        )
    return data


def main(
    path: Path,
    year: int,
    remove_extra_years: bool = True,
    remove_non_finalists: bool = False,
    remove_non_genius10: bool = False,
    verbose: bool = False,
):
    path = path.absolute()
    if not path.exists():
        return
    remove_extra_agents(
        path,
        year=year,
        remove_extra_years=remove_extra_years,
        verbose=verbose,
        override=True,
        finalists=remove_non_finalists,
        genius10=remove_non_genius10,
    )


if __name__ == "__main__":
    typer.run(main)
