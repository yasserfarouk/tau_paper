#!/usr/bin/env python3
"""Removes all TAU results results."""
from pathlib import Path

import pandas as pd
import typer
from rich import print
from rich.progress import track
from utils import DTYPES, TAUNATIVE, clean_results


def main(
    src: list[Path],
    keep_proposed: bool = True,
    dry: bool = False,
    ignore_stats_folders: bool = True,
):
    """Removes all runs that of given mechanisms for which not all negotiators are in the allowed list."""

    new_files = []
    fileyear = 0
    for file_path in src:
        if file_path.is_dir() and not (
            ignore_stats_folders and file_path.name.endswith("stats")
        ):
            for new_file in file_path.glob("**/*.csv"):
                if not new_file.is_file():
                    continue
                new_files.append(new_file)
        else:
            new_files.append(file_path)
    files = new_files
    for file_path in files:
        clean_results(file_path)
    for f in track(files):
        print(f"Processing {f}")
        if dry:
            continue
        remove_mechanism_runs(f, keep_proposed=keep_proposed)


def remove_mechanism_runs(
    path: Path, mechanism: str = "TAU", keep_proposed: bool = True
):
    data = pd.read_csv(path, dtype=DTYPES)  # type: ignore
    print(f"{len(data)} records found {path}")
    cond = ~data.mechanism_name.str.startswith(mechanism)
    if keep_proposed:
        cond = cond | (data.strategy_name.isin(TAUNATIVE))
    data = data.loc[cond, :]
    print(f"{len(data)} records kept {path}")
    data.to_csv(path, index=False)


if __name__ == "__main__":
    typer.run(main)
