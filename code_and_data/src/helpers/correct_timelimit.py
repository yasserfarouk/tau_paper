#!/usr/bin/env python3
"""Removes all incomplete domains based results."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
import typer
from rich import print
from rich.progress import track
from utils import DTYPES, clean_results, remove_invalid_lines


def process(
    src: Path,
    dst: Path,
    norun: bool,
    verbose: bool = True,
):
    """Removes all runs that of given mechanisms for which not all negotiators
    are in the allowed list."""
    remove_invalid_lines(src, override=True, verbose=verbose)
    clean_results(src)
    data = pd.read_csv(src, dtype=DTYPES)  # type: ignore
    existing_mechanisms = set(data.mechanism_name.unique())
    if verbose:
        print(f"Correcting timelimit and mechanism name typos on {src}")
    if data is None:
        print(f"\t[red]Failed to read {src}[/red]")
    n = len(data)
    if verbose:
        print(f"\tCorrecting timelimit for {len(data)} records")
    limits = data.loc[~data.time_limit.isna(), "time_limit"].unique()
    if verbose:
        print(f"\tExisting timelimits are {data.time_limit.unique()}")
    if len(limits) == 1 and limits[0] == 180:
        pass
    else:
        data.loc[(~data.time_limit.isna()), "time_limit"] = 180
        if verbose:
            print(f"Correcting mechanism name for {len(data)} records")
        limits = data.loc[~data.time_limit.isna(), "time_limit"].unique()
        assert len(limits) <= 1, f"Found multiple time limits: {limits}"
        assert (
            len(limits) == 0 or limits[0] == 180
        ), f"Found strange time limits: {limits}"
    # data.loc[data.mechanism_name == "AU0", "mechanism_name"] = "TAU0"
    data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
    assert (
        len(data) == n
    ), f"[red]Found {len(data)} records out of {n}. Correcting timelimit and mechanism name should never change the number of records[/red]"
    if norun:
        return
    if verbose:
        print(f"Saving to {dst}")
    non_existing_mechanisms = existing_mechanisms.difference(
        set(data.mechanism_name.unique())
    )
    assert (
        len(non_existing_mechanisms) == 0
    ), f"Mehcanisms {non_existing_mechanisms} were completely removed"
    data.to_csv(dst, index=False)


def main(
    src: Path,
    dst: Path,
    max_cores: int = -1,
    norun: bool = False,
    verbose: bool = False,
):
    src, dst = src.absolute(), dst.absolute()
    if src.is_dir():
        paths = [_.absolute() for _ in src.glob("**/*.csv")]
        dsts = [dst / _.relative_to(src) for _ in paths]
    else:
        paths = [src.absolute()]
        dsts = [dst.absolute()]

    futures = []
    if max_cores < 0 or len(paths) == 1:
        for s, d in track(zip(paths, dsts), total=len(paths)):
            process(s, d, norun, verbose)
    else:
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for s, d in track(zip(paths, dsts), total=len(paths)):
                futures.append(pool.submit(process, s, d, norun, verbose))
            print(f"[bold blue]Will work on {len(paths)} domains[/bold blue]")
            for f in track(as_completed(futures), total=len(futures)):
                f.result()


if __name__ == "__main__":
    typer.run(main)
