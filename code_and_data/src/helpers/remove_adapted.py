#!/usr/bin/env python3
"""Removes all adapter based results."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Sequence

import pandas as pd
import typer
from rich import print
from rich.progress import track
from utils import remove_runs, TAUNATIVE, DTYPES


def process(
    src: Path,
    dst: Path,
    mechanisms: Sequence[str],
    norun: bool,
    allowed: Sequence[str],
):
    """Removes all runs that of given mechanisms for which not all negotiators are in the allowed list."""
    data = pd.read_csv(src, dtype=DTYPES)
    if data is None:
        print(f"[red]Failed to read {src}[/red]")
    n = len(data)
    data = remove_runs(data, mechanisms=mechanisms, allowed=allowed, src=src)
    if norun:
        print(
            f"Will remove {n - len(data) } of {n} records keeping {len(data)} records"
        )
        strategies = set(data.first_strategy_name.unique()).union(
            set(data.second_strategy_name.unique())
        )
        print(f"Remaining strategies: {strategies}")
        return
    data.to_csv(dst, index=False)

def main(
    src: Path,
    dst: Path,
    mechanism: list[str] = ["TAU0"],
    norun: bool = False,
    max_cores: int = 0,
    allowed: list[str] = TAUNATIVE,
):
    mechanisms = [_.upper() for _ in mechanism]
    print(f"Will only remove strategies from mechanisms {mechanisms}")
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
            process(s, d, mechanisms, norun, allowed)
    else:
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for s, d in track(zip(paths, dsts), total=len(paths)):
                futures.append(
                    pool.submit(process, s, d, mechanisms, norun, allowed)
                )
            print(f"[bold blue]Will work on {len(paths)} domains[/bold blue]")
            for f in track(as_completed(futures), total=len(futures)):
                f.result()


if __name__ == "__main__":
    typer.run(main)
