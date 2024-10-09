#!/usr/bin/env python3
"""Removes all incomplete domains based results."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import pandas as pd
import typer
from rich import print
from rich.progress import track
from utils import remove_incomplete, TAUVARIANTS, DTYPES


def process(
    src: Path,
    dst: Path,
    ignore_impure: bool,
    norun: bool,
):
    """Removes all runs that of given mechanisms for which not all negotiators are in the allowed list."""
    data = pd.read_csv(src, dtype=DTYPES)
    if data is None:
        print(f"[red]Failed to read {src}[/red]")
    n = len(data)
    data, incomplete, _ = remove_incomplete(data, ignore_impure=ignore_impure, ignore_strategies=TAUVARIANTS)
    domains = list(incomplete.keys())
    if norun:
        print(
            f"Will remove {n - len(data) } of {n} records keeping {len(data)} records\n"
            f"{len(domains)} Domains to be removed are {domains}"
            f"{incomplete}"
        )
        remaining = set(data.domain_name.unique())
        print(f"{len(remaining)} Remaining domains: {remaining}")
        return
    data.to_csv(dst, index=False)

def main(
    src: Path,
    dst: Path,
    norun: bool = False,
    max_cores: int = 0,
    ignore_impure: bool = False,
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
            process(s, d, ignore_impure, norun)
    else:
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for s, d in track(zip(paths, dsts), total=len(paths)):
                futures.append(
                    pool.submit(process, s, d,ignore_impure, norun)
                )
            print(f"[bold blue]Will work on {len(paths)} domains[/bold blue]")
            for f in track(as_completed(futures), total=len(futures)):
                f.result()


if __name__ == "__main__":
    typer.run(main)
