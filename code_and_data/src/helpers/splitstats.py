#!/usr/bin/env python3
"""Split stats file to a folder"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import typer
from negmas.helpers.inout import dump, load
from rich import print
from rich.progress import track
from utils import (
    BASIC_STATS_FILE,
    SCENRIOS_PATH,
    STATSBASE,
    STATSFILE,
    adjustname,
    get_dirs,
)


def process(
    f: Path,
    base: Path,
    verbose: bool,
    override=True,
    remove_original=False,
):
    _ = override
    f, base = Path(f).absolute(), Path(base).absolute()
    if verbose:
        print(f"[yellow]Started Processing {str(f.relative_to(base))}[/yellow]")
    file = f / STATSFILE
    if not file.exists():
        if verbose:
            print(f"[red]File not found: {file}[/red]")
        return

    dst = f / STATSBASE
    dst.mkdir(parents=True, exist_ok=True)
    stats = load(file)

    dd = dict()
    for k, v in stats.items():
        if isinstance(v, dict):
            dump(v, dst / adjustname(k))
            continue
        dd[k] = v
    if dd:
        dump(dd, dst / BASIC_STATS_FILE)
    if remove_original:
        os.unlink(file)
    if verbose:
        print(f"[green]Done Processing {str(f.absolute().relative_to(base))}[/green]")


def main(
    base=SCENRIOS_PATH,
    verbose: bool = False,
    norun: bool = False,
    override: bool = True,
    max_cores: int = 0,
    ndomains: int = sys.maxsize,
    outcomelimit: int = 0,
    order: bool = True,
    reversed: bool = False,
    remove_original: bool = False,
):
    print(f"Discovering scenarios in {base}")
    dirs = get_dirs(base, outcomelimit, ndomains, order=order, reversed=reversed)
    print(f"Read a total of {len(dirs)} domains")
    if norun:
        print(dirs)
        return
    futures = []
    if max_cores < 0:
        for dir in track(dirs, description="Splitting ..."):
            process(dir, base, verbose, override, remove_original)
    else:
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for dir in track(dirs, description="Submitting ..."):
                futures.append(
                    pool.submit(process, dir, base, verbose, override, remove_original)
                )
            print(f"[bold blue]Will work on {len(dirs)} domains[/bold blue]")
            for f in track(
                as_completed(futures), total=len(futures), description="Splitting ..."
            ):
                f.result()


if __name__ == "__main__":
    typer.run(main)
