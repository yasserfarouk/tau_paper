#!/usr/bin/env python3
"""Adds statistics file for all domains."""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from shutil import move

import typer
from negmas.helpers.inout import dump
from negmas.inout import Scenario, load_genius_domain_from_folder
from rich import print
from rich.progress import track
from utils import F0, F1, SCENRIOS_PATH, get_dirs, get_stats

DUMMY_RUN = False

NSAMPLES = 100_000
VLARGE_SPACE = 10_000_000_000


def isint(x):
    try:
        int(x)
        return True
    except:
        return False


def addnoutcomes(f: Path, path: Path, verbose: bool = True):
    if not f.is_dir():
        return False
    if isint(f.name[:7]):
        return False
    try:
        domain = Scenario.from_genius_folder(
            f, ignore_discount=True, ignore_reserved=False
        )
    except Exception as e:
        print(
            f"[yellow]Cannot add n. outcomes to {str(f.relative_to(path))}[/yellow]: {str(e)}"
        )
        return False
    if domain is None:
        return False
    if verbose:
        print(f"[yellow] Start Outcomes to {str(f.relative_to(path))}[/yellow]")
    n = len(list(domain.agenda.enumerate_or_sample()))
    move(f, (f.parent / f"{n:07d}{f.name}"))
    return True


def add_stats(dir: Path, base: Path, verbose, f0, f1, override):
    try:
        scenario = load_genius_domain_from_folder(dir)
        ufuns, issues = scenario.ufuns, scenario.agenda
        d = get_stats(
            os=issues,
            ufuns=ufuns,
            dir=dir,
            base=base,
            verbose=verbose,
            f0s=f0,
            f1s=f1,
            override=override,
        )
        dump(d, dir / "stats.json")
        return d
    except Exception as e:
        print(
            f"[red]Cannot add stats to {dir.relative_to(base)}[/red]: {str(e)}",
            flush=True,
        )
        raise e


def normalize(f: Path, base: Path, verbose: bool = True, override=False):
    domain = Scenario.from_genius_folder(f, ignore_discount=True, ignore_reserved=False)
    if domain is None:
        return False
    try:
        if not override and domain.is_normalized():
            if verbose:
                print(f"[yellow]Already normalized {str(f.relative_to(base))}[/yellow]")
            return True
        domain.normalize()
        if not domain.is_normalized():
            print(f"[red]Cannot normalize {str(f.relative_to(base))}[/red]")
            return False
        domain.to_genius_folder(f)
    except Exception as e:
        print(f"[red]Cannot normalize {str(f.relative_to(base))}[/red]: {str(e)}")
        return False
    return True


def process(
    f: Path,
    base: Path,
    verbose: bool,
    override=True,
    f0: list[float] = F0,
    f1: list[float] = F1,
):
    f, base = Path(f).absolute(), Path(base).absolute()
    if verbose:
        print(f"[yellow]Started Processing {str(f.relative_to(base))}[/yellow]")
    if verbose:
        print("\tN. Outcomes", end=" ")
    addnoutcomes(f, base, verbose)
    if verbose:
        print("Normalizing", end=" ")
    if not normalize(f, base, verbose, override=override):
        return
    if verbose:
        print("Stats", end=" ")
    if not add_stats(f, base, verbose, f0, f1, override=override):
        return
    if verbose:
        print(f"[green]Done Processing {str(f.absolute().relative_to(base))}[/green]")


def main(
    base=SCENRIOS_PATH,
    verbose: bool = False,
    norun: bool = False,
    f0: list[float] = F0,
    f1: list[float] = F1,
    override: bool = True,
    max_cores: int = 0,
    ndomains: int = sys.maxsize,
    outcomelimit: int = 0,
    minoutcomes: int = 0,
    order: bool = True,
    reversed: bool = False,
):
    print(f"Preparing scenarios in {base}")
    dirs = get_dirs(
        base,
        outcomelimit,
        ndomains,
        order=order,
        reversed=reversed,
        minoutcomes=minoutcomes,
    )
    print(f"Read a total of {len(dirs)} domains")
    if norun:
        print(dirs)
        return
    futures = []
    if max_cores < 0:
        for dir in track(dirs, description="Preparing ..."):
            process(dir, base, verbose, override, f0, f1)
    else:
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for dir in track(dirs, description="Submitting ..."):
                futures.append(
                    pool.submit(process, dir, base, verbose, override, f0, f1)
                )
            print(f"[bold blue]Will work on {len(dirs)} domains[/bold blue]")
            for f in track(
                as_completed(futures), total=len(futures), description="Preparing ..."
            ):
                f.result()


if __name__ == "__main__":
    typer.run(main)
