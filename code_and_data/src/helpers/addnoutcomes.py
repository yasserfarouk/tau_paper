#!/usr/bin/env python
from pathlib import Path
from shutil import move

import typer
from negmas.inout import Scenario
from rich import print

from utils import SCENRIOS_PATH


def isint(x):
    try:
        int(x)
        return True
    except:
        return False


def addnoutcomes(f: Path, path: Path):
    if not f.is_dir():
        return
    if isint(f.name[:7]):
        return
    print(str(f))
    try:
        domain = Scenario.from_genius_folder(
            f, ignore_discount=True, ignore_reserved=False
        )
    except Exception as e:
        print(f"[red]{str(f.relative_to(path))}[/red]: {str(e)}")
        return
    if domain is None:
        return
    n = len(list(domain.agenda.enumerate_or_sample()))
    move(f, (f.parent / f"{n:07d}{f.name}"))


def main(path: Path = SCENRIOS_PATH):
    path = path.absolute()
    for f in path.glob("**/*"):
        addnoutcomes(f, path)


if __name__ == "__main__":
    typer.run(main)
