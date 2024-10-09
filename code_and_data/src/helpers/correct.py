#!/usr/bin/env python3

from pathlib import Path
# from shutil import copy

import typer
from rich import print
from rich.progress import track


def process(file_name: Path, verbose: bool = True):
    with open(file_name, "r") as f:
        lines = f.readlines()
    if verbose:
        print(f"{file_name.stem} Read {len(lines)} lines")
    firsts = [line.split(",")[0] for line in lines]

    def is_int(x):
        try:
            int(x)
            return True
        except:
            return False

    ints = [i for i, f in enumerate(firsts) if is_int(f)]
    if verbose:
        print(f"{file_name.stem} Has [red]{len(ints)}[/red] lines with INDEX")
    emp = [i for i, f in enumerate(firsts) if len(f.strip()) == 0]
    if verbose:
        print(f"{file_name.stem} Has [red]{len(emp)}[/red] indexed headers")
    corrected = []
    ints = set(ints)
    for i, line in enumerate(lines):
        if i in emp:
            continue
        if i > 0 and line.startswith("mechanism_"):
            continue
        if i in ints:
            line = ",".join(line.split(",")[1:])
        corrected.append(line)
    if verbose:
        print(f"{file_name.stem} Writing {len(corrected)} lines")
    with open(file_name, "w") as f:
        f.writelines(corrected)
    # for f in Path().glob("*.csv"):
    #     copy(file_name, f)


def main(file_names: list[Path], verbose: bool = True):
    added = []
    for file_path in file_names:
        if file_path.is_dir():
            added += [_ for _ in file_path.glob("*.csv")]
    files = file_names + added
    for file_path in track(files):
        if file_path.is_dir():
            continue
        process(file_path, verbose)


if __name__ == "__main__":
    typer.run(main)
