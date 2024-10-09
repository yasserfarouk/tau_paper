#!/usr/bin/env python3

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import typer
import yaml
from negmas.helpers.inout import dump
from negmas.preferences.ops import make_rank_ufun
from rich import print
from rich.progress import track
from utils import RANK_SCENARIOS_FOLDER, SCENARIOS_FOLDER, get_dirs, load_scenario

from negmas import Scenario


def smaller(path: Path):
    if not path.is_dir():
        return sys.maxsize
    try:
        return int(path.name[:7])
    except:
        return sys.maxsize


def convert_keys_to_tuples(path: Path):
    for f in path.glob("*.yml"):
        with open(f, "r") as ff:
            lines = ff.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.replace("? [", "? !!python/tuple [")
        with open(f, "w") as ff:
            ff.writelines(lines)


def process(
    d: Path, base: Path, dest: Path, verbose: bool, n_checks: int, f_checks: float
):
    if not d.is_dir():
        return
    try:
        scenario = Scenario.from_genius_folder(d, ignore_discount=True)
    except:
        return
    if scenario is None:
        return
    if verbose:
        print(f"{d}")
    ufuns = scenario.ufuns
    ordinal = [make_rank_ufun(_) for _ in ufuns]
    for u, o in zip(ufuns, ordinal):
        o.name = u.name
    new_path = dest / d.relative_to(base)
    new_path.mkdir(parents=True, exist_ok=True)
    scenario = Scenario(
        scenario.agenda,
        tuple(ordinal),
        scenario.mechanism_type,
        scenario.mechanism_params,
    )
    # new_path /= f"{new_path.name}.yaml"
    new_path.mkdir(parents=True, exist_ok=True)
    typ = "yml"
    serialized = scenario.serialize()
    dump(
        serialized["domain"],
        new_path / f"{serialized['domain']['name']}.{typ}",
        compact=True,
    )
    for u in serialized["ufuns"]:
        for k, v in u.items():
            if isinstance(v, np.floating):
                u[k] = float(v)
        if "mapping" in u.keys():
            for i, j in u["mapping"].items():
                if isinstance(j, np.floating):
                    u["mapping"][i] = float(j)
                if isinstance(j, np.integer):
                    u["mapping"][i] = int(j)
        with open(new_path / f"{u['name']}.{typ}", "w") as f:
            yaml.safe_dump(u, f, default_flow_style=True, sort_keys=False)
    convert_keys_to_tuples(new_path)
    # dump(scenario.serialize(), new_path)
    if f_checks > 1e-3 and scenario.agenda.is_discrete():
        n_checks += min(1, int(0.5 + scenario.agenda.cardinality * f_checks))
    if not n_checks:
        return
    new_scenario = load_scenario(new_path)
    assert new_scenario is not None, f"Could not load"
    assert new_scenario.agenda is not None, f"Could not read agenda"
    assert (
        new_scenario.agenda.cardinality == scenario.agenda.cardinality
    ), f"{scenario.agenda.cardinality=}\n{new_scenario.agenda.cardinality=}"
    assert len(new_scenario.ufuns) == len(
        scenario.ufuns
    ), f"{len(new_scenario.ufuns)=}, {len(scenario.ufuns)=}"
    assert (
        new_scenario.agenda == scenario.agenda
    ), f"{scenario.agenda=}\n{new_scenario.agenda=}"
    for u1, u2 in zip(scenario.ufuns, new_scenario.ufuns):
        assert (
            u1.reserved_value == u2.reserved_value
        ), f"{u1.reserved_value=}, {u2.reserved_value=}"
    outcomes = scenario.agenda.sample(n_checks, False, False)
    for outcome in outcomes:
        for u1, u2 in zip(scenario.ufuns, new_scenario.ufuns):
            assert (
                abs(u1(outcome) - u2(outcome)) < 1e-6
            ), f"Outcome {outcome} => Original ufun: {u1(outcome)}, loaded {u2(outcome)}"


def main(
    base: Path = SCENARIOS_FOLDER,
    dest: Path = RANK_SCENARIOS_FOLDER,
    verbose: bool = True,
    n_checks: int = 0,
    f_checks: int = 0,
    max_cores: int = 0,
    ndomains: int = sys.maxsize,
    outcomelimit: int = 0,
    order: bool = True,
    reversed: bool = False,
    norun: bool = False,
):
    base = base.absolute()
    dirs = get_dirs(base, outcomelimit, ndomains, order=order, reversed=reversed)
    print(f"Will work on {len(dirs)} directories at {base}")
    if norun:
        print(dirs)
        return
    futures = []
    if max_cores < 0:
        for dir in track(dirs):
            process(dir, base, dest, verbose, n_checks, f_checks)
    else:
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for dir in track(dirs):
                futures.append(
                    pool.submit(process, dir, base, dest, verbose, n_checks, f_checks)
                )
            print(f"[bold blue]Will work on {len(dirs)} domains[/bold blue]")
            for f in track(as_completed(futures), total=len(futures)):
                f.result()


if __name__ == "__main__":
    typer.run(main)
