#!/usr/bin/env python

import sys
from pathlib import Path
from shutil import copy

import typer
from negmas.helpers import unique_name
from pandas.core.computation.align import Sequence
from rich import print
from utils import F0, F1
from utils import clean_frame as do_clean_frame
from utils import correct_field_types, find_runs_with_negative_advantage
from utils import remove_extra as do_remove_extra
from utils import remove_invalid_lines, remove_repeated_header


def process(
    f: Path,
    max_trials: int,
    min_trials: int,
    remove_explicit_failures: bool,
    remove_implicit_failures: bool,
    remove_invalid: bool,
    correct_types: bool,
    remove_extra: bool,
    remove_repeated: bool,
    remove_negative_advantage: bool,
    clean_frame: bool,
    valid: dict[str, Sequence],
    verbose: bool,
    exclude_domains: list[str],
    exclude_mechanisms: list[str],
    exclude_strategies: list[str],
    outcomelimit: int,
):
    print(f"Cleaning {f}")
    # clean_results(f, verbose=verbose, override=False)
    backup = Path(__file__).parent.parent.parent / "backups"
    backup.mkdir(exist_ok=True, parents=True)
    dst = backup / unique_name(f.name, rand_digits=1, sep="")
    copy(f, dst)
    nr, n, m, nn, clean = 0, 0, 0, 0, None
    if remove_repeated:
        m = remove_repeated_header(f, verbose)
        if verbose:
            print(f"Removed {m} repeated headers")
    if remove_invalid:
        nr = remove_invalid_lines(f, verbose=verbose, override=True)
        if verbose:
            print(f"Removed {nr} invalid lines")
    if correct_types:
        correct_field_types(f, verbose=verbose, override=True)
    if remove_negative_advantage:
        neg = find_runs_with_negative_advantage(f, verbose=verbose, override=True)
        if verbose and len(neg):
            neg.to_csv(f"negative_advantage_in_{f.name}", index=False)
        nn = len(neg)
        if verbose:
            print(f"Removed {nn} records because they had negative advantage")

    if remove_extra and (max_trials > 0 or min_trials > 0 or outcomelimit > 0):
        clean = do_remove_extra(
            path=f,
            max_trials=max_trials,
            min_trials=min_trials,
            remove_failed=remove_explicit_failures,
            remove_implicit_failures=remove_implicit_failures,
            verbose=verbose,
            outcomelimit=outcomelimit,
        )
        if clean is not None:
            clean.to_csv(f, index=False)
    if clean_frame:
        _, n = do_clean_frame(
            f,
            None,
            verbose,
            override=True,
            valid=valid,
            exclude_domains=exclude_domains,
            exclude_mechanisms=exclude_mechanisms,
            exclude_strategies=exclude_strategies,
        )
    if clean is None and n < 1 and nr < 1 and nn < 1 and m < 1:
        dst.unlink(missing_ok=True)


def main(
    file_name: Path,
    pattern: str = "*.csv",
    max_trials: int = sys.maxsize,
    min_trials: int = 0,
    remove_explicit_failures: bool = False,
    remove_implicit_failures: bool = False,
    remove_invalid: bool = True,
    correct_types: bool = True,
    remove_extra: bool = True,
    remove_repeated: bool = True,
    remove_negative_advantage: bool = True,
    clean_frame: bool = True,
    outcomelimit: int = 0,
    verbose: bool = True,
    exclude_domains: list[str] = [],
    exclude_mechanisms: list[str] = [],
    exclude_strategies: list[str] = [],
    f0: list[float] = F0,
    f1: list[float] = F1,
):
    if not file_name.exists():
        return
    if file_name.is_dir():
        for f in file_name.glob(pattern):
            process(
                f,
                max_trials=max_trials,
                min_trials=min_trials,
                remove_explicit_failures=remove_explicit_failures,
                remove_implicit_failures=remove_implicit_failures,
                remove_invalid=remove_invalid,
                correct_types=correct_types,
                remove_extra=remove_extra,
                clean_frame=clean_frame,
                remove_repeated=remove_repeated,
                remove_negative_advantage=remove_negative_advantage,
                valid=dict(f0=f0 + [-1], f1=f1 + [-1]),
                verbose=verbose,
                exclude_domains=exclude_domains,
                exclude_mechanisms=exclude_mechanisms,
                exclude_strategies=exclude_strategies,
                outcomelimit=outcomelimit,
            )
    else:
        # max_trials: int,
        # min_trials: int,
        # remove_explicit_failures: bool,
        # remove_implicit_failures: bool,
        # remove_invalid: bool,
        # correct_types: bool,
        # remove_extra: bool,
        # remove_repeated: bool,
        # remove_negative_advantage: bool,
        # clean_frame: bool,
        # valid: dict[str, Sequence],
        # verbose: bool,
        # exclude_domains: list[str],
        # exclude_mechanisms: list[str],
        # exclude_strategies: list[str],
        # outcomelimit: int,
        process(
            file_name,
            max_trials=max_trials,
            min_trials=min_trials,
            remove_explicit_failures=remove_explicit_failures,
            remove_implicit_failures=remove_implicit_failures,
            remove_invalid=remove_invalid,
            remove_extra=remove_extra,
            correct_types=correct_types,
            clean_frame=clean_frame,
            remove_repeated=remove_repeated,
            remove_negative_advantage=remove_negative_advantage,
            valid=dict(f0=f0 + [-1], f1=f1 + [-1]),
            verbose=verbose,
            exclude_domains=exclude_domains,
            exclude_mechanisms=exclude_mechanisms,
            exclude_strategies=exclude_strategies,
            outcomelimit=outcomelimit,
        )


if __name__ == "__main__":
    typer.run(main)
