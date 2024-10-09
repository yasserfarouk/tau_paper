from pathlib import Path

import pandas as pd
import typer
from rich import print
from utils import DTYPES, F0, F1, clean_results, remove_extra, remove_failed_records

GROUPS = [
    "domain_name",
    "f0",
    "f1",
    "perm_indx",
    "time_limit",
    "n_rounds",
    "mechanism_name",
    "strategy_name",
]


def main(
    files: list[Path],
    output: Path = Path.cwd() / "extraremoved.csv",
    max_trials: int = 0,
    min_trials: int = 0,
    remove_failures: bool = True,
    remove_implicit_failures: bool = False,
    remove_default_reserve: bool = False,
    f0: list[float] = F0,
    f1: list[float] = F1,
    verbose: bool = False,
    outcomelimit: int = 0,
) -> None:
    x = []
    extra = []
    for file_path in files:
        if file_path.is_dir():
            extra += list(_ for _ in file_path.glob("**/*.csv"))
    files += extra
    for file_path in files:
        print(f"Reading from {file_path}")
        if file_path.is_dir():
            continue
        try:
            clean_results(file_path, override=True, verbose=verbose, f0s=f0, f1s=f1)
            data = pd.read_csv(file_path, dtype=DTYPES)
            print(f"  Found {len(data)} records")
            if outcomelimit:
                data = data.loc[data.n_outcomes <= outcomelimit, :]
                print(
                    f"  Kept {len(data)} records with outcome-limit of {outcomelimit}"
                )
        except:
            print(f"[red] FAILED on {file_path}[/red]")
            continue
        x.append(data)  # type: ignore
    data = pd.concat(x, ignore_index=True)
    cols = data.columns
    if remove_default_reserve:
        data = data.loc[(data["f0"] < -0.5) | (data["f1"] < -0.5), :]
    print(f"Read {len(data)} records")
    if len(data) < 1:
        print("Nothing to do")
        return
    data = remove_failed_records(
        data, implicit=remove_implicit_failures, explicit=remove_failures
    )
    if max_trials > 0:
        data = remove_extra(
            data=data,
            max_trials=max_trials,
            min_trials=min_trials,
            remove_failed=remove_failures,
            remove_implicit_failures=remove_implicit_failures,
            verbose=verbose,
        )
    assert data is not None
    scols, snewcols = set(cols), set(data.columns)
    assert scols == snewcols, (
        f"Columns changed while removing extra trials\nDropped: "
        f"{scols.difference(snewcols)}\nAdded: {snewcols.difference(scols)}"
    )
    print(f"Will save {len(data)} records")
    output.parent.mkdir(exist_ok=True, parents=True)
    data.to_csv(output, index=False)


if __name__ == "__main__":
    typer.run(main)
