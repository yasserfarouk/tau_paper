from pathlib import Path

import pandas as pd
import typer
from rich import print
from utils import DTYPES


def main(
    file: list[Path],
    output: Path,
    verbose: bool = False,
    pure: bool = True,
    tau: list[str] = [],
    aor: list[str] = [],
    aot: list[str] = [],
    ao: list[str] = [],
    create: bool = False,
    year: int = 0,
) -> None:
    if not create and not output.exists():
        print(
            f"[red]Output {output} does not exist and you did nto specify --create (or you specified --no-create) [/red]"
        )
        return
    x = []
    extra = []
    for file_path in file:
        if file_path.is_dir():
            extra += list(_ for _ in file_path.glob("**/*.csv"))
    file += extra
    for file_path in file:
        print(f"Reading from {file_path}")
        if file_path.is_dir():
            continue
        try:
            data = pd.read_csv(file_path, dtype=DTYPES)
            print(f"  Found {len(data)} records")
        except:
            print(f"[red] FAILED on {file_path}")
            continue
        x.append(data)  # type: ignore
    if not x:
        print(f"Nothing to copy from")
        return
    data = pd.concat(x, ignore_index=True)
    aor += ao
    aot += ao
    if pure:
        fields = ["strategy_name"]
    else:
        fields = ["first_strategy_name", "second_strategy_name"]
    results = []
    mechs = dict(AOr=aor, AOt=aot, TAU0=tau, TAU=tau)
    for mechanism, strategies in mechs.items():
        if not strategies:
            continue

        for field in fields:
            results.append(
                data.loc[
                    (data.mechanism_name == mechanism) & (data[field].isin(strategies)),
                    :,
                ]
            )
            if verbose:
                print(
                    f"Found {len(results[-1])} results for {mechanism} ({strategies}) @ {field.replace('_', ' ')}"
                )
    data = pd.concat(results)
    if verbose:
        print(f"Found {len(data)} in total")
    if output.exists():
        if verbose:
            print(f"Reading target {output}")
        orig = pd.read_csv(output, dtype=DTYPES)
        if verbose:
            print(f"Found {len(orig)} runs at the target")
        data = pd.concat((orig, data))
    if verbose:
        print(f"Will save {len(data)} records to {output}")
    output.parent.mkdir(exist_ok=True, parents=True)
    if year != 0:
        data = data.loc[data.year == year, :]
    data.to_csv(output, index=False)


if __name__ == "__main__":
    typer.run(main)
