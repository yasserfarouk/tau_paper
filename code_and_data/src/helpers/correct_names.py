from pathlib import Path

import pandas as pd
import typer
from rich import print
from utils import DOMAIN_NAME_MAP, DTYPES, STRATEGY_NAME_MAP


def correct_names(
    path: Path,
    verbose: bool = False,
    dry: bool = False,
    add_year_to_domains: bool = False,
):
    try:
        data = pd.read_csv(path, dtype=DTYPES)  # type: ignore
    except Exception as e:
        print(f"[red]Failed reading {path}[/red]: {e}")
        return

    changed = dict()
    for old, new in STRATEGY_NAME_MAP.items():
        if isinstance(new, tuple):
            year, new = new
        else:
            year = 0
        if (
            old not in data.first_strategy_name.unique()
            and old not in data.second_strategy_name.unique()
        ):
            continue
        changed.update({old: new})
        data.first_strategy_name = data.first_strategy_name.astype(str)
        data.second_strategy_name = data.second_strategy_name.astype(str)
        if year >= 2010:
            data.loc[
                (data.year == year) & (data.first_strategy_name == old),
                "first_strategy_name",
            ] = new
            data.loc[
                (data.year == year) & (data.second_strategy_name == old),
                "second_strategy_name",
            ] = new
        else:
            data.loc[data.first_strategy_name == old, "first_strategy_name"] = new
            data.loc[data.second_strategy_name == old, "second_strategy_name"] = new
        data.strategy_name = data.first_strategy_name + "-" + data.second_strategy_name
        data.loc[
            data.first_strategy_name == data.second_strategy_name, "strategy_name"
        ] = data.loc[
            data.first_strategy_name == data.second_strategy_name, "first_strategy_name"
        ]
        data.strategy_name = data.strategy_name.astype("category")
        data.first_strategy_name = data.first_strategy_name.astype("category")
        data.second_strategy_name = data.second_strategy_name.astype("category")
    for old, new in DOMAIN_NAME_MAP.items():
        if old not in data.domain_name.unique():
            continue
        changed.update({old: new})
        data.domain_name = data.domain_name.astype(str)
        data.loc[data.domain_name == old, "domain_name"] = new
        data.domain_name = data.domain_name.astype("category")
    if add_year_to_domains:
        data.domain_name = data.domain_name.astype("str")
        data.loc[
            (data.domain_name.str.startswith("domain"))
            & (data.domain_name.str.contains("@")),
            "domain_name",
        ] = (
            data.loc[
                (data.domain_name.str.startswith("domain"))
                & (data.domain_name.str.contains("@")),
                "domain_name",
            ]
            .str.split("@")
            .str[0]
        )
        data.loc[
            (data.domain_name.str.startswith("domain"))
            & (~data.domain_name.str.contains("@"))
            & (data.year.isin((2021, 2022))),
            "domain_name",
        ] = (
            data.loc[
                (data.domain_name.str.startswith("domain"))
                & (~data.domain_name.str.contains("@"))
                & (data.year.isin((2021, 2022))),
                "domain_name",
            ]
            + "@"
            + data.loc[
                (data.domain_name.str.startswith("domain"))
                & (~data.domain_name.str.contains("@"))
                & (data.year.isin((2021, 2022))),
                "year",
            ].astype(str)
        )
        data.domain_name = data.domain_name.astype("category")
        changed = True
    if changed:
        if not dry:
            data.to_csv(path, index=False)
        if verbose:
            print(f"Name corrections in {path}\n\t{changed=}")
            print(f"{data.strategy_name.unique()}")


def main(
    files: list[Path],
    verbose: bool = False,
    dry: bool = False,
    add_year_to_domains: bool = False,
) -> None:
    extra = []
    for file_path in files:
        if file_path.is_dir():
            extra += list(
                _ for _ in file_path.glob("**/*.csv") if "misisng" not in _.name
            )
    files += extra
    for file_path in files:
        if file_path.is_dir():
            continue
        correct_names(
            file_path, verbose=verbose, dry=dry, add_year_to_domains=add_year_to_domains
        )


if __name__ == "__main__":
    typer.run(main)
