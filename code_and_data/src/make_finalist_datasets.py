from pathlib import Path

import typer
from helpers.utils import STRATEGY_NAME_MAP, get_all_anac_agents
from rich import print


def adjust(x: str) -> str:
    if x == "AgentI":
        return "GAgent"
    x = STRATEGY_NAME_MAP.get(x, x)
    if not isinstance(x, str):
        return x[-1]
    return x


def getagents(y, finalists_only: bool, genius10: bool, winners_only: bool):
    return [
        adjust(_.split(":")[1])
        for _ in get_all_anac_agents(
            y,
            finalists_only=finalists_only,
            genius10=genius10,
            winners_only=winners_only,
        )
    ]


def main(
    year: list[int] = [],
    base: Path = Path.cwd() / "serverclean",
    verbose: bool = False,
):
    if not year:
        year = [2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018]
    for y in year:
        agents = set(
            getagents(y, finalists_only=False, genius10=False, winners_only=False)
        )
        genius10 = set(
            getagents(y, finalists_only=False, genius10=True, winners_only=False)
        )
        finalists = set(
            getagents(y, finalists_only=True, genius10=False, winners_only=False)
        )
        non_finalists = agents.difference(finalists)
        if not non_finalists:
            continue

        print(f"{y} ({len(non_finalists)}):{non_finalists=}")
        if verbose:
            print(f"\n\t{y} ({len(agents)}):{agents=}")
            print(f"\n\t{y} ({len(finalists)}):{finalists=}")
            print(f"\n\t{y} ({len(genius10)}):{genius10=}")

        non_finalists = list(non_finalists)

        src = base / f"y{y}" / f"y{y}.csv"
        if not src.exists():
            print(f"[red] Cannot find {src}[/red]")
            continue
        dst = base / f"y{y}finalists" / f"y{y}finalists.csv"
        dst.parent.mkdir(exist_ok=True, parents=True)
        selected = []
        with open(src, "r") as f:
            lines = f.readlines()
            n_read = len(lines)
            print(f"\tFound {n_read} lines ", end="")
            for l in lines:
                for name in non_finalists:
                    if f",{name}," in l:
                        break
                else:
                    selected.append(l)
        n = len(selected)
        if n == n_read:
            print("", flush=True)
            continue
        print(f"keeping {n} (removing {n_read-n}).", flush=True)
        with open(dst, "w") as f:
            f.writelines(selected)


if __name__ == "__main__":
    typer.run(main)
