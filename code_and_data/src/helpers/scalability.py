#!/usr/bin/env python3

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
from rich import print
from utils import FIGS_FOLDER, STATS_TIMING, read_data

matplotlib.use("TkAgg")
plt.rcParams["figure.figsize"] = (20, 20)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:.2}".format
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.rcParams.update({"font.size": 30})
# plt.rcParams.update({"font.labelsize": 30})
matplotlib.rcParams["legend.fontsize"] = 20

TIME_LIMIT = "time_limit"


def main(
    files: list[Path],
    tau: list[str] = ["WAR", "CAB"],
    ao: list[str] = ["Boulware", "MiCRO", "Atlas3", "AgentK", "NiceTfT"],
    ao_pure: bool = False,
    output: Path = FIGS_FOLDER / "scalability",
    save: bool = True,
    show: bool = True,
    verbose: bool = True,
    ci: int = 90,
    logx: bool = None,  # type: ignore
    logy: bool = None,  # type: ignore
    xaxis: list[str] = ["Outcomes", "RationalFraction"],
    rounds: bool = True,
    timelimit: bool = True,
    remove_incomplete_domains: bool = True,
    pure_only: bool = False,
    line_plot: bool = True,
    condition: str = "Mechanism",
    base_name: bool = False,
):
    if "all" in tau:
        tau = []
    if "all" in ao:
        ao = []
    output.mkdir(parents=True, exist_ok=True)
    removeparams = (
        dict(
            use_base_mechanism_name=base_name,
        )
        if not tau
        else dict(
            ignore_all_but_strategies_for=dict(TAU=["WAR", "CAB"]),
            use_base_mechanism_name=base_name,
        )
    )

    data, _ = read_data(
        files,
        failures=False,
        implicit_failures=False,
        verbose=verbose,
        remove_incomplete_domains=remove_incomplete_domains,
        pure_only=pure_only,
        remove_incomplete_params=removeparams,
    )
    print(f"Read {len(data)} data points")
    if not len(data):
        print(f"[red]No Data[/red]")
        return
    print(f"N. Rounds found: {data.Rounds.unique()}")
    print(f"Time Limits found: {data.time_limit.unique()}")
    extra = ""
    if not rounds:
        # data = data.loc[(data["Rounds"].isna()) | (data["Rounds"] <= 0), :]
        data = data.loc[
            ~((data[TIME_LIMIT].isna()) | (data[TIME_LIMIT] <= 0))
            | data.Strategy.isin(tau)
            | data.Mechanism.str.startswith("TAU"),
            :,
        ]
        print(f"Removing rounds-limited runs")
        extra += "nr"
    if not timelimit:
        data = data.loc[
            (data[TIME_LIMIT].isna())
            | (data[TIME_LIMIT] <= 0)
            | (data.Strategy.isin(tau))
            | data.Mechanism.str.startswith("TAU"),
            :,
        ]
        print(f"Removing time-limited runs")
        extra += "nt"
    if not len(data):
        print(f"[red]No Data[/red]")
        return
    if not rounds or not timelimit:
        print(f"New N. Rounds found: {data.Rounds.unique()}")
        print(f"New Time Limits found: {data.time_limit.unique()}")
    tau = data.loc[data.Mechanism.str.startswith("TAU"), "Strategy"].unique()
    ao = data.loc[~data.Mechanism.str.startswith("TAU"), "Strategy"].unique()
    if ao_pure:
        ao = [
            _
            for _ in data.loc[data.Mechanism.str.startswith("AOP"), "Strategy"].unique()
            if "-" not in _
        ]
        print(f"Will use {len(ao)} strategies for AOP:\n{ao}")

    ao_mechs = data.loc[data.Mechanism.str.startswith("AOP"), "Mechanism"].unique()
    strategies = {"TAU": tau}
    for a in ao_mechs:
        strategies[a] = ao
    print(f"Will use the following conditions: {list(strategies.keys())}")
    data["Utility"] = data["Welfare"]
    data["Utility/Second"] = data["Welfare"] / data["Time"]
    data["Utility/Round"] = data["Welfare"] / data["Rounds"]
    STATS = STATS_TIMING + ["Utility", "Utility/Second", "Utility/Round"]

    cols = [condition] + xaxis + STATS
    x = []
    for m, s in strategies.items():
        x.append(
            data.loc[data.Mechanism.str.startswith(m) & data.Strategy.isin(s), cols]
        )
        print(f"Found {len(x[-1])} records for mechanism {m}: {s}")
    data = pd.concat(x)
    data["Outcomes"] = data["Outcomes"].astype(int)
    print(f"Will use {len(data)} data points")
    print(f"N Outcomes options: [{sorted(data['Outcomes'].unique())}]")
    if not len(data):
        print(f"[red]No Data[/red]")
        return
    plotter = sns.lineplot if line_plot else sns.pointplot
    pparams = dict() if line_plot else dict(dodge=True)
    n = len(data)
    data = data.dropna()
    if len(data) != n:
        print(f"Dropped {n - len(data)} records containing NA values")
    for x in xaxis:
        if logx is None:
            logx_ = x in ["Outcomes"]
        else:
            logx_ = logx
        for y in STATS:
            logy_ = logy if logy is not None else not y == "Utility"
            _, ax = plt.subplots(figsize=(20, 10), squeeze=True)
            figM = plt.get_current_fig_manager()
            figM.resize(*figM.window.maxsize())
            try:
                plotter(
                    data=data,
                    x=x,
                    y=y,
                    hue=condition,
                    ax=ax,
                    errorbar=("ci", ci),
                    **pparams,
                )
                if logy_:
                    ax.set(yscale="log")  # type: ignore
                if logx_:
                    ax.set(xscale="log")  # type: ignore
                if save:
                    for ext in ("pdf", "png"):
                        plt.savefig(
                            output
                            / f"{x.lower().replace('.','').replace(' ', '').replace('/', '_')}-{y.replace('/', '_')}{extra}{'-line' if line_plot else '-point'}.{ext}",
                            bbox_inches="tight",
                        )
                if show:
                    plt.show()
            except Exception as e:
                print(f"[red] Failed for {x=}, {y=} [/red]: {e}")
                continue


if __name__ == "__main__":
    typer.run(main)
