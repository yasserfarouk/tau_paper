import math
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from rich import print

warnings.filterwarnings("ignore")

from helpers.utils import (
    BASE_FOLDER,
    MAX_TRIALS,
    MIN_TRIALS,
    NONSTATS,
    PRECISION,
    STATS,
    STATS_AGENT,
    STATS_AGENT_INDIVIDUAL,
    STATS_FINAL,
    STATS_FINAL_INDIVIDUAL,
    STATS_TIMING,
    TAUNATIVE,
    TAUVARIANTS,
    do_all_tests,
    filter_dominated_out,
    filter_topn,
    make_latex_table,
    read_data,
    remove_failed_records,
    remove_runs,
)

EXTENSIONS = ("pdf",)
matplotlib.use("TkAgg")
plt.rcParams["figure.figsize"] = (20, 20)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:.2}".format
plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.rcParams.update({"font.size": 30})
# plt.rcParams.update({"font.labelsize": 30})
matplotlib.rcParams["legend.fontsize"] = 20

RESERVED = ["Rational1", "Rational2"]
# STATS = ["Utility", "Welfare", "Rounds", "P. Dist", "N. Dist", "Time", "AR"]
LARGESPACE = 1.0


def make_order(values):
    order = sorted(values)
    tmp = [_ for _ in order if not any(s in _ for s in ("WAR", "CAB"))]
    tmp2 = []
    for s in order:
        for _ in ("CAB", "WAR"):
            if _ in s:
                tmp2.append(s)
    order = tmp2 + tmp
    return order


def plot_and_save_combined(
    data,
    save,
    output,
    agreements_only,
    show,
    agent_view,
    lbl,
    path,
    invertx=False,
    topn=0,
    filter_dominated=False,
    shorten_names=False,
    keep_all_equal=True,
    override=False,
):
    extended_labels = STATS_AGENT_INDIVIDUAL if agent_view else STATS_FINAL_INDIVIDUAL
    if topn > 0:
        output += f"top{topn}"
    else:
        output += "all"
    if filter_dominated:
        output += "-dom_no"
    else:
        output += "-dom_yes"
    if agent_view:
        output += "-agent"
    else:
        output += "-designer"
    if agreements_only:
        output += "-disagree_no"
    else:
        output += "-disagree_yes"
    if lbl:
        output += f"-{lbl}"
    else:
        output += "-all"
    filename = path / output
    filename.parent.mkdir(parents=True, exist_ok=override)

    def _do_plot(ax, x, y, save):
        df = data.copy()
        if y in STATS_TIMING:
            df = remove_failed_records(df, explicit=True, implicit=True)
        if topn > 0:
            df = filter_topn(
                df,
                topn,
                largest=y not in STATS_TIMING,
                stat=y,
                keep_all_equal=keep_all_equal,
                groups="Protocol" if shorten_names else "Mechanism",
                # groups = "Mechanism",
            ).copy()
        # if shorten_names:
        #     df[x] = df[x].str.split("+").str[-1]
        order = make_order(df[x].unique())

        g = None
        try:
            g = sns.barplot(
                ax=ax,
                data=df,
                x=x,
                y=y,
                order=order,
                # errorbar="se",
                errorbar=("ci", 90),
            )
            ax.tick_params(axis="x", labelrotation=90, labelsize=30)
            if stat in ("Rounds", "Time"):
                ax.set(yscale="log")
            if invertx:
                ax.invert_xaxis()
            ax.set_xlabel("")
            if save:
                for ext in EXTENSIONS:
                    plt.savefig(
                        filename.parent
                        / f"{filename.name}-{y.replace(' ', '').lower()}.{ext}",
                        bbox_inches="tight",
                    )
        except:
            pass
        return g

    for stat in extended_labels:
        _, ax = plt.subplots(figsize=(20, 10))
        figM = plt.get_current_fig_manager()
        figM.resize(*figM.window.maxsize())
        _do_plot(ax, "Condition", stat, save=save)
    if save:
        for ext in EXTENSIONS:
            plt.savefig(
                filename.parent / f"{filename.name}-stats.{ext}",
                bbox_inches="tight",
            )

    if show:
        plt.show()


def plot_and_save(
    data,
    save,
    output,
    agreements_only,
    show,
    agent_view,
    path,
    reserved_label="RationalFraction",
    lines=True,
    nlegendcols=2,
    invertx=False,
    topn=0,
    filter_dominated=False,
    shorten_names=False,
    individual=False,
    keep_all_equal=True,
    exclude_timing=False,
    override=False,
    all_individual=True,
):
    if shorten_names:
        nlegendcols += 1
    stats_table = STATS_AGENT if agent_view else STATS_FINAL
    extended_labels = STATS_AGENT_INDIVIDUAL if agent_view else STATS_FINAL_INDIVIDUAL
    if exclude_timing:
        stats_table = [_ for _ in stats_table if _ not in STATS_TIMING]
    # filter_labels = [ _ for _ in list(set(extended_labels).union(set(stats_table))) if _ not in (STATS_TIMING) ]
    plot = sns.lineplot if lines else sns.barplot
    if topn:
        output += f"top{topn}"
    else:
        output += "all"
    if filter_dominated:
        output += f"-dom_no"
    else:
        output += "-dom_yes"
    if agent_view:
        output += f"-agent"
    else:
        output += "-designer"
    if lines:
        output += f"-lines"
    else:
        output += "-bars"
    if agreements_only:
        output += f"-disagree_no"
    else:
        output += f"-disagree_yes"
    filename = path / (
        f"{output}{reserved_label.lower().replace(' ', '') if reserved_label != 'RationalFraction' else ''}"
    )
    filename.parent.mkdir(parents=True, exist_ok=override)

    def _do_plot(ax, x, y, i, save, ncols=2):
        df = data.copy()
        if y in STATS_TIMING:
            df = remove_failed_records(df, explicit=True, implicit=True)
        if topn:
            df = filter_topn(
                df,
                topn,
                stat=y,
                largest=y not in STATS_TIMING,
                keep_all_equal=keep_all_equal,
                groups="Protocol" if shorten_names else "Mechanism",
                # groups="Mechanism",
            ).copy()
        # if shorten_names:
        #     df["Condition"] = df["Condition"].str.split("+").str[-1]
        order = make_order(df["Condition"].unique())
        g = None
        try:
            g = plot(
                ax=ax,
                data=df,
                x=x,
                y=y,
                hue="Condition",
                hue_order=order,
                errorbar=("ci", 90),
                # errorbar="se",
            )
            ax.tick_params(axis="x", labelsize=30)
            if i is not None:
                if topn == 0 or filter_dominated:
                    if i == 0:
                        handles, labels = ax.get_legend_handles_labels()
                        fig.legend(
                            handles, labels, loc="upper center", ncol=nlegendcols
                        )
                    g.legend_.remove()  # type: ignore
                else:
                    # xx = -1 if i % 2 == 0 else 1
                    sns.move_legend(
                        ax, "lower left", ncol=ncols, bbox_to_anchor=(0.0, -1.05)
                    )
                    # xx = -1 if i % 2 == 0 else 1
                    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            if y in ("Rounds", "Time"):
                ax.set(yscale="log")
            if invertx:
                ax.invert_xaxis()
            if save:
                for ext in EXTENSIONS:
                    plt.savefig(
                        filename.parent
                        / f"{filename.name}-{y.replace(' ', '').lower()}.{ext}",
                        bbox_inches="tight",
                    )
        except:
            pass
        return g

    # if filter_dominated:
    #     data = filter_dominated_out(data, stats=filter_labels)
    if individual:
        for stat in extended_labels:
            fig, ax = plt.subplots(figsize=(30, 10))
            figM = plt.get_current_fig_manager()
            figM.resize(*figM.window.maxsize())
            _do_plot(ax, reserved_label, stat, None, save=save)
            fig.legend(loc="upper center", ncol=nlegendcols, bbox_to_anchor=(0.5, 1.05))
    if all_individual:
        for i, stat in enumerate(stats_table):
            r, c = i // 2, i % 2
            fig = plt.figure(
                figsize=(20 if filter_dominated or not shorten_names else 28, 15)
            )
            figM = plt.get_current_fig_manager()
            figM.resize(*figM.window.maxsize())
            ax = fig.subplots()
            _do_plot(ax, reserved_label, stat, i, save=False, ncols=3)
            if save:
                for ext in EXTENSIONS:
                    plt.savefig(
                        filename.parent / f"{filename.name}-r-{stat}.{ext}",
                        bbox_inches="tight",
                    )
            if show:
                plt.show()
        return
    fig = plt.figure(figsize=(20 if filter_dominated or not shorten_names else 28, 20))
    figM = plt.get_current_fig_manager()
    figM.resize(*figM.window.maxsize())
    axs = fig.subplots(
        int(math.ceil(len(stats_table) / 2)), 2, sharex=True, squeeze=False
    )
    if topn == 0 or filter_dominated:
        plt.subplots_adjust(hspace=0)
    else:
        plt.subplots_adjust(hspace=1.05)

    for i, stat in enumerate(stats_table):
        r, c = i // 2, i % 2
        _do_plot(axs[r, c], reserved_label, stat, i, save=False)

    if save:
        for ext in EXTENSIONS:
            plt.savefig(
                filename.parent / f"{filename.name}-stats.{ext}",
                bbox_inches="tight",
            )

    if show:
        plt.show()


def plot_summary_graphs(
    data,
    save,
    output,
    agreements_only,
    show,
    agent_view,
    path,
):
    pass


def main(
    files: list[Path] = [],
    remove_failures: bool = True,
    remove_implicit_failures: bool = False,
    rounds: bool = True,
    timelimit: bool = True,
    agreements_only: bool = False,
    results_path: Path = BASE_FOLDER,
    output: str = "figs",
    max_trials: int = MAX_TRIALS,
    min_trials: int = MIN_TRIALS,
    insignificant: bool = False,
    significant: bool = True,
    allstats: bool = False,
    stats: bool = True,
    ttests: bool = True,
    figs: bool = True,
    figs_individual: bool = False,
    figs_line: bool = True,
    figs_bar: bool = True,
    show: bool = True,
    save: bool = True,
    rational_only: bool = False,
    exceptions: bool = False,
    precision: int = PRECISION,
    pure_only: bool = False,
    impure_only: bool = False,
    include: list[str] = list(),
    exclude: list[str] = list(),
    tau_exclude: list[str] = list(),
    include_first: list[str] = list(),
    include_second: list[str] = list(),
    exclude_first: list[str] = list(),
    exclude_second: list[str] = list(),
    mechanism: list[str] = None,  # type: ignore
    exclude_mechanism: list[str] = None,  # type: ignore
    permutations: bool = True,
    year: int = 0,
    xlabel="RationalFraction",
    invertx=False,
    topn: int = 0,
    filter_dominated: bool = False,
    bilateral: bool = True,
    multilateral: bool = True,
    adapted: bool = True,
    tau_variants: bool = True,
    pure_tau: bool = False,
    ignore_second: bool = False,
    ignore_first: bool = False,
    shorten_names: bool = False,
    keep_all_equal: bool = True,
    agent_view: bool = True,
    designer_view: bool = True,
    exclude_timing: bool = False,
    exclude_impure: list[str] = [],
    remove_incomplete_domain: bool = False,
    verbose: bool = False,
    original: bool = True,
    controlled: bool = True,
    override: bool = False,
    augment_path: bool = True,
    sep: str = "/",
    remove_incompatible_reserves: bool = True,
):
    if precision > 0:
        pd.set_option("display.precision", precision)
        pd.options.display.float_format = f"{{:.{precision}f}}".format
    if figs:
        (results_path / "figs").mkdir(exist_ok=True, parents=override)
    if stats:
        (results_path / "tables").mkdir(exist_ok=True, parents=override)
    if ttests:
        (results_path / "ttests").mkdir(exist_ok=True, parents=override)
    ignore_params = dict(
        ignore_impure=pure_only,
        ignore_impure_for=("TAU0", "TAU"),
    )
    params = dict(
        failures=remove_failures,
        implicit_failures=remove_implicit_failures,
        rounds=rounds,
        timelimit=timelimit,
        agreements_only=agreements_only,
        max_trials=max_trials,
        min_trials=min_trials,
        pure_only=pure_only,
        impure_only=impure_only,
        include_first=include_first,
        include_second=include_second,
        exclude_first=exclude_first,
        exclude_second=exclude_second,
        include=include,
        exclude=exclude,
        tau_exclude=tau_exclude,
        permutations=permutations,
        year=year,
        original_reserved_values=True,
        controlled_reserved_values=True,
        rational_only=rational_only,
        bilateral=bilateral,
        multilateral=multilateral,
        ignore_second_agent=ignore_second,
        filter_dominated=filter_dominated,
        remove_incomplete_domains=remove_incomplete_domain,
        remove_incomplete_params=ignore_params,
        verbose=verbose,
        mechanisms=mechanism,
        exclude_mechanisms=exclude_mechanism,
        explicit_types=False,
        remove_incompatible_reserves=remove_incompatible_reserves,
    )
    data, data_agent = read_data(files, **params)
    if not len(data):
        print(f"Found no data: Most likely every domain failed for someone")
        params["verbose"] = True
        data, data_agent = read_data(files, **params)
        return
    if year > 0:
        data = data.loc[data.Year == year, :]
    if filter_dominated:
        n = len(data)
        filter_labels = [
            _
            for _ in list(set(STATS_AGENT_INDIVIDUAL).union(set(STATS_AGENT)))
            if _ not in (STATS_TIMING)
        ]
        data_agent = filter_dominated_out(data_agent, filter_labels, groups="Mechanism")
        filter_labels = [
            _
            for _ in list(set(STATS_FINAL_INDIVIDUAL).union(set(STATS_FINAL)))
            if _ not in (STATS_TIMING)
        ]
        data = filter_dominated_out(data, filter_labels, groups="Mechanism")
        print(
            f"Filtered out {n - len(data)} of {n} (keeping {(len(data)/n):4.1%}) records because they correspond to dominated strategies"
        )
    if pure_tau:
        data = data.loc[
            (~data.Mechanism.str.contains("TAU"))
            | (data["First Strategy"] == data["Second Strategy"])
            | (
                ~(
                    (data["First Strategy"].isin(TAUNATIVE))
                    & (data["Second Strategy"].isin(TAUNATIVE))
                )
            ),
            :,
        ]
    if exclude_impure:
        for mech in exclude_impure:
            data = data.loc[
                (data["Mechanism"].str.contains(mech))
                | (data["First Strategy"] == data["Second Strategy"]),
                :,
            ]
    assert (
        agreements_only
        and len(data.succeeded.unique()) == 1
        or not agreements_only
        and len(data.succeeded.unique()) <= 2
    ), f"{agreements_only=}, {data.succeeded.unique()=}"

    if data is None:
        print("No negotiation records")
    else:
        print(f"Total {len(data)} negotiation records.")
    if verbose:
        if data is not None:
            cols = ["Domain", "Mechanism", "First Strategy", "Second Strategy", "Year"]
            if verbose:
                cols += ["Condition"]
            for name in cols:
                x = data[name].unique()
                print(f"\t[bold]{name}s[/bold]: {len(x)} differnet values")
                if verbose or len(x) < 20:
                    print(f"\t{x}")
    if data_agent is None:
        print("No agent performance records")
    else:
        print(f"Total {len(data_agent)} agent performance records.")

    data = data[
        STATS
        + [
            "Controlled",
            "Strategy",
            "Protocol",
            "Mechanism",
            "Strategy1",
            "Strategy2",
            "MaxRounds",
            "agreement_ranks",
            "agreement_utils",
            "failed",
            "succeeded",
            "run_time",
            "steps",
            "agreement",
            # "time_limit",
        ]
        + NONSTATS
        + RESERVED
    ]
    data["time"] = data["Time"]
    if not agent_view and not designer_view:
        raise ValueError(f"Cannot pass --no-agent-view and --no-designer-view")
    if not bilateral and not multilateral:
        raise ValueError(f"Cannot pass --no-agent-view and --no-designer-view")
    if not adapted:
        data = remove_runs(data, mechanisms=["TAU0", "AU0"], allowed=TAUNATIVE)
    if not tau_variants:
        data = remove_runs(
            data, mechanisms=["TAU0", "AU0"], disallowed=TAUVARIANTS, allowed=[]
        )
    if not agent_view and not designer_view:
        raise ValueError(f"Cannot pass --no-agent-view and --no-designer-view together")
    if augment_path:
        output = f"figs/{output}"
        if year:
            output += f"{sep}y{year}"
        else:
            output += f"{sep}all_years"
        if not agent_view:
            output += f"{sep}designerview"
            data_agent = None
        if not designer_view:
            output += f"{sep}agentview"
            data = None
        if pure_tau:
            output += f"{sep}taupure"
        else:
            output += f"{sep}tauimpure"
        if not adapted:
            output += f"{sep}no_adapted"
        else:
            output += f"{sep}with_adapted"
        if not tau_variants:
            output += f"{sep}no_variants"
        else:
            output += f"{sep}with_variants"
        if not bilateral and not multilateral:
            raise ValueError(
                "Cannot pass --no-bilateral and --no-multilateral together"
            )
        elif not bilateral:
            output += f"{sep}multilateral"
        elif not multilateral:
            output += f"{sep}bilateral"
        else:
            output += f"{sep}alllateral"
        if ignore_first and ignore_second:
            raise ValueError("Cannot pass --ignore-first and --ignore-second together")
        elif ignore_second:
            output += f"{sep}first_agent"
        elif ignore_first:
            output += f"{sep}second_agent"
        else:
            output += f"{sep}all_agents"
        if shorten_names:
            output += f"{sep}short"
        else:
            output += f"{sep}long"
        if remove_incomplete_domain and remove_implicit_failures:
            output += f"{sep}dom_comp_expt_not_fail"
        if remove_incomplete_domain and not remove_implicit_failures:
            output += f"{sep}dom_comp_expt_fail"
        if not remove_incomplete_domain and remove_implicit_failures:
            output += f"{sep}dom_all_expt_not_fail"
        if not remove_incomplete_domain and not remove_implicit_failures:
            output += f"{sep}dom_all_expt_fail"
        output += sep
    output_tables = output.replace("fig", "table")
    output_ttest = output.replace("fig", "stat")

    views = []
    if agent_view:
        views.append(True)
    if designer_view:
        views.append(False)
    for agent_view in views:
        stats_table = STATS_AGENT if agent_view else STATS_FINAL
        what = "agent" if agent_view else "final"
        alldata = data_agent if agent_view else data
        if not controlled and alldata is not None:
            alldata = alldata.loc[~alldata.Controlled, :]
        if not original and alldata is not None:
            alldata = alldata.loc[alldata.Controlled, :]
        assert (
            alldata is not None
        ), f"You specified {agent_view=}, {designer_view=} and yet we found None in {'data_agent' if agent_view else 'data'}. This should never happen"
        if original:
            df = alldata.loc[~alldata.Controlled, :]
            if len(df):
                if ttests:
                    do_all_tests(
                        df,
                        insignificant,
                        allstats,
                        path=Path(results_path),
                        basename=f"{output_ttest}/original",
                        significant=significant,
                        exceptions=exceptions,
                        stats=stats_table,
                        precision=precision,
                    )
                if stats:
                    basename = f"{output_tables}/original"
                    make_latex_table(
                        df,
                        Path(results_path) / basename,
                        count=True,
                        what=what,
                    )
                if figs:
                    plot_and_save_combined(
                        df,
                        save,
                        output,
                        agreements_only,
                        show,
                        agent_view,
                        lbl="original",
                        invertx=invertx,
                        topn=topn,
                        filter_dominated=filter_dominated,
                        shorten_names=shorten_names,
                        keep_all_equal=keep_all_equal,
                        path=results_path,
                        override=True,
                    )
        if controlled:
            df = alldata.loc[alldata.Controlled, :]
            if len(df):
                if ttests:
                    do_all_tests(
                        df,
                        insignificant,
                        allstats,
                        path=Path(results_path),
                        basename=f"{output}/controlled",
                        significant=significant,
                        exceptions=exceptions,
                        stats=stats_table,
                        precision=precision,
                    )
                if stats:
                    basename = f"{output}/controlled"
                    make_latex_table(
                        df,
                        Path(results_path) / basename,
                        count=True,
                        what=what,
                    )
                if figs:
                    if figs_bar:
                        plot_and_save(
                            df,
                            save,
                            output,
                            agreements_only,
                            show,
                            agent_view,
                            lines=False,
                            reserved_label=xlabel,
                            invertx=invertx,
                            topn=topn,
                            filter_dominated=filter_dominated,
                            shorten_names=shorten_names,
                            individual=figs_individual,
                            keep_all_equal=keep_all_equal,
                            exclude_timing=exclude_timing,
                            path=results_path,
                            override=True,
                        )
                    if figs_line:
                        try:
                            plot_and_save(
                                df,
                                save,
                                output,
                                agreements_only,
                                False,
                                agent_view,
                                lines=True,
                                reserved_label=xlabel,
                                invertx=invertx,
                                topn=topn,
                                filter_dominated=filter_dominated,
                                shorten_names=shorten_names,
                                individual=figs_individual,
                                keep_all_equal=keep_all_equal,
                                exclude_timing=exclude_timing,
                                path=results_path,
                                override=True,
                            )
                        except:
                            pass
                    plot_and_save_combined(
                        df,
                        save,
                        output,
                        agreements_only,
                        show,
                        agent_view,
                        lbl="controlled",
                        invertx=invertx,
                        topn=topn,
                        filter_dominated=filter_dominated,
                        shorten_names=shorten_names,
                        keep_all_equal=keep_all_equal,
                        path=results_path,
                        override=True,
                    )
                for f in df[xlabel].unique():
                    basename = f"{output_ttest}-{str(f).replace('.', '')}"
                    x = df.loc[df[xlabel] == f, :]
                    if ttests:
                        do_all_tests(
                            x,
                            insignificant,
                            allstats,
                            path=Path(results_path),
                            basename=basename,
                            significant=significant,
                            exceptions=exceptions,
                            stats=stats_table,
                            precision=precision,
                        )
                    if stats:
                        make_latex_table(
                            x,
                            Path(results_path) / basename,
                            count=True,
                            perdomain=False,
                            what=what,
                        )
                        make_latex_table(
                            x,
                            Path(results_path) / f"{basename}/perdomain",
                            count=True,
                            perdomain=True,
                            what=what,
                        )


if __name__ == "__main__":
    typer.run(main)
