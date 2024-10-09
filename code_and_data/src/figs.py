import warnings
from json import dump
from pathlib import Path

import pandas as pd
import seaborn as sns
import typer
from helpers.utils import TAUNATIVE, factorial_test, read_data, refresh_values
from matplotlib import pyplot as plt
from pandas.errors import DtypeWarning, SettingWithCopyWarning
from rich import print

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 3)

warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_context("talk", rc={"legend.fontsize": 24})
sns.set_style("darkgrid")

BASE = Path(__file__).absolute().parent.parent / "figs"
VALID_DATASETS_ORDERED = [
    "final",
    "2010",
    "2011",
    "2012",
    "2013",
    "2015",
    "2016",
    "2017",
    "2018",
]
VALID_DATASETS = set(VALID_DATASETS_ORDERED)

XCOND = ("Size", "Difficulty", "Year")


def update_categories(data):
    for k in data.columns:
        if data[k].dtype != "category":
            continue
        data[k] = data[k].cat.remove_unused_categories()
    return data


def savefig(name: str, exts=("pdf",), base=BASE, name_stem=""):
    for ext in exts:
        path = base / f"{name_stem}{name}.{ext}"
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(
            path,
            bbox_inches="tight"
            # bbox_extra_artists=(plt.legend(),)
        )


# filename = "serverclean/final/final.csv"
# filename = "serverclean/y2010/y2010.csv"
# filename = "serverclean/y2011/y2011.csv"
# filename = "serverclean/y2012/y2012.csv"
# filename = "serverclean/y2013/y2013.csv"
# filename = "serverclean/y2015/y2015.csv"
# filename = "serverclean/y2016/y2016.csv"

# filename = "serverclean/y2019/y2019.csv" # no - data
# filename = "serverclean/y2017/y2017.csv"
# filename = "serverclean/y2018/y2018.csv"
# filename = "serverclean/y2020/y2020.csv"
# filename = "serverclean/y2021/y2021.csv"
# filename = "serverclean/y2022/y2022.csv"


def process(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "final",
    years: tuple[int] = tuple(),
    no_adapters: bool = True,
    rename_cab: bool = False,
    tau_pure: bool = True,
    remove_war: bool = True,
    with_agreements_only: bool = False,
    remove_negative_advantage: bool = False,
    no_cabwar: bool = False,
    use_average: bool = True,
    remove_missing_tau: bool = True,
    implicit_failures: bool = False,
    failures: bool = False,
    tests: bool = True,
    scatter: bool = True,
    bars: bool = True,
    verbose: bool = False,
    name_stem="",
):
    if dataset not in VALID_DATASETS:
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    peryear = not dataset.startswith("final")
    controlled_only, original_only = not peryear, peryear
    remove_irrational_agents = peryear
    filename = (
        f"serverclean/{dataset}/{dataset}.csv"
        if dataset.startswith("final")
        else f"serverclean/y{dataset}/y{dataset}.csv"
    )
    files = [BASE.parent / filename]
    data, agents = read_data(
        files, implicit_failures=implicit_failures, failures=failures
    )
    BASE_NAME = f"{files[0].stem}/" if files[0].stem != "final" else "final/"
    OTHERS_LABEL = "Average" if use_average else "Others"
    STRATEGY_TYPE = "Strategy Type"
    QUALITY_PRIVACY = "Quality/Privacy Balance"
    n_original, n_agents_original = len(data), len(agents)

    if years:
        data = data.loc[data.Year.isin(years), :]
        if verbose:
            print(f"Data: {len(data)}, Agents: {len(agents)} for years: {years}")

    if controlled_only:
        data = data.loc[data.Controlled, :]
        agents = agents.loc[agents.Controlled, :]
        if verbose:
            print(f"Data: {len(data)}, Agents: {len(agents)} controlled")
    if original_only:
        data = data.loc[~data.Controlled, :]
        agents = agents.loc[~agents.Controlled, :]
        if verbose:
            print(f"Data: {len(data)}, Agents: {len(agents)} original")
    if with_agreements_only:
        data = data.loc[data.succeeded, :]
        agents = agents.loc[agents.succeeded, :]
        if verbose:
            print(f"Data: {len(data)}, Agents: {len(agents)} with agreements")
    if verbose:
        print(data.groupby("Mechanism").size())
    if tau_pure:
        n = len(data)
        data.Strategy = data.Strategy.astype(str)
        data.Strategy1 = data.Strategy1.astype(str)
        data = data.loc[
            (data.Mechanism != "TAU") | (data.Strategy == data.Strategy1), :
        ]
        data = refresh_values(data)
        assert "TAU" in data.Mechanism.unique()
        data.Strategy = data.Strategy.astype("category")
        data.Strategy1 = data.Strategy1.astype("category")

        agents.Strategy = agents.Strategy.astype(str)
        agents.StrategyPartner = agents.StrategyPartner.astype(str)
        agents = agents.loc[
            (agents.Mechanism != "TAU") | (agents.Strategy == agents.StrategyPartner), :
        ]
        agents.Mechanism = agents.Mechanism.cat.remove_unused_categories()
        assert "TAU" in agents.Mechanism.unique()
        agents.Strategy = agents.Strategy.astype("category")
        agents.StrategyPartner = agents.StrategyPartner.astype("category")

        if verbose:
            print(
                f"TAU Pure: Removed {n - len(data)} records Remaining {len(data)} records"
            )
            print(f"Data: {len(data)}, Agents: {len(agents)} Pure TAU")
            print(data.groupby("Mechanism").size())
    if remove_war:
        n = len(data)
        data = data.loc[((data.Strategy1 != "WAR") & (data.Strategy2 != "WAR")), :]
        data = refresh_values(data)
        agents = agents.loc[
            ((agents.Strategy != "WAR") & (agents.StrategyPartner != "WAR")), :
        ]
        agents = refresh_values(agents)
        assert "TAU" in data.Mechanism.unique()
        if verbose:
            print(
                f"Remove WAR: Removed {n - len(data)} records Remaining {len(data)} records"
            )
            print(f"Data: {len(data)}, Agents: {len(agents)} not WAR")
            print(data.groupby("Mechanism").size())

    if no_cabwar:
        data = data.loc[
            (~data.Strategy.str.contains("CAB")) & (~data.Strategy.str.contains("WAR")),
            :,
        ]
        agents = agents.loc[
            (~agents.Strategy.str.contains("CAB"))
            & (~agents.Strategy.str.contains("WAR"))
            & (~agents.StrategyPartner.str.contains("CAB"))
            & (~agents.StrategyPartner.str.contains("WAR")),
            :,
        ]
        if verbose:
            print(f"Data: {len(data)}, Agents: {len(agents)} not CAB/WAR")
            print(data.groupby("Mechanism").size())
    if remove_irrational_agents:
        IRRATIONAL_SECOND = ("AgentK", "NiceTfT", "AgentGG", "Caduceus")
        # IRRATIONAL_SECOND = tuple()
        data = data.loc[
            (~data.Strategy2.isin(IRRATIONAL_SECOND))
            #                | (data.RationalFraction > 0.95)
            | (~data.Controlled) | (data.Advantage2 >= 1e-4),
            :,
        ]
        agents = agents.loc[
            (~agents.Strategy.isin(IRRATIONAL_SECOND)) | (agents.Starting)
            #                | (agents.RationalFraction > 0.95)
            | (~agents.Controlled) | (agents.Advantage >= 1e-4),
            :,
        ]
        if verbose:
            print(
                f"Data: {len(data)}, Agents: {len(agents)} avoiding irrational second agents"
            )
            print(data.groupby("Mechanism").size())
    if rename_cab:
        for col in (
            "Strategy",
            "Strategy1",
            "Strategy2",
            "StrategyPartner",
            "Condition",
        ):
            n = len(data)
            for df in (data, agents):
                if col not in df.columns:
                    continue
                df[col] = df[col].astype(str)
                x = df.loc[df[col].str.contains("CAB"), col].str.replace("CAB", "SCS")
                assert len(x), f"{col} collapsed when trying to rename CAB"
                df.loc[df[col].str.contains("CAB"), col] = x
                df[col] = df[col].astype("category")
            data = refresh_values(data)
            agents = refresh_values(agents)
            assert "TAU" in data.Mechanism.unique()
            if verbose:
                print(
                    f"Renaming CAB in {col}: Removed {n - len(data)} records Remaining {len(data)} records"
                )
                print(f"Data: {len(data)}, Agents: {len(agents)} CAB -> SCS")
                print(data.groupby("Mechanism").size())
    if verbose:
        print(f"Before Refreshing: {len(data)}, Agents: {len(agents)}")
    data = refresh_values(data)
    agents = refresh_values(agents)
    if verbose:
        print(f"After Refreshing: {len(data)}, Agents: {len(agents)}")
    data.Type = data.Type.cat.remove_unused_categories()
    data.Strategy = data.Strategy.cat.remove_unused_categories()
    data.Strategy1 = data.Strategy1.cat.remove_unused_categories()
    data.Strategy2 = data.Strategy2.cat.remove_unused_categories()
    data["Speed"] = 1 / data["Time"]
    agents["Speed"] = 1 / agents["Time"]
    if verbose:
        print(data.groupby("Mechanism").size())

    # check for cases with negative advantage. Should find none
    dd = data.loc[
        ~((data.Advantage1 >= 0) & (data.Advantage2 >= 0)),
        [
            "Mechanism",
            "Strategy",
            "Year",
            "Domain",
            "Advantage1",
            "Advantage2",
            "agreement_utils",
            "reserved0",
            "reserved1",
            "Rational1",
            "Rational2",
            "Strategy1",
            "Strategy2",
            "Outcomes",
        ],
    ]
    dd = update_categories(dd)
    if len(dd):
        print(f"Negative advantage in {len(dd)} cases")
        for col in ["Mechanism", "Domain", "Strategy", "Rational2", "Rational1"]:
            x = dd.groupby([col]).size()
            x = x[x > 0]
            print(x)
        dd = dd[
            [
                "Mechanism",
                "Year",
                "Domain",
                "Rational1",
                "Rational2",
                "Strategy1",
                "Strategy2",
                "Outcomes",
            ]
        ].reset_index(None)
        dd.Mechanism = dd.Mechanism.cat.rename_categories(
            {"AOP($n_o$)": "AOr", "AOP(3min)": "AOt", "TAU": "TAU0"}
        )
        if verbose:
            print(dd)
        dd.to_csv(Path.cwd().parent.parent / "negative_advantage.csv", index=False)
    if remove_negative_advantage:
        data = data.loc[(data.Advantage1 >= 0) & (data.Advantage2 >= 0), :]
    if verbose:
        print(data.groupby("Mechanism").size())

    data["Speed (Relative)"] = 1 / data["Time"]
    data[QUALITY_PRIVACY] = data.Quality * data.Privacy
    data[STRATEGY_TYPE] = "AOP-TAU"
    data.loc[
        (data.Strategy1.isin(TAUNATIVE)) & (data.Strategy2.isin(TAUNATIVE)),
        STRATEGY_TYPE,
    ] = "TAU Native"
    data.loc[
        (~data.Strategy1.isin(TAUNATIVE)) & (~data.Strategy2.isin(TAUNATIVE)),
        STRATEGY_TYPE,
    ] = "AOP Native"
    data[STRATEGY_TYPE] = data[STRATEGY_TYPE].astype("category")

    agents["Speed (Relative)"] = 1 / agents["Time"]
    agents[STRATEGY_TYPE] = "AOP-TAU"
    agents.loc[(agents.Strategy.isin(TAUNATIVE)), STRATEGY_TYPE] = "TAU Native"
    agents.loc[(~agents.Strategy.isin(TAUNATIVE)), STRATEGY_TYPE] = "AOP Native"
    agents[STRATEGY_TYPE] = agents[STRATEGY_TYPE].astype("category")

    if no_adapters:
        data = data.loc[
            ~(
                (data[STRATEGY_TYPE].str.startswith("AOP"))
                & data.Mechanism.str.startswith("TAU")
            ),
            :,
        ]
        agents = agents.loc[
            ~(
                (agents[STRATEGY_TYPE].str.startswith("AOP"))
                & agents.Mechanism.str.startswith("TAU")
            ),
            :,
        ]
    data.Type = data.Type.cat.remove_unused_categories()
    data.Strategy = data.Strategy.cat.remove_unused_categories()
    data.Strategy1 = data.Strategy1.cat.remove_unused_categories()
    data.Strategy2 = data.Strategy2.cat.remove_unused_categories()
    data[STRATEGY_TYPE] = data[STRATEGY_TYPE].cat.remove_unused_categories()
    data.groupby(STRATEGY_TYPE).size()

    # print(data.loc[data.Mechanism == "TAU", "Strategy"].unique())

    data["Advantage"] = (data.Advantage1 + data.Advantage2) / 2
    data["Optimality x Completeness"] = data.Score
    data["Overall Score"] = data.Advantage * data.DesignerScore * data.Speed
    data["Score"] = data.Advantage * data.DesignerScore
    data["Designer"] = data["DesignerScore"]
    data["Agent"] = data["Advantage"]

    print(
        f"\tRead {n_original} negotiation records ({n_agents_original} agent specific records)\n"
        f"\tWill use {len(data)} negotiation records ({len(agents)} agent specific records)\n"
        f"\t{len(data.Domain.unique())} domains, {len(data.Strategy.unique())} strategy combinations, {len(agents.Strategy.unique())} strategies\n"
    )
    print(data.groupby("Mechanism").size())

    if scatter:
        for xcond in XCOND:
            cols = ["Mechanism", "Strategy", xcond]
            mech_score = "Designer"
            agent_score = "Agent"
            dscore = data
            dscore.Type = dscore.Type.cat.remove_unused_categories()
            dscore = (
                dscore.groupby(cols)[[mech_score, agent_score]].mean().reset_index()
            )
            plt.figure()
            ax = sns.scatterplot(
                data=dscore,
                x=mech_score,
                y=agent_score,
                hue="Mechanism",
                style=xcond if not peryear else None,
            )
            if peryear:
                sns.move_legend(
                    ax,
                    "lower center",
                    bbox_to_anchor=(0.5, 1),
                    ncol=3,
                    title=None,
                    frameon=False,
                    fontsize=17,
                )
            else:
                sns.move_legend(
                    ax,
                    "upper left",
                    bbox_to_anchor=(1.05, 1.02),
                    title=None,
                    frameon=False,
                    fontsize=17,
                )
            savefig(
                f"{BASE_NAME}scatter_{xcond}_{agent_score}_{mech_score}",
                name_stem=name_stem,
            )
            plt.close()

        data["Designer"] = data["DesignerScore"]
        data["Agent"] = data["Advantage"]
        cols = ["Mechanism", "Strategy"] + (["Domain"] if peryear else [])
        mech_score = "Designer"
        agent_score = "Agent"
        dscore = data
        dscore.Type = dscore.Type.cat.remove_unused_categories()
        dscore = dscore.groupby(cols)[[mech_score, agent_score]].mean().reset_index()
        plt.figure()
        ax = sns.scatterplot(
            data=dscore,
            x=mech_score,
            y=agent_score,
            hue="Mechanism",
            style="Domain" if peryear else None,
        )
        if peryear:
            sns.move_legend(
                ax,
                "upper left",
                bbox_to_anchor=(1.05, 1.02),
                fontsize=11 if peryear else 17,
            )
        else:
            sns.move_legend(
                ax,
                "lower center",
                bbox_to_anchor=(0.5, 1),
                ncol=3,
                title=None,
                frameon=False,
                fontsize=10 if peryear else 17,
            )
        savefig(
            f"{BASE_NAME}scatter_{'agg' if not peryear else 'details'}_{agent_score}_{mech_score}",
            name_stem=name_stem,
        )
        plt.close()

    # data[" Fairness"] = data["Fairness"]
    # data["Fairness"] = data[[_ for _ in data.columns if "Fairness" in _]].min(axis=1)
    METRICS = dict(
        data=[
            "Quality",
            "Overall Score",
            "DesignerScore",
            "Score",
            "IRUB",
            "Privacy",
            "Optimality",
            "Completeness",
            "Welfare",
            "Advantage",
            "Time",
            "Rounds",
            "Speed",
            "Uniqueness",
        ]
        + [_ for _ in data.columns if _.endswith("Score")]
        + [_ for _ in data.columns if _.endswith("Fairness")],
        agents=["AgentScore", "IRUB", "Privacy", "Advantage", "Utility"],
    )
    DF = dict(data=data, agents=agents)
    # metrics = ["DesignerScore"]
    LOG_METRICS = ["Rounds", "Time"]
    MIN_METRICS = ["Time", "Rounds", "IRUB"]
    AGREEMENT_ONLY_METRICS = [_ for _ in data.columns if "Fairness" in _]
    NTOP = 2
    CAT = ("Strategy", "Strategy1", "Strategy2", "StrategyPartner", "Condition", "Type")
    plotfn = sns.barplot
    plot_params = dict(errorbar="se")
    # legend_params = dict(loc= "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    legend_params = dict(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.02),
        title=None,
        frameon=False,
        fontsize=16,
    )
    figsize = (6, 3)
    if bars:
        # ttests = {"TAU-AOP(3min)":dict(), {"TAU-AOP($n_o$)": dict()}}
        # wtests = {"TAU-AOP(3min)":dict(), {"TAU-AOP($n_o$)": dict()}}
        for dfname, metrics in METRICS.items():
            df = DF[dfname]
            for metric in metrics:
                plt.figure()
                best = dict()
                for mechanism in data.Mechanism.unique():
                    if metric in AGREEMENT_ONLY_METRICS:
                        x = df.loc[data.succeeded, :]
                    else:
                        x = df
                    x = (
                        x.loc[x.Mechanism == mechanism, :]
                        .groupby(["Mechanism", "Strategy"])[metric]
                        .mean()
                    )
                    if metric in MIN_METRICS:
                        x = x.nsmallest(NTOP)
                    else:
                        x = x.nlargest(NTOP)
                    best[mechanism] = [_[1] for _ in x.index.tolist()]
                best = list(zip(best.keys(), best.values()))
                # keep only data from top NTOP agents for AOP while keeping TAU
                cond = (df.Mechanism == best[0][0]) & (df.Strategy.isin(best[0][1]))
                for k, v in best[1:]:
                    cond |= (df.Mechanism == k) & (df.Strategy.isin(v))
                x = df.copy()
                others = (~cond) & (x.Mechanism != "TAU")
                for col in ("Strategy", "Condition", "Mechanism"):
                    x[col] = x[col].astype(str)
                x.loc[others, "Strategy"] = OTHERS_LABEL
                x.loc[others, "Condition"] = (
                    x.loc[others, "Mechanism"] + "+" + x.loc[others, "Strategy"]
                )
                for col in ("Strategy", "Condition", "Mechanism"):
                    x[col] = x[col].astype("category")
                if use_average:
                    y = x.loc[
                        (x.Mechanism != "TAU") & (x.Strategy != OTHERS_LABEL), :
                    ].copy()
                    x = pd.concat((x, y), ignore_index=True)
                # x = x.loc[cond, :]

                for col in CAT:
                    if col not in x.columns:
                        continue
                    x[col] = x[col].cat.remove_unused_categories()
                for xaxis in XCOND:
                    try:
                        plt.figure(figsize=figsize)
                        ax = plotfn(
                            data=x, x=xaxis, y=metric, hue="Condition", **plot_params  # type: ignore
                        )
                        if metric in LOG_METRICS:
                            plt.yscale("log")
                        sns.move_legend(ax, **legend_params)
                        # plt.setp(ax.get_legend().get_texts(), fontsize='10')
                        savefig(
                            f"{BASE_NAME}{dfname}_{xaxis}_{metric}", name_stem=name_stem
                        )
                        plt.close()
                    except Exception as e:
                        print(f"[red]ERROR[/red]{xaxis} ({metric}) failed with {e}")
                        plt.close()
                try:
                    plt.figure(figsize=figsize)
                    ax = plotfn(data=x, y=metric, x="Condition", **plot_params)  # type: ignore
                    if metric in LOG_METRICS:
                        plt.yscale("log")
                    plt.xticks(rotation=90)
                    ax.bar_label(ax.containers[0], fmt="%3.2f")  # type: ignore
                    savefig(f"{BASE_NAME}{dfname}_all_{metric}", name_stem=name_stem)
                    plt.close()
                except Exception as e:
                    print(f"[red]ERROR[/red]all ({metric}) failed with {e}")
                    plt.close()

        if tests:
            test_names = ["ttest", "wilcoxon", "median"]
            results = dict(zip(test_names, [dict() for _ in test_names]))
            printed = False
            for test in test_names:
                results[test] = dict()
                for dfname, metrics in METRICS.items():
                    df = DF[dfname]
                    results[test][dfname] = dict(zip(metrics, [None for _ in metrics]))
                    for stat in metrics:
                        r = factorial_test(
                            data=df,
                            stat=stat,
                            type="wilcoxon",  # ttest, median
                            allstats=False,
                            insignificant=False,
                            significant=True,
                            exceptions=True,
                            tbl_name=None,
                            # str(
                            #     BASE.parent.absolute()
                            #     / "stat"
                            #     / f"{BASE_NAME}{test}_{dfname}_{stat}"
                            # ),
                            ignore_nan_ps=True,
                            precision=3,
                            print_na=not printed,
                        )
                        if r:
                            results[test][dfname][stat] = r
                        printed = True
                    if not results[test][dfname]:
                        del results[test][dfname]
                if results[test]:
                    stats_path = BASE / "stats" / f"{BASE_NAME}{test}.json"
                    stats_path.parent.mkdir(parents=True, exist_ok=True)
                    if verbose:
                        print(f"\tSaving stats to {stats_path}")
                        print(results[test].keys())
                    with open(stats_path, "w") as f:
                        dump(
                            results[test],
                            f,
                            indent=2,
                            sort_keys=True,
                        )


def main(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "final",
    year: list[int] = [],
    adapters: bool = False,
    rename_cab: bool = False,
    tau_pure: bool = True,
    remove_war: bool = True,
    with_agreements_only: bool = False,
    remove_negative_advantage: bool = False,
    cabwar: bool = True,
    use_average: bool = True,
    remove_missing_tau: bool = True,
    implicit_failures: bool = False,
    failures: bool = False,
    tests: bool = True,
    scatter: bool = True,
    bars: bool = True,
    name_stem: str = "",
):
    if dataset not in VALID_DATASETS and dataset != "all":
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    if dataset == "all":
        for d in VALID_DATASETS_ORDERED:
            print(f"Working for dataset {d}")
            process(
                dataset=d,
                no_adapters=not adapters,
                rename_cab=rename_cab,
                tau_pure=tau_pure,
                remove_war=remove_war,
                with_agreements_only=with_agreements_only,
                remove_negative_advantage=remove_negative_advantage,
                no_cabwar=not cabwar,
                use_average=use_average,
                remove_missing_tau=remove_missing_tau,
                implicit_failures=implicit_failures,
                failures=failures,
                name_stem=name_stem,
                years=tuple(year),
            )
        return
    process(
        dataset=dataset,
        no_adapters=not adapters,
        rename_cab=rename_cab,
        tau_pure=tau_pure,
        remove_war=remove_war,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        no_cabwar=not cabwar,
        use_average=use_average,
        remove_missing_tau=remove_missing_tau,
        implicit_failures=implicit_failures,
        failures=failures,
        tests=tests,
        scatter=scatter,
        bars=bars,
        name_stem=name_stem,
        years=tuple(year),
    )


if __name__ == "__main__":
    typer.run(main)
