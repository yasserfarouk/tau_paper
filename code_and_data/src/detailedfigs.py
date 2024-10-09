import warnings
from json import dump
from pathlib import Path

import pandas as pd
import seaborn as sns
import typer
from figutils import (MECHANISM_TYPE, VALID_DATASETS, VALID_DATASETS_ORDERED,
                      read_and_adjust)
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
NTOP = 2
BASE = Path(__file__).absolute().parent.parent / "figs" / "ajcai"
DATA_LOC = Path(__file__).absolute().parent.parent / "serverclean"
SCENARIOS_LOC = Path(__file__).absolute().parent.parent / "scenarios"
# XCOND = ("Size", "Difficulty", "Year")
XCOND = tuple()
ALWAYS_SHOWN = {
    "AOP": ["MiCRO"],
    "TAU": ["CAB", "WAR"],
}
DESIGNER_METRIC = "DesignerScore"

METRICS = dict(
    data=[
        "OutcomeScore",
        "DesignerScore",
        "Privacy",
        "Optimality",
        "Completeness",
        "Fairness",
        "Welfare",
        "Speed",
        "N_Fairness",
        "O_N_Fairness",
        "K_Fairness",
        "O_K_Fairness",
        "RK_Fairness",
        "O_RK_Fairness",
        # "Quality",
        # "Overall Score",
        # "Score",
        # "IRUB",
        # "Advantage",
        # "Time",
        # "Rounds",
        "Uniqueness",
    ],
    agents=["AdvantagePartner", "Advantage", "Privacy", "AgentScore"],
)
# metrics = [designer_metric]
LOG_METRICS = ["Rounds", "Time"]
MIN_METRICS = ["Time", "Rounds", "IRUB"]
CAT = ("Strategy", "Strategy1", "Strategy2", "StrategyPartner", "Condition", "Type")


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


def process(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "final",
    years: tuple[int] = tuple(),
    no_adapters: bool = False,
    rename_cab: bool = True,
    tau_pure: bool = False,
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
    designer_metric=DESIGNER_METRIC,
    name_stem="",
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = True,
    privacy_factor: float = 0,
    experiment_results_path: Path = DATA_LOC,
    scenarios_path: Path = SCENARIOS_LOC,
    ntop: int = NTOP,
):
    peryear = not dataset.startswith("final")
    agents, data, base_name, filename = read_and_adjust(
        dataset=dataset,
        years=years,
        remove_aop=False,
        no_adapters=no_adapters,
        remove_mixed=False,
        rename_cab=rename_cab,
        separate_pure=False,
        separate_cabwar=False,
        tau_pure=tau_pure,
        remove_war=remove_war,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        no_cabwar=no_cabwar,
        implicit_failures=implicit_failures,
        failures=failures,
        verbose=verbose,
        multilateral=True,
        bilateral=True,
        privacy_factor=privacy_factor,
        add_opposition=False,
        experiment_results_path=experiment_results_path,
        scenarios_path=scenarios_path,
        correct=correct_reading_errors,
        overwrite=overwrite_data_files,
    )

    if agents is None:
        return
    assert (
        agents is not None
        and data is not None
        and base_name is not None
        and filename is not None
    )

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
                f"{base_name}scatter_{xcond}_{agent_score}_{mech_score}",
                name_stem=name_stem,
            )
            plt.close()

        data["Designer"] = data[designer_metric]
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
            f"{base_name}scatter_{'agg' if not peryear else 'details'}_{agent_score}_{mech_score}",
            name_stem=name_stem,
        )
        plt.close()

    # data[" Fairness"] = data["Fairness"]
    # data["Fairness"] = data[[_ for _ in data.columns if "Fairness" in _]].min(axis=1)
    DF = dict(data=data, agents=agents)
    AGREEMENT_ONLY_METRICS = [_ for _ in data.columns if "Fairness" in _]
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
    figsize = (8, 3)
    OTHERS_LABEL = "Average" if use_average else "Others"
    if bars:
        base_col = "Mechanism"
        for dfname, metrics in METRICS.items():
            df = DF[dfname]
            for metric in metrics:
                plt.figure()
                best = dict()
                for mechanism in data[base_col].unique():
                    if metric in AGREEMENT_ONLY_METRICS:
                        x = df.loc[data.succeeded, :]
                    else:
                        x = df
                    x = (
                        x.loc[x[base_col] == mechanism, :]
                        .groupby([base_col, "Strategy"])[metric]
                        .mean()
                    )
                    if metric in MIN_METRICS:
                        x = x.nsmallest(ntop)
                    else:
                        x = x.nlargest(ntop)
                    best[mechanism] = [_[1] for _ in x.index.tolist()]
                for k in best.keys():
                    for alk, vals in ALWAYS_SHOWN.items():
                        if k.startswith(alk):
                            best[k] = list(set(list(best[k]) + vals))
                best = list(zip(best.keys(), best.values()))
                # keep only data from top NTOP agents for AOP while keeping TAU
                cond = (df[base_col] == best[0][0]) & (df.Strategy.isin(best[0][1]))
                for k, v in best[1:]:
                    cond |= (df[base_col] == k) & (df.Strategy.isin(v))
                x = df.copy()
                others = ~cond  # & (x[base_col] != "TAU")
                for col in ("Strategy", "Condition", base_col):
                    x[col] = x[col].astype(str)
                x.loc[others, "Strategy"] = OTHERS_LABEL
                x.loc[others, "Condition"] = (
                    x.loc[others, base_col] + "+" + x.loc[others, "Strategy"]
                )
                for col in ("Strategy", "Condition", base_col):
                    x[col] = x[col].astype("category")
                if use_average:
                    y = x.loc[
                        # (x[base_col] != "TAU") &
                        (x.Strategy != OTHERS_LABEL),
                        :,
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
                            f"{base_name}{dfname}_{xaxis}_{metric}", name_stem=name_stem
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
                    savefig(f"{base_name}{dfname}_all_{metric}", name_stem=name_stem)
                    plt.close()
                except Exception as e:
                    print(f"[red]ERROR[/red]all ({metric}) failed with {e}")
                    plt.close()


def main(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "all",
    year: list[int] = [],
    adapters: bool = True,
    rename_cab: bool = False,
    tau_pure: bool = False,
    remove_war: bool = False,
    with_agreements_only: bool = False,
    remove_negative_advantage: bool = False,
    cabwar: bool = True,
    use_average: bool = True,
    remove_missing_tau: bool = True,
    implicit_failures: bool = False,
    failures: bool = False,
    tests: bool = False,
    scatter: bool = True,
    bars: bool = True,
    name_stem: str = "",
    designer_metric: str = DESIGNER_METRIC,
    verbose: bool = False,
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = False,
    privacy_factor: float = 0.25,
    ntop: int = NTOP,
):
    if dataset not in VALID_DATASETS and dataset != "all":
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    params = dict(
        no_adapters=not adapters,
        rename_cab=rename_cab,
        ntop=ntop,
        tau_pure=tau_pure,
        remove_war=remove_war,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        no_cabwar=not cabwar,
        use_average=use_average,
        remove_missing_tau=remove_missing_tau,
        implicit_failures=implicit_failures,
        privacy_factor=privacy_factor,
        failures=failures,
        tests=tests,
        scatter=scatter,
        bars=bars,
        name_stem=name_stem,
        designer_metric=designer_metric,
        years=tuple(year),
        verbose=verbose,
        correct_reading_errors=correct_reading_errors,
        overwrite_data_files=overwrite_data_files,
    )
    if dataset == "all":
        for d in VALID_DATASETS_ORDERED:
            print(f"Working for dataset {d}")
            process(dataset=d, **params)
        return
    process(dataset=dataset, **params)


if __name__ == "__main__":
    typer.run(main)
