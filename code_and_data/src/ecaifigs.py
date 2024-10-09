import warnings
from collections import defaultdict
from json import dump
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from figutils import (
    DATA_LOC,
    DIFFICULTY,
    MECHANISM_TYPE,
    OPPOSITION,
    SCENARIOS_LOC,
    STRATEGY_TYPE,
    VALID_DATASETS,
    VALID_DATASETS_ORDERED,
    read_and_adjust,
)
from helpers.utils import factorial_test
from matplotlib import pyplot as plt
from negmas.helpers.inout import load
from negmas.inout import Scenario
from negmas.preferences.ops import opposition_level
from pandas.errors import DtypeWarning, SettingWithCopyWarning
from rich import print

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 3)

warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("darkgrid")

MOVE_LEGEND_UP = True
BASE = Path(__file__).absolute().parent.parent / "figs" / "ecai"
# XCOND = ["Size", DIFFICULTY, "Year", "Domain"]
XCOND = [DIFFICULTY]
# QUALITY_PRIVACY = "Quality/Privacy Balance"
OPP, RES = "opposition", "reserved_values"
PIVOT_AGENT = [
    # ["Domain", "Strategy", "StrategyPartner", "Rational", "RationalPartner"],
    # ["Domain"],
    # ["Strategy"],
    # ["Domain", "Strategy"],
    # ["Domain", "Strategy2"],
    # ["Domain", "Strategy1", "Strategy2"],
    # ["Domain", "Rational1"],
    # ["Domain", "Rational2"],
    # ["Domain", "Strategy1"],
    # ["Domain", "Rational1", "Strategy1"],
    # ["Strategy1", "Strategy2", "Rational1", "Rational2"],
    # ["Domain", "Strategy", "StrategyPartner"],
    # ["Domain", "Strategy1"],
    ["Domain", "Rational", "RationalPartner"],
    # ["Domain", "Strategy2", "Rational1", "Rational2"],
]
PIVOT_DESIGNER = [
    # ["Domain"],
    # ["Strategy"],
    # ["Domain", "Strategy"],
    # ["Domain", "Strategy2"],
    # ["Domain", "Strategy1", "Strategy2"],
    # ["Domain", "Rational1"],
    # ["Domain", "Rational2"],
    # ["Domain", "Strategy1"],
    # ["Domain", "Rational1", "Strategy1"],
    # ["Strategy1", "Strategy2", "Rational1", "Rational2"],
    # ["Domain", "Strategy1", "Strategy2"],
    # ["Domain", "Strategy1"],
    ["Domain", "Rational1", "Rational2"],
    # ["Domain", "Strategy2", "Rational1", "Rational2"],
    # ["Domain", "Strategy1", "Strategy2", "Rational1", "Rational2"],
]
BASES_AGENT = [
    # ["Domain"],
    # ["Strategy"],
    # ["Domain", "Strategy"],
    # ["Domain", "Strategy2"],
    # ["Domain", "Strategy1", "Strategy2"],
    # ["Domain", "Rational1"],
    # ["Domain", "Rational2"],
    # ["Domain", "Strategy1"],
    # ["Domain", "Rational", "Strategy"],
    # ["Strategy1", "Strategy2", "Rational1", "Rational2"],
    # ["Domain", "Strategy1", "Strategy2"],
    # ["Domain", "Strategy1", "Strategy2", "Rational1", "Rational2"],
    # ["Domain", "Strategy1"],
    ["Domain", "Rational1", "Rational2"],
    # ["Domain", "Strategy2", "Rational1", "Rational2"],
]
BASES_DESIGNER = [
    ["Domain"],
    ["Strategy"],
    ["Strategy1", "Strategy2"],
    ["Domain", "Strategy1", "Strategy2"],
    ["Domain", "Strategy1", "Strategy2", "Rational1", "Rational2"],
]

OPPORDER = ["Low", "Medium", "High", "Extreme"]
DIFFORDER = ["Few", "Some", "Many", "All"]

METRICS = dict(
    designer=[
        "DesignerScore",
        "OutcomeScore",
        "Privacy",
        "Optimality",
        "Completeness",
        "Fairness",
        "N_Fairness",
        "K_Fairness",
        "Welfare",
        "Advantage",
        "Time",
        "Rounds",
        "Speed",
        "Uniqueness",
    ],
    agent=["AgentScore", "Privacy", "Advantage", "AdvantagePartner"],
)
TEST_METRICS = METRICS
BAR_METRICS = dict(
    designer=[
        "DesignerScore",
        "OutcomeScore",
        "Optimality",
        "Welfare",
        "Completeness",
        "Fairness",
        "N_Fairness",
        "K_Fairness",
        "Speed",
        "Uniqueness",
    ],
    agent=["AgentScore", "Privacy", "Advantage", "AdvantagePartner"],
)
# metrics = ["DesignerScore"]
TWO_SIDED = []
LESS = ["Privacy"]
MIN_METRICS = ["Time", "Rounds", "IRUB"]
DESIGNER_STATS = (
    "DesignerScore",
    "OutcomeScore",
    "Welfare",
    "Optimality",
    "Completeness",
    "Fairness",
    "N_Fairness",
    "K_Fairness",
    "O_K_Fairness",
    "O_N_Fairness",
    "Speed",
    "Rounds",
    "Uniqueness",
)
AGENT_STATS = ("AgentScore", "Privacy", "Advantage", "AdvantagePartner")
# DETAILED_STATS = (OPPOSITION, DIFFICULTY, "Size")
DETAILED_STATS = (DIFFICULTY,)
MAX_X_AXIS = 30
MAX_BAR_LABELS = 10
MAX_X_NO_ROTATION = 6
MAX_STYLE_LEGEND = 8
LOG_METRICS = ["Rounds", "Time"]
ROTATION_METRICS = ["Domain"]


def update_categories(data):
    for k in data.columns:
        if data[k].dtype != "category":
            continue
        data[k] = data[k].cat.remove_unused_categories()
    return data


X = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12)
xs = {
    "2010": (0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0),
}


def rational(os, ufuns):
    n = 0
    for outcome in os.enumerate():
        if all(not ufun.is_worse(outcome, None) for ufun in ufuns):
            n += 1
    return round(n / os.cardinality, 1)


def plot_hists(
    bases,
    metric,
    mech,
    df,
    splitter=None,
    hists_only=True,
    improvement=True,
    relative=False,
    with_title=False,
    move_legend_up=MOVE_LEGEND_UP,
    close=False,
    relative_log=True,
    nativ_only=False,
):
    if nativ_only:
        df = df.loc[df[STRATEGY_TYPE] == "AOP Native", :]
    if splitter:
        vals = df[splitter].unique()
        for v in vals:
            succeeded = plot_hists(
                bases=bases,
                metric=metric,
                mech=mech,
                splitter=None,
                df=df.loc[df[splitter] == v, :],
                hists_only=True,
                with_title=False,
                move_legend_up=move_legend_up,
                close=close,
            )
            if not succeeded:
                print(
                    f"[red]Hist Error[/red] Cannot plot histogram with splitter {splitter} on value {v} because there are no TAU runs"
                )
                continue
            if with_title:
                plt.title(f"{metric} for {splitter}: {v}")
            if close:
                plt.close()
        return True
    if isinstance(bases, str):
        bases = [bases]
    else:
        bases = list(bases)
    bases_name_added = False
    if len(bases) > 1:
        bases_name = "-".join(bases)
        df[bases_name] = df[bases[0]].astype(str)
        for bb in bases[1:]:
            df[bases_name] += df[bb].astype(str)  # type: ignore
        df[bases_name] = df[bases_name].astype("category")
        bases = [bases_name]
    else:
        bases_name = bases[0]
    x = (
        df.groupby(bases + [mech])[[metric]]
        .mean()
        .reset_index()
        .pivot_table(values=metric, index=bases_name, columns=mech)
        .reset_index()
    )
    mm = "Improvement" if improvement else "Loss"
    tau_cols = ["TAU", "TAU (Adapted)", "TAU (Mixed)", "TAU (Native)"]
    if all(_ not in x.columns for _ in tau_cols):
        if bases_name_added:
            x = x.drop(columns=(bases_name,))
        return False
    for tau_col in tau_cols:
        if tau_col not in x.columns:
            continue
        for col in x.columns:
            if col in bases + [mech] + [tau_col]:
                continue
            dst = f"{tau_col} {mm} over {col}"
            x[dst] = x[tau_col] - x[col]
            if relative:
                x.loc[x[col] > 0.0000001, dst] = (
                    x.loc[x[col] > 0.0000001, dst] / x.loc[x[col] > 0.0000001, col]
                )
            if not hists_only:
                plt.figure()
                ax = sns.barplot(data=x, x=bases_name, y=dst, errorbar="se")
                if len(x[bases].unique()) > MAX_X_AXIS:
                    ax.set_xticks([], color="white")  # type: ignore
                else:
                    plt.xticks(rotation=90)
                if metric in LOG_METRICS or (relative and relative_log):
                    plt.yscale("log")
                if with_title:
                    plt.title(metric)
                if close:
                    plt.close()

    plt.figure()
    colors = sns.color_palette()
    i = 0
    plt.axvline(x=0, color="red", linestyle=":")
    for tau_col in tau_cols:
        for col in x.columns:
            if not col.startswith(f"{tau_col} {mm}"):
                continue
            ax = sns.histplot(data=x, x=col, kde=True, label=col)  # , align="edge")
            plt.axvline(x=x[col].mean(), color=colors[i % len(colors)])
            plt.xlabel("")
            if move_legend_up:
                plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1))
            else:
                plt.legend()
            if metric in LOG_METRICS or (relative and relative_log):
                plt.xscale("log")
            if with_title:
                plt.title(metric)

            i += 1
    if close:
        plt.close()
    if bases_name_added:
        x = x.drop(columns=(bases_name,))
    return True


def process(
    dataset: str,
    years: tuple[int, ...],
    no_adapters: bool,
    remove_mixed: bool,
    rename_cab: bool,
    separate_pure: bool,
    separate_cabwar: bool,
    tau_pure: bool,
    remove_war: bool,
    remove_cab_vs_war: bool,
    with_agreements_only: bool,
    remove_negative_advantage: bool,
    no_cabwar: bool,
    use_average: bool,
    remove_incomplete_domains: bool,
    implicit_failures: bool,
    failures: bool,
    tests: bool,
    scatter: bool,
    bars: bool,
    verbose: bool,
    name_stem,
    multilateral: bool,
    bilateral: bool,
    privacy_factor: float,
    add_opposition: bool,
    hists: bool,
    hist_details: bool,
    score_bars: bool,
    weight_effect: bool,
    experiment_results_path: Path,
    scenarios_path: Path,
    test_col: str,
    fill_nas: bool,
    drop_nas: bool,
    save_latex: bool,
    alternative: str,
    relative: bool,
    remove_aop: bool,
    metrics=METRICS,
    test_metrics=TEST_METRICS,
    designer_stats=DESIGNER_STATS,
    bar_metrics=BAR_METRICS,
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = False,
    bar_labels: bool = True,
):
    agents, data, base_name, filename = read_and_adjust(
        dataset=dataset,
        years=years,
        remove_aop=remove_aop,
        no_adapters=no_adapters,
        remove_mixed=remove_mixed,
        rename_cab=rename_cab,
        separate_pure=separate_pure,
        separate_cabwar=separate_cabwar,
        tau_pure=tau_pure,
        remove_war=remove_war,
        remove_cab_vs_war=remove_cab_vs_war,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        no_cabwar=no_cabwar,
        remove_incomplete_domains=remove_incomplete_domains,
        implicit_failures=implicit_failures,
        failures=failures,
        verbose=verbose,
        multilateral=multilateral,
        bilateral=bilateral,
        privacy_factor=privacy_factor,
        add_opposition=add_opposition,
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

    if not test_col:
        test_col = "Mechanism" if no_cabwar or remove_aop else MECHANISM_TYPE
    print(f"Final records per {test_col}: data")
    print(data.groupby(test_col)["Agent"].count().reset_index())
    if verbose:
        print("Final records per Mechanism: data")
        print(data.groupby("Mechanism")["Agent"].count().reset_index())
        print("Final scores per strategy: agents")
        print(agents.groupby("Strategy")["Agent"].count().reset_index())

    def savefig(
        name: str, exts=("pdf",), base=Path(BASE) / base_name, name_stem=name_stem
    ):
        path = base.absolute()
        name = name.replace(" ", "_")
        for ext in exts:
            if name_stem:
                path = path / name_stem
            path = path / f"{name}.{ext}"
            path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(
                path,
                bbox_inches="tight",
                # bbox_extra_artists=(plt.legend(),)
            )

    others_label = "Average" if use_average else "Others"
    AGREEMENT_ONLY_METRICS = [_ for _ in data.columns if "Fairness" in _]
    NTOP = 2
    CAT = [
        "Strategy",
        "Strategy1",
        "Strategy2",
        "StrategyPartner",
        "Condition",
    ]  # , "Type"
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
    DF = dict(designer=data, agent=agents)
    sns.set_context("talk", rc={"legend.fontsize": 24})
    mechanism_order = sorted([_ for _ in data[test_col].unique()])
    sns.set_context("poster")
    figsize = (9, 3)
    if add_opposition:
        try:
            data["FullDomain"] = data.Domain.astype(str) + "@" + data.Year.astype(str)
            agents["FullDomain"] = (
                agents.Domain.astype(str) + "@" + agents.Year.astype(str)
            )
            data[OPP], agents[OPP] = np.nan, np.nan
            domains = data.FullDomain.unique()
            outcomes = [
                data.loc[data.FullDomain == _, "Outcomes"].min() for _ in domains
            ]
            years = tuple(data.loc[data.FullDomain == _, "Year"].max() for _ in domains)
            domain_names = [
                f"y{y}/{a:07}{b.split('@')[0]}"
                for y, a, b in zip(years, outcomes, domains)
            ]
            base_path = Path(scenarios_path)
            oppositions = []

            for folder, o, d, y in zip(
                [base_path / x for x in domain_names], outcomes, domains, years
            ):
                dname = d.split("@")[0]
                orig_added = False
                scenario = Scenario.from_genius_folder(folder)
                assert scenario is not None
                ufuns = scenario.ufuns
                # opposition = opposition_level(ufuns=scenario.ufuns, issues=scenario.outcome_space.issues)
                sfile = folder / "stats.json"
                if sfile.exists():
                    stats = load(sfile)
                else:
                    sfolder = folder / "stats"
                    stats = dict()
                    for f in sfolder.glob("*.json"):
                        stats[f.name] = load(f)
                oppo = float("inf")
                if "_base" not in stats:
                    print(f"Cannot find base stats for {folder}")
                else:
                    v = stats["_base"]
                    reservedo = v[RES]
                    if "cardinal" in v.keys():
                        v = v["cardinal"]
                    f0o = rational(scenario.outcome_space, [scenario.ufuns[0]])
                    f1o = rational(scenario.outcome_space, scenario.ufuns[1:])
                    oppo = v[OPP] / sqrt(2)

                    data.loc[(data.Domain == dname) & (data.Year == y), OPP] = oppo
                    agents.loc[(agents.Domain == dname) & (agents.Year == y), OPP] = (
                        oppo
                    )
                    if OPP in v:
                        orig_added = True
                        oppositions.append(
                            dict(
                                outcomes=o,
                                opposition=oppo,
                                domain=d,
                                original=True,
                                reserved0=reservedo[0],
                                reserved1=reservedo[1],
                                Rational0=f0o,
                                Rational1=f1o,
                                year=y,
                                multilateral=len(ufuns) > 2,
                                as_original=False,
                            )
                        )
                    else:
                        print(f"No original opposition for {folder}: {list(v.keys())}")
                for k, v in stats.items():
                    if k in ("basic_stats",) or k.startswith("(,-1"):
                        continue
                    if not isinstance(v, dict):
                        continue
                    if not k.startswith("("):
                        continue
                    reserved = v[RES]
                    for r, u in zip(reserved, ufuns):
                        u.reserved_value = r
                    opp = opposition_level(
                        ufuns=ufuns, issues=scenario.outcome_space.issues
                    ) / sqrt(2)
                    if "cardinal" in v.keys():
                        v = v["cardinal"]
                    if OPP not in v.keys():
                        print(f"{folder} No {OPP} in {k} {list(v.keys())=}")
                        continue
                    _, f0, f1 = eval(k)
                    data.loc[
                        (data.Domain == dname)
                        & (data.Year == y)
                        & (
                            (
                                (np.abs(data.Rational1 - f0) < 1e-3)
                                & (np.abs(data.Rational2 - f1))
                            )
                            | (
                                (np.abs(data.Rational1 - f1) < 1e-3)
                                & (np.abs(data.Rational2 - f0) < 1e-3)
                            )
                        ),
                        OPP,
                    ] = opp
                    agents.loc[
                        (agents.Domain == dname)
                        & (agents.Year == y)
                        & (
                            (
                                (np.abs(agents.Rational - f0) < 1e-3)
                                & (np.abs(agents.RationalPartner - f1))
                            )
                            | (
                                (np.abs(agents.Rational - f1) < 1e-3)
                                & (np.abs(agents.RationalPartner - f0) < 1e-3)
                            )
                        ),
                        OPP,
                    ] = opp
                    if f0 < -0.5 or f1 < -0.5:
                        continue
                    oppositions.append(
                        dict(
                            outcomes=o,
                            opposition=opp,
                            domain=d,
                            original=False,
                            reserved0=reserved[0],
                            reserved1=reserved[1],
                            Rational0=f0,
                            Rational1=f1,
                            year=y,
                            multilateral=len(ufuns) > 2,
                            as_original=orig_added and abs(opp - oppo) < 1e-3,
                        )
                    )
            data[OPPOSITION] = (
                data[OPP]
                .transform(
                    lambda x: "Unknown"
                    if np.isnan(x) or x < 0
                    else (
                        "Extreme"
                        if x >= 1 / sqrt(2)
                        else ("High" if x > 0.47 else ("Medium" if x > 0.24 else "Low"))
                    )
                )
                .astype("category")
            )
            agents[OPPOSITION] = (
                agents[OPP]
                .transform(
                    lambda x: "Unknown"
                    if np.isnan(x) or x < 0
                    else (
                        "Extreme"
                        if x >= 1 / sqrt(2)
                        else ("High" if x > 0.47 else ("Medium" if x > 0.24 else "Low"))
                    )
                )
                .astype("category")
            )
            oppdf = pd.DataFrame.from_records(oppositions)
            if len(oppdf) > 0:
                oppdf.loc[oppdf[OPP] > 1, OPP] = 1.0
                plt.figure()
                # palette = sns.color_palette('hls', len(oppdf.original.unique()))
                ax = sns.scatterplot(
                    data=oppdf.loc[~oppdf.as_original, :],
                    x="outcomes",
                    y=OPP,
                    style="multilateral",
                    hue="original",
                )
                plt.xscale("log")
                savefig("opposition_scatter")
                plt.close()
                plt.figure()
                sns.histplot(data=oppdf, y=OPP)
                try:
                    sns.move_legend(ax, loc="lower left")
                except:
                    pass
                savefig("opposition_hist")
                plt.close()
                data[DIFFICULTY].cat.rename_categories(
                    dict(zip(["Hard", "Medium", "Simple", "Trivial"], DIFFORDER))
                )
                oppdf2 = oppdf.loc[oppdf.original, :]
                if len(oppdf2) > 0 and OPP in oppdf2.columns:
                    oppdf2.loc[oppdf2[OPP] > 1, OPP] = 1.0
                    plt.figure()
                    sns.scatterplot(
                        data=oppdf2, x="outcomes", y=OPP, style="multilateral"
                    )
                    plt.xscale("log")
                    savefig("opposition_orig_scatter")
                    plt.close()
                    plt.figure()
                    sns.histplot(data=oppdf2, y=OPP)
                    savefig("opposition_orig_hist")
                    plt.close()
            for current_metric in ("AgentScore", "DesignerScore", "OutcomeScore"):
                ax = sns.lineplot(data=data, x=OPP, y=current_metric, hue=test_col)
                # ax.invert_xaxis()
                try:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1.02))
                except:
                    pass
                savefig(f"{current_metric}_opposition")
                plt.close()
                # ax.invert_xaxis()
                try:
                    ax = sns.barplot(
                        data=data,
                        x=OPPOSITION,
                        y=current_metric,
                        hue=test_col,
                        order=OPPORDER,
                        **plot_params,  # type: ignore
                    )
                    sns.move_legend(ax, **legend_params)  # type: ignore
                except:
                    pass
                # ax.bar_label(ax.containers[0], fmt="%3.2f")  # type: ignore
                savefig(f"{current_metric}_opposition_level")
                plt.close()
        except:
            pass
    sns.set_context("talk", rc={"legend.fontsize": 24})
    sns.set_context("poster")
    figsize = (6, 3)
    if tests:
        for ttest_col in (test_col, "Condition", "Mechanism"):
            agents["OldCondition"] = agents.Condition
            data["OldCondition"] = data.Condition
            agents["Condition"] = (agents[ttest_col].astype(str)).astype("category")
            data["Condition"] = (data[ttest_col].astype(str)).astype("category")
            test_names = ["ttest", "wilcoxon", "median", "ttest_ind"]
            results = dict(zip(test_names, [dict() for _ in test_names]))  # type: ignore
            printed = False
            done_saving = False
            PIVOTS = dict(designer=PIVOT_DESIGNER, agent=PIVOT_AGENT)
            # for pivot_indx in PIVOT:
            for test in test_names:
                results[test] = dict()
                stats_path = BASE / "stats" / base_name
                table_path = BASE / "tables" / base_name
                if name_stem:
                    stats_path /= name_stem
                    table_path /= name_stem
                stats_path = stats_path / ttest_col.replace(" ", "") / f"{test}.json"
                table_path = table_path / ttest_col.replace(" ", "") / f"{test}.csv"
                for dfname, metrics in test_metrics.items():
                    df = DF[dfname]
                    pivots = PIVOTS[dfname]
                    for pivot_index in pivots:
                        results[test][dfname] = dict(
                            zip(metrics, [None for _ in metrics])
                        )
                        for current_metric in metrics:
                            alternative = "two-sided"
                            if current_metric in LESS:
                                alternative = "less"
                            elif current_metric in TWO_SIDED:
                                alternative = "two-sided"
                            try:
                                r = factorial_test(
                                    data=df,
                                    stat=current_metric,
                                    type=test,
                                    allstats=False,
                                    insignificant=False,
                                    significant=True,
                                    exceptions=False,
                                    tbl_name=None
                                    if not save_latex or done_saving
                                    else str(table_path),
                                    ignore_nan_ps=True,
                                    precision=3,
                                    print_na=not printed,
                                    condition_col="Condition",
                                    fill_nas=fill_nas,
                                    drop_nas=drop_nas,
                                    pivot_index=pivot_index,
                                    alternative=alternative,
                                )
                                if r:
                                    results[test][dfname][current_metric] = r
                                printed = True
                            except Exception as e:
                                print(
                                    f"[red]Failed in {test} for {dataset} on {dfname}: {current_metric}[/red]: {e}"
                                )
                        if not results[test][dfname]:
                            del results[test][dfname]
                        if results[test]:
                            my_stats_path = (
                                stats_path.parent
                                / "_".join(pivot_index)
                                / (
                                    stats_path.stem
                                    + f"-{alternative}"
                                    + stats_path.suffix
                                )
                            )
                            my_stats_path.parent.mkdir(parents=True, exist_ok=True)
                            if verbose:
                                print(f"\tSaving stats to {my_stats_path}")
                                print(results[test].keys())
                            with open(my_stats_path, "w") as f:
                                dump(
                                    results[test],
                                    f,
                                    indent=2,
                                    sort_keys=True,
                                )
                        else:
                            print(
                                f"[red]No differences for {test} on condition {ttest_col} for any metric[/red]"
                            )
                        done_saving = False  # resave every time
            agents["Condition"] = agents.OldCondition
            data["Condition"] = data.OldCondition
            agents = agents.drop(columns=["OldCondition"])
            data = data.drop(columns=["OldCondition"])
    if weight_effect:
        XX: tuple[float, ...] = X
        for k, v in xs.items():
            if k in filename:
                XX = v
                break
        trend = defaultdict(list)
        WEGITH = "Privacy Weight"
        for w in XX:
            agents.AgentScore = (agents.Advantage + w * agents.Privacy) / (1 + w)
            d = dict(agents.groupby([test_col])[["AgentScore"]].mean().to_records())
            for k, v in d.items():
                trend[k].append(v)
        trend[WEGITH] = XX  # type: ignore
        trend = pd.DataFrame.from_dict(trend)
        plt.figure()
        for c in trend.columns:
            if c == WEGITH:
                continue
            plt.plot(trend[WEGITH], trend[c], label=c)
            plt.xlabel(WEGITH)
            plt.xscale("log")
            plt.ylabel("AgentScore")
        plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1.02))
        savefig("weight_effect")
        plt.close()
    sns.set_context("talk", rc={"legend.fontsize": 24})
    if scatter:
        figsize = (3, 3)
        for mech_score, agent_score in (
            ("DesignerScore", "AgentScore"),
            ("OutcomeScore", "AgentScore"),
            ("DesignerScore", "Advantage"),
            ("OutcomeScore", "Advantage"),
        ):
            data[mech_score] = data["DesignerScore"]
            data[agent_score] = data["Advantage"]
            for xcond in XCOND + [None]:
                try:
                    cols = [test_col, "Strategy"] + ([xcond] if xcond else [])
                    dscore = data
                    dscore.Type = dscore.Type.cat.remove_unused_categories()
                    dscore = (
                        dscore.groupby(cols)[[mech_score, agent_score]]
                        .mean()
                        .reset_index()
                    )
                    style_legend = (
                        xcond and len(dscore[xcond].unique()) < MAX_STYLE_LEGEND
                    )

                    plt.figure()
                    ax = sns.scatterplot(
                        data=dscore,
                        x=mech_score,
                        y=agent_score,
                        hue=test_col,
                        style=xcond if style_legend else None,
                    )
                    if not xcond:
                        try:
                            sns.move_legend(
                                ax,
                                "lower center",
                                bbox_to_anchor=(0.5, 1),
                                ncol=3,
                                title=None,
                                frameon=False,
                                fontsize=17,
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            sns.move_legend(
                                ax,
                                "upper left",
                                bbox_to_anchor=(1.05, 1.02),
                                title=None,
                                frameon=False,
                                fontsize=17,
                            )
                        except Exception:
                            pass
                    savefig(
                        f"scatter_{xcond if xcond else ''}_{agent_score}_{mech_score}",
                    )
                    plt.close()
                except Exception as e:
                    print(f"[red]Failed in {dataset} scatter for {xcond}[/red]: {e}")

            cols = [test_col, "Strategy"]  # + (["Domain"] if peryear else [])
            dscore = data
            dscore.Type = dscore.Type.cat.remove_unused_categories()
            dscore = (
                dscore.groupby(cols)[[mech_score, agent_score]].mean().reset_index()
            )
            plt.figure()
            try:
                ax = sns.scatterplot(
                    data=dscore,
                    x=mech_score,
                    y=agent_score,
                    hue=test_col,
                    # style="Domain" if peryear else None,
                    style=None,
                )
                try:
                    sns.move_legend(
                        ax,
                        "lower center",
                        bbox_to_anchor=(0.5, 1),
                        ncol=3,
                        title=None,
                        frameon=False,
                        # fontsize=10 if peryear else 17,
                        fontsize=10,
                    )
                except Exception:
                    pass
                savefig(
                    f"scatter_{'agg'}_{agent_score}_{mech_score}",
                )
            except Exception as e:
                print(
                    f"[red]Failed in {dataset} scatter for scatter_{dataset}_{agent_score}_{mech_score}[/red]: {e}"
                )
            finally:
                plt.close()
    sns.set_context("talk", rc={"legend.fontsize": 24})
    if bars:
        figsize = (8, 3)
        # ttests = {"TAU-AOP(3min)":dict(), {"TAU-AOP($n_o$)": dict()}}
        # wtests = {"TAU-AOP(3min)":dict(), {"TAU-AOP($n_o$)": dict()}}
        for dfname, metrics in bar_metrics.items():
            df = DF[dfname]
            for metric in metrics:
                plt.figure()
                try:
                    best = dict()
                    for mechanism in data[test_col].unique():
                        if metric in AGREEMENT_ONLY_METRICS:
                            x = df.loc[df.succeeded, :]
                        else:
                            x = df
                        x = (
                            x.loc[x[test_col] == mechanism, :]
                            .groupby([test_col, "Strategy"])[metric]
                            .mean()
                        )
                        if metric in MIN_METRICS:
                            x = x.nsmallest(NTOP)
                        else:
                            x = x.nlargest(NTOP)
                        best[mechanism] = [_[1] for _ in x.index.tolist()]
                    best = list(zip(best.keys(), best.values()))
                    # keep only data from top NTOP agents for AOP while keeping TAU
                    cond = (df[test_col] == best[0][0]) & (df.Strategy.isin(best[0][1]))
                    for k, v in best[1:]:
                        cond |= (df[test_col] == k) & (df.Strategy.isin(v))
                    x = df.copy()
                    others = (~cond) & (x[test_col] != "TAU")
                    for col in ("Strategy", "Condition", test_col):
                        x[col] = x[col].astype(str)
                    x.loc[others, "Strategy"] = others_label
                    x.loc[others, "Condition"] = (
                        x.loc[others, test_col] + "+" + x.loc[others, "Strategy"]
                    )
                    for col in ("Strategy", "Condition", test_col):
                        x[col] = x[col].astype("category")
                    if use_average:
                        y = x.loc[
                            (~x[test_col].str.startswith("TAU"))
                            & (x.Strategy != others_label),
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
                                # data=x, x=xaxis, y=metric, hue="Condition", **plot_params  # type: ignore
                                data=x,
                                x=xaxis,
                                y=metric,
                                hue=test_col,
                                **plot_params,  # type: ignore
                            )
                            # if bar_labels and len(x[xaxis].unique()) < MAX_BAR_LABELS:
                            #     ax.bar_label(ax.containers[0], fmt="%3.2f")  # type: ignore
                            if metric in LOG_METRICS:
                                plt.yscale("log")
                            if len(x[xaxis].unique()) > MAX_X_AXIS:
                                ax.set_xticks([], color="white")  # type: ignore
                            elif (
                                metric in ROTATION_METRICS
                                or len(x[metric].unique()) > MAX_X_NO_ROTATION
                            ):
                                plt.xticks(rotation=90)
                            try:
                                sns.move_legend(ax, **legend_params)  # type: ignore
                            except Exception:
                                pass
                            # plt.setp(ax.get_legend().get_texts(), fontsize='10')
                            savefig(
                                f"{dfname}_{xaxis}_{metric}",
                            )
                        except Exception as e:
                            print(f"[red]ERROR[/red]{xaxis} ({metric}) failed with {e}")
                        finally:
                            plt.close()
                except Exception as e:
                    print(f"[red]Failed for {dataset}[/red]{dfname} ({metric}): {e}")
                    continue
                finally:
                    plt.close()
                try:
                    plt.figure(figsize=figsize)
                    ax = plotfn(data=x, y=metric, x=test_col, **plot_params)  # type: ignore
                    if bar_labels and len(x[test_col].unique()) < MAX_BAR_LABELS:
                        ax.bar_label(ax.containers[0], fmt="%3.2f")  # type: ignore
                    if metric in LOG_METRICS:
                        plt.yscale("log")
                    plt.xticks(rotation=90)
                    savefig(f"{dfname}_all_{metric}")
                    plt.close()
                except Exception as e:
                    print(f"[red]ERROR[/red]all ({metric}) failed with {e}")
                    plt.close()

    if score_bars:
        figsize = (8, 3)
        for dfname, metrics in bar_metrics.items():
            df = DF[dfname]
            for metric in metrics:
                fig = plt.figure(figsize=(6, 3))
                ax = fig.subplots()
                if metric in AGREEMENT_ONLY_METRICS:
                    df = df.loc[df.succeeded, :].copy()
                x = df.groupby([test_col])[[metric]].mean().reset_index()
                ax = sns.barplot(
                    ax=ax,  # type: ignore
                    data=df,
                    x=test_col,
                    y=metric,
                    order=mechanism_order,
                    errorbar="se",
                )
                if bar_labels and len(df[test_col].unique()) < MAX_BAR_LABELS:
                    ax.bar_label(ax.containers[0], fmt="%3.2f")  # type: ignore
                if len(df[test_col].unique()) > MAX_X_AXIS:
                    ax.set_xticks([], color="white")  # type: ignore
                else:
                    plt.xticks(rotation=90)
                savefig(f"{dfname}_{metric}_bar")
                plt.close()
                fig = plt.figure(figsize=(15, 5))
                ax = fig.subplots()
                strategy_order = sorted(
                    [_ for _ in df.Strategy.unique()],
                    key=lambda x: x if x not in ("CAB", "SCS", "WAR") else f"zzzz{x}",
                )
                x = df.groupby(["Strategy", test_col])[[metric]].mean().reset_index()
                ax = sns.barplot(
                    ax=ax,  # type: ignore
                    data=df,
                    x="Strategy",
                    y=metric,
                    hue=test_col,
                    order=strategy_order,
                    hue_order=mechanism_order,
                    errorbar=None,
                )
                try:
                    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.05, 1.02))
                except:
                    pass
                if len(x.Strategy.unique()) > MAX_X_AXIS:
                    ax.set_xticks([], color="white")  # type: ignore
                else:
                    plt.xticks(rotation=90)
                savefig(f"{dfname}_{metric}_bar_details")

                plt.close()
    extra = "agent_" if set(AGENT_STATS).intersection(set(designer_stats)) else ""
    sns.set_context("poster")
    if hists:
        figsize = (6, 3)
        for bases_agent in BASES_AGENT:
            x_axis_name = "_".join(bases_agent)
            for metric in AGENT_STATS:
                plot_hists(
                    bases=bases_agent,
                    metric=metric,
                    mech="Mechanism",
                    splitter=None,
                    df=agents,
                    relative=relative,
                )
                savefig(f"adapter_{extra}{metric}_{x_axis_name}")
                plt.close()
        for bases_designer in BASES_DESIGNER:
            x_axis_name = "_".join(bases_designer)
            for metric in designer_stats:
                plot_hists(
                    bases=bases_designer,
                    metric=metric,
                    mech="Mechanism",
                    splitter=None,
                    df=data,
                    relative=relative,
                )
                savefig(f"adapter_{metric}_{x_axis_name}")
                plt.close()
    sns.set_context("poster")
    if hist_details:
        figsize = (6, 3)
        detailed_stats = [_ for _ in DETAILED_STATS if _ in data.columns]
        for splitter in detailed_stats:
            for bases_agent in BASES_AGENT:
                x_axis_name = "_".join(bases_agent)
                for metric in AGENT_STATS:
                    plot_hists(
                        bases=bases_agent,
                        metric=metric,
                        mech="Mechanism",
                        splitter=splitter,
                        df=agents,
                        relative=relative,
                    )
                    savefig(f"adapter_details_{extra}{metric}_{splitter}_{x_axis_name}")
                    plt.close()
            for bases_designer in BASES_DESIGNER:
                x_axis_name = "_".join(bases_designer)
                for metric in designer_stats:
                    plot_hists(
                        bases=bases_designer,
                        metric=metric,
                        mech="Mechanism",
                        splitter=splitter,
                        df=data,
                        relative=relative,
                    )
                    savefig(f"adapter_details_{metric}_{splitter}_{x_axis_name}")
                    plt.close()


def main(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "final",
    name_stem: str = "",
    year: list[int] = [],
    adapters: bool = True,
    remove_mixed: bool = False,
    rename_cab: bool = False,
    separate_pure: bool = True,
    separate_cabwar: bool = True,
    tau_pure: bool = False,
    remove_war: bool = False,
    remove_cab_vs_war: bool = False,
    with_agreements_only: bool = False,
    remove_negative_advantage: bool = False,
    cabwar: bool = True,
    remove_aop: bool = False,
    use_average: bool = True,
    remove_incomplete_domains: bool = True,
    implicit_failures: bool = False,
    failures: bool = False,
    tests: bool = False,
    scatter: bool = False,
    bars: bool = False,
    multilateral: bool = True,
    bilateral: bool = True,
    add_opposition: bool = True,
    hists: bool = False,
    hist_details: bool = False,
    score_bars: bool = True,
    weight_effect: bool = True,
    experiment_results_path: Path = DATA_LOC,
    privacy_factor: float = 0.25,
    scenarios_path=SCENARIOS_LOC,
    test_col: str = "",
    fill_nas: bool = True,
    drop_nas: bool = True,
    save_latex: bool = True,
    alternative: str = "greater",
    relative: bool = False,
    verbose: bool = False,
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = False,
):
    print(f"Working for dataset {dataset}")
    print(f"Will get results from {experiment_results_path}")
    print(f"Will get scenario information from {scenarios_path}")
    if dataset not in VALID_DATASETS and dataset != "all":
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    params = dict(
        no_adapters=not adapters,
        remove_mixed=remove_mixed,
        rename_cab=rename_cab,
        separate_pure=separate_pure,
        separate_cabwar=separate_cabwar,
        tau_pure=tau_pure,
        remove_war=remove_war,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        remove_cab_vs_war=remove_cab_vs_war,
        no_cabwar=not cabwar,
        use_average=use_average,
        remove_incomplete_domains=remove_incomplete_domains,
        tests=tests,
        scatter=scatter,
        bars=bars,
        implicit_failures=implicit_failures,
        failures=failures,
        name_stem=name_stem,
        years=tuple(year),
        multilateral=multilateral,
        bilateral=bilateral,
        privacy_factor=privacy_factor,
        add_opposition=add_opposition,
        hists=hists,
        hist_details=hist_details,
        score_bars=score_bars,
        weight_effect=weight_effect,
        experiment_results_path=experiment_results_path,
        scenarios_path=scenarios_path,
        test_col=test_col,
        fill_nas=fill_nas,
        drop_nas=drop_nas,
        save_latex=save_latex,
        alternative=alternative,
        relative=relative,
        verbose=verbose,
        remove_aop=remove_aop,
        correct_reading_errors=correct_reading_errors,
        overwrite_data_files=overwrite_data_files,
    )
    if dataset == "all":
        for d in VALID_DATASETS_ORDERED:
            print(f"[blue]Working for dataset[/blue] {d}")
            process(dataset=d, **params)  # type: ignore
        return
    process(dataset=dataset, **params)  # type: ignore


if __name__ == "__main__":
    typer.run(main)
