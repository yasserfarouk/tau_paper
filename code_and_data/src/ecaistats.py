import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from figutils import (BASE, DATA_LOC, MECHANISM_TYPE, SCENARIOS_LOC,
                      TAU_NATIVE_LABEL, VALID_DATASETS, VALID_DATASETS_ORDERED,
                      read_and_adjust)
from negmas.genius.ginfo import itertools
from pandas.errors import DtypeWarning, SettingWithCopyWarning
from rich import print

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 4)

warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.options.display.float_format = "${:,.4f}".format

PERFORMANCE_MEASURE = "Measure"
COLMAP = dict(level_0=PERFORMANCE_MEASURE, level_1="Statistic")


def update_categories(data):
    for k in data.columns:
        if data[k].dtype != "category":
            continue
        data[k] = data[k].cat.remove_unused_categories()
    return data


X = (0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0)
xs = {
    "2010": (0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0),
}


def rational(os, ufuns):
    n = 0
    for outcome in os.enumerate():
        if all(not ufun.is_worse(outcome, None) for ufun in ufuns):
            n += 1
    return round(n / os.cardinality, 1)


def save_results(
    df: pd.DataFrame,
    path: Path,
    dataset: str,
    save_latex: bool = True,
    precision=2,
    remove_cab_vs_war: bool = False,
    summary_stats=[
        "DesignerScore",
        "Advantage",
        "Speed",
    ],
    summary_stats_final=[
        "DesignerScore",
        "Advantage",
        "Speed",
        "Completeness",
        "Optimality",
        "Fairness",
        "Welfare",
        "Privacy",
    ],
    relative: bool = False,
    relative_base: str = "AOP(3min)",
):
    if dataset.startswith("final"):
        summary_stats = summary_stats_final
    dst = path / "csv" / f"{dataset}.csv"
    dst.parent.mkdir(exist_ok=True, parents=True)
    df.rename(columns=COLMAP, inplace=True)
    df.to_csv(dst, index=False)
    cond = df[PERFORMANCE_MEASURE].isin(summary_stats)
    tbl1, tbl2 = df.loc[cond, :], df.loc[~cond, :]
    for tbl, typ in zip([tbl1, tbl2], ["summary", "details"]):
        indx = [
            _
            for _ in tbl.columns
            if not _.startswith("AO")
            and not _.startswith("TAU")
            and not _ in COLMAP.values()
        ] + [_ for _ in COLMAP.values() if _ != "Statistic"]
        tbl = tbl.pivot(index=indx, columns=["Statistic"])
        if remove_cab_vs_war:
            tbl = tbl[
                [
                    _
                    for _ in tbl.columns
                    if not _[0]
                    in (TAU_NATIVE_LABEL, TAU_NATIVE_LABEL.replace("CAB", "SCS"))
                ]
            ]
        if relative:
            for col in tbl.columns:
                if col[0].startswith("AO") or col[0].startswith("TAU"):
                    if col[0] == relative_base:
                        continue
                    tbl[col] /= tbl[(relative_base, col[-1])]
            for col in tbl.columns:
                if col[0].startswith("AO") or col[0].startswith("TAU"):
                    if col[0] == relative_base:
                        tbl[col] /= tbl[(relative_base, col[-1])]

        if not save_latex:
            continue

        dst = path / "latex" / f"{dataset}_{typ}.tex"
        dst.parent.mkdir(exist_ok=True, parents=True)
        styler = tbl.style.highlight_max(
            axis=1,
            props="bfseries: ;",
        )
        # styler = tbl.style.highlight_min(
        #     axis=1,
        #     props="underline: ;",
        # )
        formatted = [_ for _ in tbl.columns if _ not in COLMAP.values()]
        if precision > 0:
            styler.format(
                dict(zip(formatted, itertools.repeat(f"{{:0.{precision}f}}")))
            )

        styler.to_latex(dst, hrules=True)


def process(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "final",
    years: tuple[int] = tuple(),  # type: ignore
    no_adapters: bool = True,
    rename_cab: bool = False,
    tau_pure: bool = True,
    remove_war: bool = True,
    with_agreements_only: bool = False,
    remove_negative_advantage: bool = False,
    no_cabwar: bool = False,
    remove_cab_vs_war: bool = False,
    remove_mixed: bool = True,
    separate_cabwar: bool = True,
    separate_pure: bool = False,
    implicit_failures: bool = False,
    failures: bool = False,
    verbose: bool = False,
    multilateral: bool = True,
    bilateral: bool = True,
    privacy_factor: float = 0,
    add_opposition: bool = True,
    experiment_results_path: Path = DATA_LOC,
    scenarios_path: Path = SCENARIOS_LOC,
    remove_incomplete_domains: bool = True,
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = True,
    test_col: str = "",
    statistics: list[str] = ("mean",),
    reltaive: bool = False,
):
    if not test_col:
        test_col = "Mechanism" if no_cabwar else MECHANISM_TYPE
    experiment_results_path = Path(experiment_results_path)
    scenarios_path = Path(scenarios_path)
    if not experiment_results_path.exists():
        print(f"Experiment results path given {experiment_results_path} does not exist")
        return
    if not scenarios_path.exists() and add_opposition:
        print(f"Scenarios path given {scenarios_path} does not exist")
        return
    if dataset not in VALID_DATASETS:
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    agents, data, base_name, filename = read_and_adjust(
        dataset=dataset,
        years=years,
        no_adapters=no_adapters,
        remove_mixed=remove_mixed,
        rename_cab=rename_cab,
        remove_cab_vs_war=remove_cab_vs_war,
        tau_pure=tau_pure,
        remove_war=remove_war,
        separate_pure=separate_pure,
        separate_cabwar=separate_cabwar,
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
    )
    if agents is None:
        return
    assert (
        agents is not None
        and data is not None
        and base_name is not None
        and filename is not None
    )
    METRICS = dict(
        designer=[
            # "Quality",
            # "Overall Score",
            "DesignerScore",
            # "AgentScore",
            # "Score",
            # "IRUB",
            # "Privacy",
            "Optimality",
            "Completeness",
            "Fairness",
            "N_Fairness",
            "O_N_Fairness",
            "K_Fairness",
            "O_K_Fairness",
            "RK_Fairness",
            "O_RK_Fairness",
            "Welfare",
            # "Advantage",
            "Time",
            "Rounds",
            "Speed",
            "Uniqueness",
        ],
        agent=["AgentScore", "Privacy", "Advantage", "AdvantagePartner"],
    )
    # metrics = ["DesignerScore"]
    # TWO_SIDED = []
    # LESS = ["Privacy"]
    # MIN_METRICS = ["Time", "Rounds", "IRUB"]
    # AGREEMENT_ONLY_METRICS = [_ for _ in data.columns if "Fairness" in _]
    # CAT = [
    #     "Strategy",
    #     "Strategy1",
    #     "Strategy2",
    #     "StrategyPartner",
    #     "Condition",
    # ]  # , "Type"
    DF = dict(designer=data, agent=agents)
    dfs = []
    for dfname, metrics in METRICS.items():
        df = DF[dfname]
        dfs.append(
            df[metrics + [test_col]]  # type: ignore
            .groupby([test_col])
            # .describe()
            .agg(statistics)  # type: ignore
            .transpose()
            .reset_index()
        )
        print(dfs[-1])
    return pd.concat(dfs, axis=0, ignore_index=True)


def main(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "final",
    year: list[int] = [],
    adapters: bool = True,
    rename_cab: bool = False,
    tau_pure: bool = False,
    remove_war: bool = False,
    remove_cab_vs_war: bool = False,
    cabwar: bool = True,
    remove_mixed: bool = False,
    with_agreements_only: bool = False,
    separate_cabwar: bool = True,
    remove_negative_advantage: bool = False,
    implicit_failures: bool = False,
    failures: bool = False,
    multilateral: bool = True,
    bilateral: bool = True,
    add_opposition: bool = True,
    experiment_results_path: Path = DATA_LOC,
    privacy_factor: float = 0,
    scenarios_path=SCENARIOS_LOC,
    test_col: str = "",
    save_latex: bool = True,
    verbose: bool = False,
    remove_incomplete_domains: bool = True,
    name_stem: str = "",
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = False,
    statistics: list[str] = ("mean",),
    precision: int = 2,
    relative: bool = False,
):
    print(f"Working for dataset {dataset}")
    print(f"Will get results from {experiment_results_path}")
    print(f"Will get scenario information from {scenarios_path}")
    if dataset not in VALID_DATASETS and dataset != "all":
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    params: dict[str, Any] = dict(
        no_adapters=not adapters,
        rename_cab=rename_cab,
        tau_pure=tau_pure,
        remove_war=remove_war,
        # remove_cab_vs_war=remove_cab_vs_war,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        no_cabwar=not cabwar,
        statistics=statistics,
        implicit_failures=implicit_failures,
        failures=failures,
        separate_cabwar=separate_cabwar,
        years=tuple(year),
        multilateral=multilateral,
        remove_incomplete_domains=remove_incomplete_domains,
        bilateral=bilateral,
        privacy_factor=privacy_factor,
        add_opposition=add_opposition,
        experiment_results_path=experiment_results_path,
        scenarios_path=scenarios_path,
        test_col=test_col,
        remove_mixed=remove_mixed,
        verbose=verbose,
        correct_reading_errors=correct_reading_errors,
        overwrite_data_files=overwrite_data_files,
    )
    base_path = BASE / "summary"
    if name_stem:
        base_path /= name_stem
    saveparams = dict(
        save_latex=save_latex,
        precision=precision,
        relative=relative,
        remove_cab_vs_war=remove_cab_vs_war,
    )
    if dataset == "all":
        dfs = []
        for d in VALID_DATASETS_ORDERED:
            print(f"Working for dataset {d}")
            df = process(dataset=d, **params)
            if df is None or len(df) == 0:
                print(f"[red]Found nothing for {dataset}[/red]")
                continue
            save_results(df, base_path, dataset=d, **saveparams)
            dfs.append(df)
            dfs[-1]["Dataset"] = d
        if not dfs:
            print(f"[red]Found nothing for {dataset}[/red]")
            return

        save_results(
            pd.concat(dfs, ignore_index=True), base_path, dataset=dataset, **saveparams
        )
        print(f"Results saved to {base_path.absolute()}")
        return
    df = process(dataset=dataset, **params)
    if df is None:
        print(f"[red]Found nothing for {dataset}[/red]")
        return

    print(df.head())
    save_results(df, base_path, dataset=dataset, **saveparams)
    print(f"Results saved to {base_path.absolute()}")


if __name__ == "__main__":
    typer.run(main)
