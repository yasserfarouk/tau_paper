from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain, product
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from attr import asdict
from negmas.helpers.inout import add_records
from negmas.preferences.ops import (
    calc_outcome_optimality,
    estimate_max_dist,
    make_rank_ufun,
)
from pandas.errors import EmptyDataError
from rich import print
from rich.progress import track
from utils import (
    DTYPES,
    F0,
    F1,
    MAX_TRIALS,
    MIN_TRIALS,
    RESULTS_FOLDER,
    SCENARIOS_FOLDER,
    TAUNATIVE,
    clean_results,
    find_domains,
    get_rank,
    get_ufuns_for,
    key_,
    load_scenario,
    read_all_stats,
    remove_failed_records,
    set_reservation,
)

from negmas import Scenario, ScenarioStats, calc_outcome_distances

BASEGROUPS = [
    "domain_name",
    # "mechanism_name",
    "strategy_name",
    "f0cat",
    "f1cat",
    "perm_indx",
    # "n_outcomes",
]
GROUPFIELDS = [
    "year",
    "domain_name",
    "f0",
    "f1",
    "perm_indx",
    "time_limit",
    "n_rounds",
    "mechanism_name",
    "strategy_name",
]
FIELDS = [
    "year",
    "domain_name",
    "n_outcomes",
    "f0",
    "f1",
    "perm_indx",
    "time_limit",
    "n_rounds",
    "mechanism_name",
    "strategy_name",
]

# VARS = ["mechanism_name", "strategy_name"]
VARS = ["strategy_name"]
GROUPS = [_ for _ in GROUPFIELDS if _ not in VARS]


def read_data(
    files: list[Path],
    year: int = 0,
    outcomelimit: int = 0,
    remove_implicit_failures=False,
    remove_explicit_failures=True,
    remove_duplicates=True,
    verbose=True,
) -> pd.DataFrame:
    x, added = [], []
    for file_path in files:
        if file_path.is_dir():
            added += [_ for _ in file_path.glob("*.csv")]
    files = files + added
    for file_path in files:
        if file_path.is_dir():
            continue
        clean_results(file_path)
        try:
            x.append(pd.read_csv(file_path))  # , dtype=DTYPES))  # type: ignore
        except EmptyDataError:
            continue
    data = pd.concat(x, ignore_index=True)
    if outcomelimit:
        data = data.loc[data["n_outcomes"] <= outcomelimit, :]
    if remove_explicit_failures or remove_implicit_failures:
        data = remove_failed_records(
            data, explicit=remove_explicit_failures, implicit=remove_implicit_failures
        )
    data = data.loc[data.mechanism_name != "AU0", :]
    data["year"] = data["year"].astype(int)
    if year:
        data = data.loc[data["year"] == year, :]
    pd.options.mode.use_inf_as_na = True
    # print(data.n_outcomes.unique())
    # print(data.groupby('n_outcomes')["strategy_name"].count())
    cond = (
        (data.n_rounds.isna())
        & (data.time_limit.isna())
        & (data.mechanism_name == "TAU0")
    )
    data.loc[cond, "n_rounds"] = data.loc[cond, "n_outcomes"] + 1

    for c in ("time_limit", "n_rounds", "n_outcomes", "perm_indx"):
        data.loc[data[c].isnull(), c] = -2000
        data[c] = data[c].fillna(-1000)
        data[c] = data[c].astype(int)

    # data.loc[data["year"] == 2023, "year"] = 2022
    data.loc[(data.time_limit > 0) & (data.n_rounds > 0), "n_rounds"] = -1000
    data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
    for k, v in DTYPES.items():
        if k not in data.columns or v != "category":
            continue
        data[k] = data[k].cat.remove_unused_categories()
        if verbose:
            print(f"{k}: {data[k].cat.categories}")

    if not remove_duplicates:
        return data
    n = len(data)
    data = data.drop_duplicates()
    if len(data) < n:
        print(f"Dropped {n - len(data)} completey duplicated records")
    return data


def expected_to_run(d: tuple, edict: dict):
    time_limit, n_rounds = d[FIELDS.index("time_limit")], d[FIELDS.index("n_rounds")]
    strategy = d[FIELDS.index("strategy_name")]
    mechanism = d[FIELDS.index("mechanism_name")]
    if mechanism.startswith("TAU") and not (time_limit < 0 and n_rounds < 0):
        return False
    if mechanism.startswith("AO") and (time_limit < 0 and n_rounds < 0):
        return False
    if not any(_ in strategy for _ in ("WAR", "CAN", "WAN", "CAR", "WAB", "CAB")) and (
        time_limit < 0 and n_rounds < 0
    ):
        return False
    # if mechanism.startswith("TAU") and any(_ in strategy for _ in ("CAB", "WAR")):
    #     return False
    if mechanism.startswith("AO") and any(_ in strategy for _ in ("CAB", "WAR")):
        return False
    # exclusion
    if mechanism not in edict.keys():
        return True
    if strategy.split("-")[0] in edict[mechanism]:
        return False
    if strategy.split("-")[-1] in edict[mechanism]:
        return False
    return True


def get_agreement_info(
    year: int,
    n_outcomes: int,
    domain_name: str,
    perm_index: int,
    f0: float,
    f1: float,
    verbose: bool = True,
    fast: bool = False,
) -> dict[str, Any]:
    spath = SCENARIOS_FOLDER / f"y{year}" / f"{n_outcomes:07d}{domain_name}"
    if verbose:
        print(f"Getting agreement info for {spath}")
    assert spath.is_dir(), f"{spath=}"
    domain = Scenario.from_genius_folder(
        spath, ignore_discount=True, ignore_reserved=False
    )
    assert domain is not None
    assert (
        domain.agenda.cardinality == n_outcomes
    ), f"{domain.agenda.cardinality=}, {n_outcomes=}"
    ordinal_path = Path(str(spath).replace("scenarios", "rank_scenarios"))
    try:
        ordinal_scenario = load_scenario(ordinal_path)
        ufuns_ordinal_ = ordinal_scenario.ufuns
    except:
        print(
            f"[yellow][red]Cannot load ordinal scenarios[/red] from {ordinal_path} [/yellow]"
        )
        if fast:
            print(f"[blue]Ignoring {ordinal_path}[/blue]")
            return dict()
        ufuns_ordinal_ = [make_rank_ufun(_) for _ in domain.ufuns]
    ufuns_ = get_ufuns_for(domain.ufuns, perm_index)
    ufuns_ = set_reservation(ufuns_, f0, f1)
    domain.ufuns = ufuns_
    ufuns_ordinal_ = get_ufuns_for(ufuns_ordinal_, perm_index)
    ufuns_ordinal_ = set_reservation(ufuns_ordinal_, f0, f1)
    allstats = read_all_stats(Path(spath))
    if not allstats:
        return dict()
    domain_info = allstats[key_(perm_index, f0, f1)]
    res = domain_info["ordinal_reserved_values"]
    resrank0, resrank1 = res[0], res[1]

    stats = ScenarioStats(**domain_info["cardinal"])
    stats_ordinal = ScenarioStats(**domain_info["ordinal"])

    agreement = None
    agreement_utils = tuple(ufun(agreement) for ufun in ufuns_)
    agreement_ranks = tuple(ufun(agreement) for ufun in ufuns_ordinal_)
    dists = calc_outcome_distances(agreement_utils, stats)
    optim = calc_outcome_optimality(dists, stats, estimate_max_dist(ufuns_))
    dists_ordinal = calc_outcome_distances(agreement_ranks, stats_ordinal)
    optim_ordinal = calc_outcome_optimality(
        dists_ordinal, stats_ordinal, estimate_max_dist(ufuns_ordinal_)
    )
    welfare = sum(agreement_utils)
    agreement_ranks = tuple(get_rank(ufun, agreement) for ufun in ufuns_)
    welfare_rank = sum(agreement_ranks)
    return dict(
        resrank0=resrank0,
        resrank1=resrank1,
        optim=optim,
        dists=dists,
        optim_ordinal=optim_ordinal,
        dists_ordinal=dists_ordinal,
        welfare_rank=welfare_rank,
        opposition=stats.opposition,
        reserved0=ufuns_[0].reserved_value,
        reserved1=ufuns_[1].reserved_value,
        reserved=tuple(u.reserved_value for u in ufuns_),
        max_pareto_welfare=max(sum(_) for _ in stats.pareto_utils),
        agreement_utils=agreement_utils,
        agreement_ranks=agreement_ranks,
        welfare=welfare,
        welfare_ran=welfare_rank,
        utility_ranges=stats.utility_ranges,
        rank_ranges=stats_ordinal.utility_ranges,
        ordinal_opposition=stats_ordinal.opposition,
    )


def complete_missing(
    path: Path, missing_data: dict, agreement_info: dict, n_records: int, verbose=False
) -> None:
    if n_records < 1:
        print(f"[red]Specified 0 records to add for {tuple(missing_data.values())}")
        return
    if verbose:
        print(f"Will addd {n_records} values of type: {tuple(missing_data.values())}")
    mechanism_name = missing_data["mechanism_name"]
    strategy_name = missing_data["strategy_name"]
    year = missing_data["year"]
    n_outcomes = missing_data["n_outcomes"]
    domain_name = missing_data["domain_name"]
    perm_index = missing_data["perm_indx"]
    f0 = missing_data["f0"]
    f1 = missing_data["f1"]
    time_limit = missing_data["time_limit"]
    if time_limit < 0:
        time_limit = None
    n_rounds = missing_data["n_rounds"]
    if n_rounds < 0:
        n_rounds = None
    result: dict[str, Any] = dict(
        mechanism_name=mechanism_name,
        strategy_name=strategy_name,
        first_strategy_name=strategy_name.split("-")[0],
        second_strategy_name=strategy_name.split("-")[-1],
        domain_name=domain_name,
        year=year,
        n_outcomes=n_outcomes,
        opposition=agreement_info["opposition"],
        time_limit=time_limit,
        n_rounds=n_rounds,
        f0=f0,
        f1=f1,
        reserved0=agreement_info["reserved0"],
        reserved1=agreement_info["reserved1"],
        reserved=agreement_info["reserved"],
        resrank0=agreement_info["resrank0"],
        resrank1=agreement_info["resrank1"],
        strategy_params="",
        second_strategy_params="",
        perm_indx=perm_index,
        max_pareto_welfare=agreement_info["max_pareto_welfare"],
        utility_ranges=agreement_info["utility_ranges"],
        rank_ranges=agreement_info["rank_ranges"],
        ordinal_opposition=agreement_info["ordinal_opposition"],
        time=float("nan"),
        run_time=float("nan"),
        succeeded=False,
        failed=False,
        **asdict(agreement_info["dists"]),
        **{
            f"ordinal_{k}": v
            for k, v in asdict(agreement_info["dists_ordinal"]).items()
        },
        **asdict(agreement_info["optim"]),
        **{
            f"ordinal_{k}": v
            for k, v in asdict(agreement_info["optim_ordinal"]).items()
        },
        full_time=float("nan"),
        agreement_utils=agreement_info["agreement_utils"],
        agreement_ranks=agreement_info["agreement_ranks"],
        welfare=agreement_info["welfare"],
        welfare_rank=agreement_info["welfare_rank"],
        steps=0,
    )
    add_records(path, pd.DataFrame([result] * n_records))


def do_completion(missing, path, title, verbose, fast):
    if missing is None or len(missing) < 1:
        return
    missing_dicts = missing.to_dict(orient="records")
    infos = dict()
    if verbose:
        print(f"Will complete {len(missing_dicts)} cases ({title})")
    for missing_data in track(
        missing_dicts,
        total=len(missing_dicts),
        description=f"Completing {title} ...",
    ):
        year = missing_data["year"]
        n_outcomes = missing_data["n_outcomes"]
        domain_name = missing_data["domain_name"]
        perm_index = missing_data["perm_indx"]
        f0 = missing_data["f0"]
        f1 = missing_data["f1"]
        f0str = repr(f0)
        f1str = repr(f1)
        key = (year, n_outcomes, domain_name, perm_index, f0str, f1str)
        if verbose:
            print(f"Adding: {key}")
        domain_info = infos.get(key, None)
        if domain_info is None:
            domain_info = get_agreement_info(
                year,
                n_outcomes,
                domain_name,
                perm_index,
                f0,
                f1,
                verbose,
                fast=fast,
            )
            infos[key] = domain_info
        n_to_add = missing_data["n_missing"]
        if domain_info and n_to_add:
            complete_missing(
                path,
                missing_data,
                domain_info,
                n_to_add,
                verbose=verbose,
            )


def update_categories(data: pd.DataFrame):
    for k, v in DTYPES.items():
        if v != "category" or k not in data.columns:
            continue
        data[k] = data[k].cat.remove_unused_categories()
    return data


def process(
    data,
    min_n_trials,
    n_trials,
    complete_missing_all,
    complete_missing_some,
    complete_tau,
    edict,
    completion_folder,
    verbose,
    mechanism,
    stats_folder,
    fast: bool,
    complete_proposed: bool,
    save_missing_all: bool,
    save_missing_some: bool,
    save_missing: bool,
    original_only: bool,
):
    stats_folder.mkdir(parents=True, exist_ok=True)
    completion_folder.mkdir(parents=True, exist_ok=True)
    complete_path = completion_folder / f"completed_{mechanism}.csv"
    # # convert condition columns back to float if needed
    # for col in FIELDS:
    #     if f"_orig{col}" in data.columns:
    #         data[col] = data[f"_orig{col}"]
    #         data.drop(f"_orig{col}", axis=1)
    # print(data.year.unique())
    data = data.loc[data.mechanism_name == mechanism, :]
    if original_only:
        data = data.loc[(data.f0 < 0) & (data.f1 < 0), :]
        data["f0cat"] = "-1"
        data["f1cat"] = "-1"
        data["f0cat"] = data["f0cat"].astype("category")
        data["f1cat"] = data["f1cat"].astype("category")
    else:
        data["f0cat"] = pd.cut(
            data.f0,
            bins=[-2.0, 0.0, 0.25, 0.75, 0.95, 2.0],
            labels=["-1", "0.1", "0.5", "0.9", "1.0"],
        )
        data["f1cat"] = pd.cut(
            data.f1,
            bins=[-2.0, 0.0, 0.25, 0.75, 0.95, 2.0],
            labels=["-1", "0.1", "0.5", "0.9", "1.0"],
        )
    data = update_categories(data)
    data = data[BASEGROUPS]
    counts = data.groupby(BASEGROUPS).size()
    # print(counts.head())
    print(f"\t{mechanism}: Found cases with {counts.unique()} runs")
    counts = counts.reset_index()
    if original_only:
        counts = counts.loc[
            ((counts.f0cat == "-1") & (counts.f1cat == "-1")),
            :,
        ]
    else:
        counts = counts.loc[
            (
                (counts.f0cat == "-1") & (counts.f1cat == "-1")
                | (counts.f0cat == "1.0") & (counts.f1cat == "0.1")
                | (counts.f0cat == "1.0") & (counts.f1cat == "0.5")
                | (counts.f0cat == "1.0") & (counts.f1cat == "0.9")
                | (counts.f0cat == "1.0") & (counts.f1cat == "1.0")
            ),
            :,
        ]
    # counts.fillna(0, inplace=True)
    counts.rename(columns={0: "count"}, inplace=True)  # type: ignore
    missingsome = counts.loc[
        (counts["count"] > 0) & (counts["count"] < min_n_trials)
    ].copy()
    missingsome = update_categories(missingsome)
    missingsome["n_missing"] = min_n_trials - missingsome["count"]
    missingsome = missingsome.drop(columns=["count"])
    if save_missing_some and len(missingsome):
        print(
            f"[yellow]Found {len(missingsome)} conditions missing some trials[/yellow]"
        )
        missingsome.to_csv(stats_folder / f"missingsome_{mechanism}.csv", index=False)

    missingall = counts.loc[(counts["count"] == 0)].copy()
    missingall["n_missing"] = min_n_trials
    missingall = missingall.drop(columns=["count"])
    missingall = update_categories(missingall)
    if save_missing_all and len(missingall):
        print(f"[red]Found {len(missingall)} conditions missing ALL trials[/red]")
        missingall.to_csv(stats_folder / f"missingall_{mechanism}.csv", index=False)
    label = ""
    if not complete_missing_some and not complete_missing_all:
        missing = pd.DataFrame([], columns=missingall.columns)
        label = "MISSING NONE"
    elif complete_missing_some and not complete_missing_all:
        missing = missingsome
        label = "MISSING SOME But not all"
    elif not complete_missing_some and complete_missing_all:
        missing = missingall
        label = "MISSING ALL"
    else:
        missing = pd.concat((missingall, missingsome), ignore_index=True)
        label = "MISSING ANY"
    if not complete_tau:
        for s in TAUNATIVE:
            missing = missing.loc[~missing.strategy_name.str.contains(s), :]
    if not complete_proposed:
        TAUALL = [
            f"{_[0]}-{_[1]}" for _ in product(TAUNATIVE, TAUNATIVE) if _[0] != _[1]
        ] + list(TAUNATIVE)
        missing = missing.loc[missing.strategy_name.isin(TAUALL), :]

    if len(missing) < 1:
        return len(missing)
    print(f"\t{mechanism}: [blue]Completing {len(missing)} cases[/blue]")
    if not complete_missing_some:
        missing.n_missing = 1
    do_completion(missing, complete_path, label, verbose=verbose, fast=fast)
    return len(missing)
    # # for col in FIELDS:
    # #     if f"_orig{col}" in counts.columns:
    # #         counts[col] = counts[f"_orig{col}"]
    # #         counts.drop(f"_orig{col}", axis=1)
    # pivot = counts.pivot_table(  # type: ignore
    #     values="count",
    #     columns=VARS,
    #     index=GROUPS,
    # )
    # pivot.transpose().reset_index().to_csv(
    #     stats_folder / f"runs_{mechanism}.csv", index=False
    # )
    # flattened = pivot.fillna(0).values.flatten().astype(int)
    # runs = set(flattened)
    # if n_trials == 0:
    #     n_trials = int(stats.mode(flattened, keepdims=True).mode)
    # print(f"\t{mechanism}: After pivoting, found cases with {runs} runs")
    # if 0 not in runs:
    #     print(
    #         f"\t{mechanism}: [green]Everything OK. No need for adding failures[/green]"
    #     )
    # # print(f"Will ignore {n_trials} to all domains that were never tried")
    # with_nans = pivot[pivot.isna().any(axis=1)]  # type: ignore
    # with_nans: pd.DataFrame
    # rows = with_nans.index.to_list()
    # cols = []
    # for c in with_nans.columns:
    #     if with_nans[c].isna().any():  # type: ignore
    #         cols.append(c)
    # noruns = list(_ for _ in product(rows, cols))
    # noruns = [
    #     tuple(list(a) + (list(b) if not isinstance(b, str) else [b])) for a, b in noruns
    # ]
    # noruns = [_ for _ in noruns if expected_to_run(_, edict)]
    # noruns = list(set(noruns))
    # ind = FIELDS.index("n_outcomes")
    # noruns = sorted(noruns, key=lambda x: x[ind])
    # if not complete_tau:
    #     ind = FIELDS.index("mechanism_name")
    #     noruns = [_ for _ in noruns if "TAU" not in _[ind]]
    # if not complete_proposed:
    #     ind = FIELDS.index("strategy_name")
    #     noruns = [_ for _ in noruns if "CAB" not in _[ind] and "WAR" not in _[ind]]
    # # titles = ("year", "domain_name", "n_outcomes", "f0", "f1",
    # #           "perm_index", "time_limit", "n_rounds", "mechanism_name", "strategy_name", "count")
    # # missingsome = pd.DataFrame([dict(zip(titles, list(_) + [0])) for _ in noruns])
    # # missingsome.to_csv(output_folder / "noruns.csv", index=False)
    #
    # if len(noruns):
    #     print(f"\t{mechanism}: Completing {len(noruns)} cases with no runs", flush=True)
    # if min_n_trials and save_missing_all:
    #     norunsdict = [dict(zip(FIELDS, _)) for _ in noruns]
    #     missingall = pd.DataFrame.from_records(norunsdict)
    #     missingall["n_missing"] = min_n_trials
    #     missingall.to_csv(stats_folder / f"missingall_{mechanism}.csv", index=False)
    #     if missingsome is not None:
    #         missing_final = pd.concat((missingall, missingsome), ignore_index=True)
    #     else:
    #         missing_final = missingall
    #
    #     def do_completion(missing, path, title):
    #         if missing is None or len(missing) < 1:
    #             return
    #         missing_dicts = missing.to_dict(orient="records")
    #         infos = dict()
    #         if verbose:
    #             print(f"Will complete {len(missing_dicts)} cases ({title})")
    #         for missing_data in track(
    #             missing_dicts,
    #             total=len(missing_dicts),
    #             description=f"Completing {title} ...",
    #         ):
    #             year = missing_data["year"]
    #             n_outcomes = missing_data["n_outcomes"]
    #             domain_name = missing_data["domain_name"]
    #             perm_index = missing_data["perm_indx"]
    #             f0 = missing_data["f0"]
    #             f1 = missing_data["f1"]
    #             f0str = repr(f0)
    #             f1str = repr(f1)
    #             key = (year, n_outcomes, domain_name, perm_index, f0str, f1str)
    #             if verbose:
    #                 print(f"Adding: {key}")
    #             domain_info = infos.get(key, None)
    #             if domain_info is None:
    #                 domain_info = get_agreement_info(
    #                     year,
    #                     n_outcomes,
    #                     domain_name,
    #                     perm_index,
    #                     f0,
    #                     f1,
    #                     verbose,
    #                     fast=fast,
    #                 )
    #                 infos[key] = domain_info
    #             n_to_add = missing_data["n_missing"]
    #             if domain_info and n_to_add:
    #                 complete_missing(
    #                     path,
    #                     missing_data,
    #                     domain_info,
    #                     n_to_add,
    #                     verbose=verbose,
    #                 )
    #
    #     if complete_missing_all and missingall is not None and len(missingall) > 0:
    #         print(
    #             f"\t{mechanism}: [blue]Completing {len(missingall)} cases missing all results[/blue]"
    #         )
    #         if not complete_missing_some:
    #             missingall.n_missing = 1
    #         do_completion(missingall, complete_all_path, "Missing ALL")
    #     if complete_missing_some and missingsome is not None and len(missingsome) > 0:
    #         print(
    #             f"\t{mechanism}: [/yellow]Completing {len(missingsome)} cases missing some results[yellow]"
    #         )
    #         do_completion(missingsome, complete_some_path, "missing some")
    # else:
    #     missing_final = missingsome
    # if missing_final is not None and save_missing:
    #     missing_final = missing_final.copy()
    #     missing_final.loc[missing_final.time_limit < 0, "time_limit"] = None
    #     missing_final.loc[missing_final.n_rounds < 0, "n_rounds"] = None
    #     missing_final["year"] = missing_final.year.astype(int)
    #     missing_final.to_csv(stats_folder / f"missing_{mechanism}.csv", index=False)
    # return len(noruns)


def save_aggregates(data: pd.DataFrame, stats_folder: Path, suffix=""):
    data.groupby(["mechanism_name"]).size().reset_index().rename(
        columns={0: "count"}
    ).to_csv(stats_folder / f"runs_per_mechanism_{suffix}.csv")
    for k in (
        "strategy_name",
        "first_strategy_name",
        "second_strategy_name",
        "domain_name",
    ):
        data.groupby(["mechanism_name", k]).size().reset_index().rename(
            columns={0: "count"}
        ).pivot_table(values="count", columns="mechanism_name", index=k).to_csv(
            stats_folder / f"runs_per_{k.replace('_name', '')}_{suffix}.csv"
        )
    data.groupby(
        ["mechanism_name", "strategy_name", "domain_name"]
    ).size().reset_index().rename(columns={0: "count"}).pivot_table(
        values="count", columns=["mechanism_name", "domain_name"], index="strategy_name"
    ).to_csv(
        stats_folder / f"runs_per_strategy_and_domain_{suffix}.csv"
    )


def main(
    file: list[Path],
    n_trials: int = MAX_TRIALS,
    min_n_trials: int = MIN_TRIALS,
    complete_missing_all: bool = False,
    complete_missing_some: bool = False,
    complete_tau: bool = True,
    complete_proposed: bool = True,
    stats_folder: Path = RESULTS_FOLDER,
    completion_folder: Path = RESULTS_FOLDER,
    year: int = 0,
    excluded: list[str] = [],
    # [
    #     f"TAU:{_}"
    #     for _ in [  # "AspirationNegotiator",
    #         # "MiCRONegotiator",
    #         # "Atlas3",
    #         "AgentK",
    #         "Caduceus",
    #         "HardHeaded",
    #         "CUHKAgent",
    #         "NiceTitForTat",
    #         "ConcederTBNegotiator",
    #         "LinearTBNegotiator",
    #         "Conceder",
    #         "Linear",
    #     ]
    #     + TAUVARIANTS
    # ],
    outcomelimit: int = 0,
    remove_implicit_failures: bool = True,
    remove_explicit_failures: bool = True,
    remove_duplicates: bool = True,
    verbose: bool = False,
    fast: bool = True,
    serial: bool = None,  # type: ignore
    save_missing_all: bool = True,
    save_missing_some: bool = True,
    save_missing: bool = False,
    original_runs_only: bool = False,
):
    edict = defaultdict(list)
    allexcluded = [_ for _ in chain(*tuple(_.split(";") for _ in excluded))]
    for k, _, v in [_.partition(":") for _ in allexcluded]:
        edict[k].append(v)
    if edict:
        print(f"Will exclude {edict} from checking failures")
    data = read_data(
        file,
        year=year,
        outcomelimit=outcomelimit,
        remove_implicit_failures=remove_implicit_failures,
        remove_explicit_failures=remove_explicit_failures,
        remove_duplicates=remove_duplicates,
        verbose=verbose,
    )
    if remove_duplicates and n_trials < 2:
        n = len(data)
        data.drop_duplicates(FIELDS, inplace=True)
        if len(data) < n:
            print(
                f"Removed {n - len(data)} records with duplications in all fields that matter (keeping {len(data)} records)"
            )
    if original_runs_only:
        print(f"Will use original runs only")
        data = data.loc[(data.f0 < 0) & (data.f1 < 0), :]
        f0, f1 = [], []
    else:
        f0, f1 = F0, F1
    alldomains = set(_[7:] for _ in find_domains(year, outcomelimit=outcomelimit))
    n_domains = len(alldomains)
    founddomains = set(data["domain_name"].unique())
    n_domains_found = len(founddomains)
    n_aom = len(
        data.loc[data.mechanism_name.str.startswith("AO"), "mechanism_name"].unique()
    )
    n_aop = len(
        data.loc[data.mechanism_name.str.startswith("AO"), "strategy_name"].unique()
    )
    tau_strategies = data.loc[
        data.mechanism_name.str.startswith("TAU"), "strategy_name"
    ].unique()
    if "TAU" in edict.keys():
        tau_strategies = sorted(
            [_ for _ in tau_strategies if not any(x in _ for x in edict["TAU"])]
        )
    n_tau = len(tau_strategies)
    n_conditions = n_domains_found * ((n_aop * n_aom) + n_tau) * (len(f0) * len(f1) + 1)
    n_runs = n_conditions * min_n_trials
    n_records = len(data)
    if n_domains_found > n_domains:
        print(f"[red]Found {n_domains_found} domains (Expected {n_domains})[/red]")
        print(
            f"Domains not expected in data: {list(alldomains.difference(founddomains))}"
        )
    elif n_domains > n_domains_found:
        print(f"[green]Found {n_domains_found} domains (Expected {n_domains})[/green]")
        print(f"Domains not found in data: {list(founddomains.difference(alldomains))}")
    print(f"Found {n_records} records")
    print(
        f"Expected {n_conditions} conditions ({n_runs} runs): {n_aom} AOP variants, {n_aop} AOP strategies, {n_tau} TAU strategies"
    )
    mechanisms = data.mechanism_name.unique()
    print(f"Found {len(mechanisms)} mechanisms: {data.mechanism_name.unique()}")
    print(f"Found {len(data.strategy_name.unique())} strategies")
    n_no_runs = 0
    save_aggregates(data, stats_folder, "before")
    if serial is None and len(data) > 50_000:
        serial = True

    if serial:
        for mechanism in mechanisms:
            print(f"Processing {mechanism}")
            n_no_runs += process(
                data.loc[data.mechanism_name == mechanism, :],
                min_n_trials,
                n_trials,
                complete_missing_all,
                complete_missing_some,
                complete_tau,
                edict,
                completion_folder,
                verbose,
                mechanism,
                stats_folder,
                fast=fast,
                complete_proposed=complete_proposed,
                save_missing_all=save_missing_all,
                save_missing_some=save_missing_some,
                save_missing=save_missing,
                original_only=original_runs_only,
            )
    else:
        futures = []
        cpus = cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for mechanism in mechanisms:
                futures.append(
                    pool.submit(
                        process,
                        data.loc[data.mechanism_name == mechanism, :],
                        min_n_trials,
                        n_trials,
                        complete_missing_all,
                        complete_missing_some,
                        complete_tau,
                        edict,
                        completion_folder,
                        verbose,
                        mechanism,
                        stats_folder,
                        original_only=original_runs_only,
                        fast=fast,
                        complete_proposed=complete_proposed,
                        save_missing_all=save_missing_all,
                        save_missing_some=save_missing_some,
                        save_missing=save_missing,
                    )
                )
            for f in track(
                as_completed(futures), total=len(futures), description="Processing ..."
            ):
                n_no_runs += f.result()

    save_aggregates(data, stats_folder, "after")
    diff = n_runs - n_records
    if 0 < diff < n_no_runs:
        print(f"[red]Expected at most {diff} cases with no runs[/red]", flush=True)
    else:
        print(f"[green]Expected at most {diff} cases with no runs[/green]", flush=True)


if __name__ == "__main__":
    typer.run(main)
