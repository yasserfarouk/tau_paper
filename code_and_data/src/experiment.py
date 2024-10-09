import random
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from random import shuffle
from socket import gethostname
from time import perf_counter, sleep
from typing import Any, Callable, Sequence, Type
from uuid import uuid4

import pandas as pd
import typer
from adapters.tau import TAUNegotiatorAdapter
from attr import asdict, dataclass, field
from helpers.utils import (
    DTYPES,
    F0,
    F1,
    FIGS_FOLDER,
    GENIUSMARKER,
    IRKEY,
    MAX_TRIALS,
    PERMUTAION_METHOD,
    RESULTS_PATH,
    SCENRIOS_PATH,
    TAUNATIVE,
    calc_rational_fraction,
    clean_frame,
    clean_results,
    correct_field_types,
    count_runs,
    get_all_anac_agents,
    get_genius_proper_class_name,
    get_rank,
    get_ufuns_for,
    key_,
    load_scenario,
    read_all_stats,
    read_stats_of,
    remove_invalid_lines,
    remove_repeated_header,
    set_reservation,
)

# from negmas.gb import TAUMechanism
# from negmas.gb.adapters.tau import TAUNegotiatorAdapter
from mechanisms import TAUMechanism
from negmas.helpers import unique_name
from negmas.helpers.inout import add_records
from negmas.helpers.strings import humanize_time, itertools
from negmas.helpers.types import get_class, get_full_type_name, instantiate
from negmas.inout import Scenario, ScenarioStats
from negmas.mechanisms import Mechanism
from negmas.preferences.ops import (
    calc_outcome_distances,
    calc_outcome_optimality,
    estimate_max_dist,
    make_rank_ufun,
)
from negmas.sao import SAOMechanism
from numpy import mean
from pandas.errors import EmptyDataError
from rich import print
from rich.traceback import install

install(show_locals=False)
from rich.progress import track

from negmas import GeniusNegotiator, Negotiator

EPS = 0.002
FINAL_TRIAL_TIME = 10 * 60
N_EXCEPTION_LINES = 7
ALLYEARS = [2010, 2011, 2012, 2013, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
IGNORED_COMBINATIONS = [
    # dict(
    #     first="Caduceus".lower(),
    #     year=2012,
    #     domain="Outfit".lower(),
    #     mechanism="AO".lower(),
    # ),
    # dict(
    #     first="Caduceus".lower(),
    #     year=2012,
    #     domain="Fitness".lower(),
    #     mechanism="AO".lower(),
    # ),
    dict(
        first="Caduceus".lower(),
    ),
    dict(
        second="Caduceus".lower(),
    ),
    dict(
        second="Caduceus".lower(),
        year=2012,
        domain="Outfit".lower(),
        mechanism="AO".lower(),
        f1=0.1,
    ),
    dict(
        second="Caduceus".lower(),
        year=2012,
        domain="RentalHouse".lower(),
        mechanism="AO".lower(),
        f1=0.1,
    ),
    # dict(
    #     second="Caduceus".lower(),
    #     year=2012,
    #     domain="Fitness".lower(),
    #     mechanism="AO".lower(),
    #     f1=0.1,
    # ),
]


def run_with_feedback(m: Mechanism, msg: str):
    for _ in m:
        print(
            f"{msg}: [green]{m.relative_time}[/green] [{m.current_step}/(n:{m.nmi.n_steps}, t:{m.nmi.time_limit})]"
        )


def is_match(
    pattern: dict,
    mechanism_name: str,
    s1: str,
    s2: str,
    domain_name: str,
    domain_path: str,
    f0: float,
    f1: float,
) -> bool:
    domain_path = str(domain_path)
    for k, v in pattern.items():
        if k == "first" and v not in s1.lower():
            return False
        if k == "second" and v not in s2.lower():
            return False
        if k == "year" and f"y{v}/" not in domain_path.lower():
            return False
        if k == "domain" and not domain_name.lower().endswith(v):
            return False
        if k == "mechanism" and v in mechanism_name.lower():
            return False
        if k == "f0" and abs(f0 - v) > 1e-3:
            return False
        if k == "f1" and abs(f1 - v) > 1e-3:
            return False
    return True


def is_ignored(
    mechainsm_name: str,
    s1: str,
    s2: str,
    domain_name: str,
    domain_path: str,
    f0: float,
    f1: float,
    patterns: list[dict[str, Any]] = IGNORED_COMBINATIONS,
) -> bool:
    return any(
        is_match(_, mechainsm_name, s1, s2, domain_name, domain_path, f0, f1)
        for _ in patterns
    )


def isint(x):
    try:
        int(x)
        return True
    except:
        return False


def full_name(name: str):
    has_adapter, x = False, name
    if name.startswith(ADAPTERMARKER):
        has_adapter, x = True, x.split("/")[-1]
    if x in ("AverageTitForTat", "atft", "at4t", "avgtitfortat"):
        x = "AverageTitForTat"
    if x in ("HardHeaded", "hh", "hardheaded"):
        x = "HardHeaded"
    if x in ("AgentK", "k", "agentk"):
        x = "AgentK"
    if x in ("Atlas3", "atlas3"):
        x = "Atlas3"
    if x in ("CUHKAgent", "cuhk"):
        x = "CUHKAgent"
    if x in ("gg", "GG", "agentgg"):
        x = "AgentGG"
    if x == "gatlas3":
        x = "Atlas3"
    if x in ("gagentk", "gk"):
        x = "AgentK"
    if x in ("ghardheaded", "ghh"):
        x = "HardHeaded"
    if x == "gcuhk":
        x = "CUHKAgent"
    if x == "caduceus":
        x = "Caduceus"
    if x in ("ntft", "nicetitfortat", "nt4t"):
        x = "NiceTitForTat"
    if x in ("aspiration", "asp"):
        x = "AspirationNegotiator"
    if x in ("concede", "conceder"):
        x = "ConcederTBNegotiator"
    if x in ("linear", "lin"):
        x = "LinearTBNegotiator"
    if x in ("linear", "linear"):
        x = "LinearTBNegotiator"
    if x == "micro":
        x = "MiCRONegotiator"
    if x in (
        "AspirationNegotiator",
        "MiCRONegotiator",
        "ConcederTBNegotiator",
        "LinearTBNegotiator",
    ):
        y = f"negmas.sao.{x}"
    elif x in (
        "AgentGG",
        "AverageTitForTat",
        "AgentK",
        "Atlas3",
        "HardHeaded",
        "CUHKAgent",
    ):
        return f"helpers.negotiators.{x}"
    else:
        y = f"negmas.genius.gnegotiators.{x}"
    if has_adapter:
        return f"{ADAPTERMARKER}/{y}"
    return y


ADAPTERMARKER = "__adapter__"
TAUNEGOTIATORSSHORT = {"can", "car", "cab", "wan", "war", "wab"}
TAUNEGOTIATORS = {_.upper() + "Negotiator" for _ in TAUNEGOTIATORSSHORT}


def full_tau_name(x: str):
    if x.lower() in TAUNEGOTIATORSSHORT:
        x = x.upper() + "Negotiator"
    if x in TAUNEGOTIATORS:
        return f"negmas.gb.negotiators.{x}"
    else:
        return f"{ADAPTERMARKER}/{full_name(x)}"


NTRIALS = MAX_TRIALS
NDOMAINS = None
HIDDEN_TIME_LIMIT = 72 * 60 * 60
SAO_TIME_LIMIT = 3 * 60
SAO_ROUNDS = 1.0
# NTRIALS = 1
# NDOMAINS = 2
# SAO_TIME_LIMIT = 10
MECHANISMS: dict[str, Type[Mechanism]] = dict(AO=SAOMechanism, TAU=TAUMechanism)
RESULTS_FILE_NAME = (RESULTS_PATH / "results.csv").absolute()


def safediv(a, b):
    return (a / b) if abs(b) > 1e-13 else 1.0 if abs(a) < 1e-13 else float("inf")


@dataclass
class RunInfo:
    key: tuple
    perm: tuple[int]
    domain_path: Path
    mechanism_name: str
    strategy_name: str
    domain_name: str
    mechanism: str
    first_strategy: str
    path: Path
    domain_info: dict[str, Any] | None
    trial: int
    mechanism_params: dict[str, Any]
    f0: float
    f1: float
    has_adapter1: bool
    has_adapter2: bool
    perm_index: int = 0
    first_strategy_params: dict[str, Any] = field(factory=dict)
    second_strategy_name: str | None = None
    second_strategy: str | None = None
    second_params: dict[str, Any] = field(factory=dict)
    previous_results: dict = dict()

    @property
    def allsao(self):
        lst = ("WAN", "WAB", "WAR", "CAR", "CAB", "CAN", "FAR", "BAR")
        return not any(_ in self.first_strategy for _ in lst) and (
            not self.second_strategy or not any(_ in self.first_strategy for _ in lst)
        )

    @property
    def anysao(self):
        lst = ("WAN", "WAB", "WAR", "CAR", "CAB", "CAN", "FAR", "BAR")
        return not any(_ in self.first_strategy for _ in lst) or (
            not self.second_strategy or not any(_ in self.first_strategy for _ in lst)
        )


def read_found(
    paths: list[Path],
    f0s: list[float],
    f1s: list[float],
    remove_failed: bool = True,
    update_stats: bool = False,
    verbose: int = 0,
) -> tuple[
    dict[tuple[str, str, str, str], int],
    list[dict],
    dict[tuple[str, str, str, str, int], dict],
]:
    extra = []
    for path in paths:
        if path.is_dir():
            extra += [_ for _ in path.glob("**/*.csv")]
    paths += extra
    dfs = []
    for path in paths:
        if not path.exists():
            if verbose > 0:
                print(f"[red]Cannot get records from {path} (does not exist)[/red]")
            continue
        if path.is_dir():
            continue
        try:
            dfs.append(pd.read_csv(path, dtype=DTYPES))  # type: ignore
        except EmptyDataError:
            continue
        except Exception:
            remove_repeated_header(path, verbose=verbose > 1, override=True)
            remove_invalid_lines(path, verbose=verbose > 1, override=True)
            correct_field_types(path, verbose=verbose > 1, override=True)
            try:
                dfs.append(pd.read_csv(path, dtype=DTYPES))  # type: ignore
            except Exception as e:
                if verbose > 0:
                    print(
                        f"[red]Cannot get records from {path} (invalid)[/red]: {str(e)}"
                    )
                continue
    if not dfs:
        return defaultdict(int), [], dict()
    data = pd.concat(dfs)
    data, _ = clean_frame(
        data=data,
        verbose=verbose > 1,
        override=verbose > 1,
        valid=dict(f0=[-1] + f0s, f1=[-1] + f1s),
    )
    if data is None:
        raise ValueError(
            f"Cannot get data to find existing runs even thoug we could read the paths at\n{paths}"
        )
    # data.loc[
    #     (data.domain_name.str.contains("@")) & (data.year.isin((2021, 2022))),
    #     "domain_name",
    # ] = (
    #     data.loc[
    #         (data.domain_name.str.contains("@")) & (data.year.isin((2021, 2022))),
    #         "domain_name",
    #     ]
    #     .str.split("@")
    #     .str[0]
    # )
    data["failed"] = data["failed"].fillna(True).astype(bool)
    data["perm_indx"] = data.perm_indx.astype(int)
    records = data.loc[~data["failed"], :].to_dict(orient="records")  # type: ignore
    found = defaultdict(int)
    prevs = dict()
    for record in records:
        if remove_failed and "failed" in record and record["failed"]:
            continue
        key = (
            record["mechanism_name"],
            record["strategy_name"],
            record["domain_name"],
            key_(record["perm_indx"], record["f0"], record["f1"]),
            # safe_key_(record["time_limit"]),
            # safe_key_(record["n_rounds"]),
        )
        prevs[tuple(list(key) + [found[key]])] = record
        found[key] += 1

    return found, records if not update_stats else [], prevs


def get_full_name(s1: str, s2: str | None) -> str:
    if s2 is None or s1 == s2:
        return f"{s1}"
    return f"{s1}-{s2}"


def run_once(
    info: RunInfo,
    file_name: Path,
    ntotal: int,
    strt,
    update_stats=False,
    save_agreement=True,
    extra_checks=True,
    debug=False,
    verbosity: int = 1,
    max_allowed_time=0,
    reserved_value_on_failure=False,
    n_retries=3,
    break_on_exception=False,
    ignore_irrational=False,
    timelimit_final_retry=False,
    save_fig=False,
    only2d=True,
    n_steps_to_register_failure: int = -1,
    f_steps_to_register_failure: float = -1,
) -> dict[str, Any] | None:
    _ = update_stats
    sname_full = get_full_name(info.strategy_name, info.second_strategy_name)
    _tick = perf_counter()
    if verbosity > 1:
        print(
            f"[blue]Entering RunOnce for: {info.mechanism_name} {sname_full}"
            f" on {info.path.name} ({info.f0}, {info.f1})"
            f" Perm {info.perm_index}"
            f" ... [/blue] at {datetime.now()}",
            flush=True,
        )
    if info.domain_info is None:
        info.domain_info = read_stats_of(
            info.domain_path, info.perm_index, info.f0, info.f1
        )
        dn = info.domain_path
        if ignore_irrational and not info.domain_info.get(
            IRKEY,
            check_rational(info.domain_info, dn, info.perm, info.f0, info.f1),
        ):
            if verbosity > 0:
                print(
                    f"\t[cyan]Ignoring Domain with no rational outcomes [/cyan] {(dn.name, info.f0, info.f1)}",
                    flush=True,
                )
            return None
    my_domain_info = info.domain_info
    try:
        if verbosity > 1:
            print(
                f"[yellow]Reading domain for: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [/yellow] after {humanize_time(perf_counter() - _tick)}",
                flush=True,
            )
        domain = Scenario.from_genius_folder(
            info.path, ignore_discount=True, ignore_reserved=False
        )
        try:
            year = int(info.path.parent.name[1:])
        except:
            year = None
        if domain is None:
            raise FileNotFoundError(f"Cannot load domain from {str(info.path)}")
        assert (
            info.perm_index == my_domain_info["perm_index"]
        ), f"{info.perm_index=} but {my_domain_info['perm_index']=}"
        n_rounds = info.mechanism_params.get("n_steps", None)
        time_limit = info.mechanism_params.get("time_limit", None)
        assert not (
            n_rounds and time_limit
        ), f"Cannnot pass both n.rounds ({n_rounds}) and time-limit ({time_limit})"
        assert not (
            info.mechanism_name.startswith("AO")
            and not time_limit
            and not info.mechanism_params.get("hidden_time_limit", None)
        ), f"Must pass either time-limit ({time_limit}) or hidden time limit for {info.mechanism_name}"
        if verbosity > 1:
            print(
                f"[yellow]Reading ORDINAL domain for: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ...[/yellow] after {humanize_time(perf_counter() - _tick)}",
                flush=True,
            )
        ordinal_path = Path(str(info.path).replace("scenarios", "rank_scenarios"))
        year_folder = info.path.parent.name
        dom_folder = info.path.name
        ordinal_base_name = "rank_scenarios"
        ordinal_path = (
            info.path.parent.parent.parent
            / ordinal_base_name
            / year_folder
            / dom_folder
        )
        try:
            ordinal_scenario = load_scenario(ordinal_path)
            ufuns_ordinal_ = ordinal_scenario.ufuns
        except Exception as e:
            if verbosity > 0:
                print(
                    f"[yellow][red]Cannot load ordinal scenarios[/red] from {ordinal_path} [/yellow]: {e}"
                )
            if break_on_exception:
                raise e
            ufuns_ordinal_ = [make_rank_ufun(_) for _ in domain.ufuns]
        if verbosity > 1:
            print(
                f"[yellow]Getting ufuns for: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [/yellow] after {humanize_time(perf_counter() - _tick)}",
                flush=True,
            )
        ufuns_ = get_ufuns_for(domain.ufuns, info.perm_index)
        ufuns_ = set_reservation(ufuns_, info.f0, info.f1)
        domain.ufuns = ufuns_
        if verbosity > 1:
            print(
                f"[yellow]Getting ORDINAL ufuns for: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [/yellow] after {humanize_time(perf_counter() - _tick)}",
                flush=True,
            )
        ufuns_ordinal_ = get_ufuns_for(ufuns_ordinal_, info.perm_index)
        ufuns_ordinal_ = set_reservation(ufuns_ordinal_, info.f0, info.f1)
        if extra_checks:
            if verbosity > 1:
                print(
                    f"[yellow]Extra checks for: {info.mechanism_name} {sname_full}"
                    f" on {info.path.name} ({info.f0}, {info.f1})"
                    f" Perm {info.perm_index}"
                    f" ... [/yellow] after {humanize_time(perf_counter() - _tick)}",
                    flush=True,
                )
            saved_reservs = tuple(my_domain_info["reserved_values"])
            reservs = tuple(_.reserved_value for _ in ufuns_)
            assert all(
                abs(a - b) < 1e-6 or (a < 1e-12 and b < 1e-12)
                for a, b in zip(reservs, saved_reservs, strict=True)
            ), f"Rotation of ufun reserved values did not go well for {info.path.name} {info.perm_index=}, {info.f0=}, {info.f1=}\n{reservs=}\n{saved_reservs=}"
        second = info.second_strategy if info.second_strategy else info.first_strategy
        second_name = (
            info.second_strategy_name
            if info.second_strategy_name
            else info.strategy_name
        )
        ssparams = (
            info.second_params
            if info.second_strategy_name
            else info.first_strategy_params
        )
        # sname_full = f"{info.strategy_name}-{info.second_strategy_name}"
        if verbosity > 1:
            print(
                f"[yellow]Starting: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [/yellow] at {datetime.now()} after {humanize_time(perf_counter() - _tick)}",
                flush=True,
            )
        elif verbosity > 0:
            print(
                f"[yellow]Starting: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [/yellow] at {datetime.now()}",
                flush=True,
            )
        stats = ScenarioStats(**my_domain_info["cardinal"])
        stats_ordinal = ScenarioStats(**my_domain_info["ordinal"])
        _full_start = perf_counter()
        dd = info.mechanism_params
        if info.anysao and "hidden_time_limit" not in dd.keys():
            dd["hidden_time_limit"] = min(
                domain.outcome_space.cardinality, HIDDEN_TIME_LIMIT
            )

        timeratio = 1
        if max_allowed_time:
            if dd.get("time_limit", None) is not None:
                timeratio = max(1, int(dd["time_limit"] / max_allowed_time + 0.5))
            for kk in ("time_limit", "hidden_time_limit"):
                if info.anysao and kk in dd.keys():
                    try:
                        dd[kk] = min(max_allowed_time, dd[kk])
                    except:
                        continue
        dd["dynamic_entry"] = False
        dd["enforce_issue_types"] = False
        dd["cast_offers"] = False
        dd["check_offers"] = False
        dd["id"] = unique_name(f"{info.domain_name}:{sname_full}")
        dd["name"] = unique_name(f"{info.domain_name}:{sname_full}")
        strategies = [info.first_strategy] + [second for _ in ufuns_[1:]]
        if any(_.startswith(ADAPTERMARKER) for _ in strategies):
            dd["n_steps"] = domain.outcome_space.cardinality + 1  # type: ignore
        failed, timelimited_for_final = True, False
        result: dict[str, Any] = dict()
        failed_after_init = False
        for _trial in range(n_retries):
            if (
                timelimit_final_retry
                and _trial >= n_retries - 1
                and any(_.startswith(ADAPTERMARKER) for _ in strategies)
            ):
                timelimited_for_final = True
                dd.pop("n_steps", None)
                dd["time_limit"] = FINAL_TRIAL_TIME
            m = instantiate(
                info.mechanism,
                outcome_space=domain.outcome_space,
                verbosity=verbosity - 2
                if domain.outcome_space.cardinality > 10_000
                else verbosity - 3,
                **dd,
            )
            agreement = None
            res = my_domain_info["ordinal_reserved_values"]
            resrank0, resrank1 = res[0], res[1]
            if extra_checks:
                res = my_domain_info["reserved_values"]
                reserved0, reserved1 = res[0], res[1]
                assert abs(ufuns_[0].reserved_value - reserved0) < 1e-6
                assert abs(ufuns_[1].reserved_value - reserved1) < 1e-6

            result: dict[str, Any] = dict(
                run_id=str(uuid4()),
                mechanism_name=info.mechanism_name,
                strategy_name=sname_full,
                first_strategy_name=info.strategy_name,
                second_strategy_name=second_name,
                domain_name=info.path.name[7:]
                if isint(info.path.name[:7])
                else info.path.name,
                year=year,
                n_outcomes=domain.outcome_space.cardinality,
                opposition=stats.opposition,
                time_limit=info.mechanism_params.get("time_limit", None),
                n_rounds=info.mechanism_params.get("n_steps", None),
                f0=info.f0,
                f1=info.f1,
                reserved0=ufuns_[0].reserved_value,
                reserved1=ufuns_[1].reserved_value,
                reserved=tuple(u.reserved_value for u in ufuns_),
                resrank0=resrank0,
                resrank1=resrank1,
                resrank=res,
                strategy_params=str(info.first_strategy_params)
                if info.first_strategy_params
                else "",
                second_strategy_params=str(ssparams) if ssparams else "",
                perm_indx=info.perm_index,
                max_pareto_welfare=max(sum(_) for _ in stats.pareto_utils),
                utility_ranges=stats.utility_ranges,
                rank_ranges=stats_ordinal.utility_ranges,
                ordinal_opposition=stats_ordinal.opposition,
            )
            if verbosity > 1:
                print(
                    f"[yellow]Negotiation Execution for: {info.mechanism_name} {sname_full}"
                    f" on {info.path.name} ({info.f0}, {info.f1})"
                    f" Perm {info.perm_index}"
                    f" ... [/yellow] after {humanize_time(perf_counter() - _tick)}",
                    flush=True,
                )
            try:
                _start = perf_counter()
                strategies = [info.first_strategy] + [second for _ in ufuns_[1:]]
                adapters = [info.has_adapter1] + [info.has_adapter2 for _ in ufuns_[1:]]
                sparams = [ssparams for _ in ufuns_]
                strategies[0], sparams[0] = (
                    info.first_strategy,
                    info.first_strategy_params,
                )
                for i, (has_adapter, ufun, s, sp) in enumerate(
                    zip(adapters, ufuns_, strategies, sparams)
                ):
                    if has_adapter:
                        full_class_name = s.split("/")[-1]
                        if full_class_name.startswith(GENIUSMARKER):
                            full_class_name = get_genius_proper_class_name(
                                full_class_name
                            )
                            added = m.add(
                                TAUNegotiatorAdapter(
                                    base=GeniusNegotiator(
                                        java_class_name=full_class_name, **sp
                                    ),
                                    name=f"{full_class_name.split('.')[-1]}{i:02}",
                                    id=f"{full_class_name.split('.')[-1]}{i:02}",
                                ),
                                ufun=ufun,
                            )
                        else:
                            assert (
                                full_class_name.split(".")[-1] not in TAUNATIVE
                            ), f"{full_class_name} is TAU native but adapted!!"
                            added = m.add(
                                TAUNegotiatorAdapter(
                                    base=instantiate(
                                        full_class_name,
                                        name=f"{full_class_name.split('.')[-1]}{i:02}",
                                        id=f"{full_class_name.split('.')[-1]}{i:02}",
                                        **sp,
                                    ),
                                    name=f"{full_class_name.split('.')[-1]}{i:02}",
                                    id=f"{full_class_name.split('.')[-1]}{i:02}",
                                ),
                                ufun=ufun,
                            )
                    elif s.startswith(GENIUSMARKER):
                        fnn = get_genius_proper_class_name(s)
                        added = m.add(
                            GeniusNegotiator(
                                java_class_name=fnn,
                                name=f"{fnn.split('.')[-1]}{i:02}",
                                id=f"{fnn.split('.')[-1]}{i:02}",
                                **sp,
                            ),
                            ufun=ufun,
                        )
                    else:
                        added = m.add(
                            instantiate(
                                s,
                                name=f"{s.split('.')[-1]}{i:02}",
                                id=f"{s.split('.')[-1]}{i:02}",
                                **sp,
                            ),
                            ufun=ufun,
                        )
                    if not added:
                        print(
                            f"[yellow]Failed to add {s}({sp if sp else ''}) at position {i}: {info.mechanism_name} {sname_full}"
                            f" on {info.path.name} ({info.f0}, {info.f1})"
                            f" Perm {info.perm_index} Adapt: ({info.has_adapter1}, {info.has_adapter2})"
                            f" ... [/yellow] after {humanize_time(perf_counter() - _tick)}",
                            flush=True,
                        )
                        raise ValueError(
                            f"Failed to add {s} at position {i} in negotiation on {info.path.name} ({info.perm_index}, {info.f0}, {info.f1})"
                        )

                negotiator_ids = m.nmi.negotiator_ids
                _start_run = perf_counter()
                try:
                    failed_after_init = False
                    if verbosity > 1:
                        run_with_feedback(
                            m,
                            f"[blue]{info.mechanism_name} {sname_full}"
                            f" {info.path.name} ({info.f0}, {info.f1}"
                            f" {info.perm_index})[/blue] {datetime.now()}",
                        )
                    else:
                        m.run()
                except Exception as e:
                    if (
                        # not break_on_exception
                        # and
                        n_steps_to_register_failure > 0
                        and m.current_step
                        >= min(n_steps_to_register_failure, m.outcome_space.cardinality)
                    ) or (
                        # not break_on_exception
                        # and
                        f_steps_to_register_failure > 0
                        and m.current_step
                        >= min(
                            f_steps_to_register_failure * m.outcome_space.cardinality,
                            m.outcome_space.cardinality,
                        )
                    ):
                        failed_after_init = True
                        estr = str(e).split("\n")[0]
                        print(
                            f"[cyan]Ignoring at trial [white]({_trial+1}/{n_retries})[/cyan]: {info.mechanism_name}, "
                            f"{sname_full} (Perm {info.perm_index}, f0={info.f0}, f1={info.f1}"
                            f" Adapt: [{info.has_adapter1}, {info.has_adapter2}])"
                            f", {info.domain_name} on step {m.current_step} [/white]: {estr}",
                            file=sys.stderr,
                        )
                    else:
                        raise e

                if save_fig:
                    fname = (
                        FIGS_FOLDER
                        / "sessions"
                        / (
                            unique_name(
                                f"{sname_full}-{info.mechanism_name}-{info.path.name}-{info.f0}-{info.f1}-{info.perm_index}",
                                sep=".",
                                add_time=True,
                                rand_digits=3,
                            )
                            + ".png"
                        )
                    )
                    try:
                        fname.parent.mkdir(exist_ok=True, parents=True)
                        m.plot(only2d=only2d, save_fig=True, path=fname, fast=True)
                    except Exception as e:
                        print(f"[red]Failed to plot and save to {fname.name} [/red]")
                elapsed = perf_counter() - _start
                elapsed_run = perf_counter() - _start_run
                agreement = m.agreement
                agreement_utils = tuple(ufun(agreement) for ufun in ufuns_)
                rvals = [ufun.reserved_value for ufun in ufuns_]
                if any([a + EPS < b for a, b in zip(agreement_utils, rvals)]):
                    print(
                        f"[red]NEGATIVE Advantage detected in {info.mechanism_name} {sname_full}"
                        f" {info.path.name} ({info.f0}, {info.f1}"
                        f" {info.perm_index})[/red] agreement: {agreement} ({agreement_utils=}, {rvals=})",
                    )
                agreement_ranks = tuple(ufun(agreement) for ufun in ufuns_ordinal_)
                dists = calc_outcome_distances(agreement_utils, stats)
                optim = calc_outcome_optimality(dists, stats, estimate_max_dist(ufuns_))
                dists_ordinal = calc_outcome_distances(agreement_ranks, stats_ordinal)
                optim_ordinal = calc_outcome_optimality(
                    dists_ordinal, stats_ordinal, estimate_max_dist(ufuns_ordinal_)
                )
                failed = False
                offer_sets = [set(m.negotiator_offers(_)) for _ in negotiator_ids]
                nn = len(offer_sets)
                noffers = [len(_) for _ in offer_sets]
                try:
                    nreceived = [
                        len(
                            set.union(
                                *(
                                    [
                                        offer_sets[i]
                                        for i in itertools.chain(
                                            range(0, i), range(i + 1, nn)
                                        )
                                    ]
                                )
                            )
                        )
                        for i in range(nn)
                    ]
                except TypeError as e:
                    nreceived = [0 for _ in range(nn)]
                try:
                    nseen = len(set.union(*([s for s in offer_sets])))
                except TypeError as e:
                    nseen = 0
                result.update(
                    dict(
                        time=elapsed * timeratio,
                        run_time=elapsed_run * timeratio,
                        succeeded=agreement is not None,
                        failed=False,
                        noffers0=noffers[0],
                        noffers1=int(mean(noffers[1:]) + 0.5),
                        noffers=str(noffers),
                        nreceived0=nreceived[0],
                        nreceived1=int(mean(nreceived[1:]) + 0.5),
                        nreceived=str(nreceived),
                        nseen=nseen,
                        **asdict(dists),
                        **{f"ordinal_{k}": v for k, v in asdict(dists_ordinal).items()},
                        **asdict(optim),
                        **{f"ordinal_{k}": v for k, v in asdict(optim_ordinal).items()},
                        steps=m.current_step,
                        failed_after_init=failed_after_init,
                        trial=_trial,
                    )
                )
                break
            except Exception as e:
                if break_on_exception:
                    raise e
                estr = str(e).split("\n")[0]
                try:
                    stepp = m.current_step
                except:
                    stepp = "unknown"
                print(
                    f"[magenta]Failed ({_trial+1}/{n_retries}): {info.mechanism_name}, "
                    f"{sname_full} (Perm {info.perm_index}, f0={info.f0}, f1={info.f1}"
                    f" Adapt: [{info.has_adapter1}, {info.has_adapter2}])"
                    f", {info.domain_name}[/magenta]  on step {stepp} : {estr}",
                    file=sys.stderr,
                )
        else:
            failed = True
            # failed on all trials. Just save a failure record
            agreement = None
            agreement_utils = tuple(ufun(agreement) for ufun in ufuns_)
            agreement_ranks = tuple(ufun(agreement) for ufun in ufuns_ordinal_)
            if any(
                [
                    a + EPS < b
                    for a, b in zip(
                        agreement_utils, [ufun.reserved_value for ufun in ufuns_]
                    )
                ]
            ):
                print(
                    f"[red]NEGATIVE Advantage detected in {info.mechanism_name} {sname_full}"
                    f" {info.path.name} ({info.f0}, {info.f1}"
                    f" {info.perm_index})[/red] Failure (agreement set to None).",
                )
            dists = calc_outcome_distances(agreement_utils, stats)
            optim = calc_outcome_optimality(dists, stats, estimate_max_dist(ufuns_))
            dists_ordinal = calc_outcome_distances(agreement_ranks, stats_ordinal)
            optim_ordinal = calc_outcome_optimality(
                dists_ordinal, stats_ordinal, estimate_max_dist(ufuns_ordinal_)
            )
            result.update(
                dict(
                    time=None,
                    run_time=None,
                    failed=not reserved_value_on_failure,
                    succeeded=False,
                    noffers0=None,
                    noffers1=None,
                    noffers="",
                    nreceived0=None,
                    nreceived1=None,
                    nreceived="",
                    nseen=None,
                    **asdict(dists),
                    **{f"ordinal_{k}": v for k, v in asdict(dists_ordinal).items()},
                    **asdict(optim),
                    **{f"ordinal_{k}": v for k, v in asdict(optim_ordinal).items()},
                    steps=0,
                    failed_after_init=False,
                    trial=sys.maxsize,
                )
            )

        if verbosity > 1:
            print(
                f"[yellow]Finished Running for: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [/yellow] after {humanize_time(perf_counter() - _tick)}",
                flush=True,
            )
        elapsed_full = perf_counter() - _full_start
        welfare = sum(agreement_utils)
        agreement_ranks = tuple(get_rank(ufun, agreement) for ufun in ufuns_)
        welfare_rank = sum(agreement_ranks)
        # todo add util0, rank0, util1, rank1 here
        result.update(
            dict(
                full_time=elapsed_full,
                agreement_utils=agreement_utils,
                agreement_ranks=agreement_ranks,
                welfare=welfare,
                welfare_rank=welfare_rank,
            )
        )
        if save_agreement:
            result.update(dict(agreement=agreement))
        # for k in ("n_outcomes", "opposition"):
        #     result[k] = my_domain_info[k]

        if verbosity > 1:
            print(
                f"[white]Adding record for: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [/white] after {humanize_time(perf_counter() - _tick)}",
                flush=True,
            )
        add_records(file_name, [result])
        cond = (
            "[red]Fail[/red]"
            if failed
            else f"[blue]None {agreement_utils}[/blue]"
            if not agreement
            else f"[green]OK {agreement_utils}[/green]"
        )

        # remove_repeated_header(file_name, verbose=False, override=True)
        # remove_invalid_lines(file_name, verbose=False, override=True)
        # correct_field_types(file_name, verbose=False, override=True)
        ndone = count_runs(file_name)

        if verbosity > 0 or failed:
            print(
                f"{cond}: {info.mechanism_name} {sname_full}"
                f" on {info.path.name} ({info.f0}, {info.f1})"
                f" Perm {info.perm_index}"
                f" ... [bold blue]DONE[/bold blue] at {datetime.now()} ({humanize_time(elapsed_full)}) "
                f"{ndone}/{ntotal} ({ndone/ntotal:4.2%}) ETA {humanize_time(ntotal *(perf_counter() - strt) / (max(ndone, 1)))}"
                f"{' [timelimited last]' if timelimited_for_final else ''}",
                flush=True,
                file=sys.stderr if failed else sys.stdout,
            )
        return result
    except Exception as e:
        if break_on_exception:
            raise e
        print(
            f"[red bold]DID NOT STAART[/red bold]: "
            f" on {info.path.name} ({info.f0}, {info.f1})"
            f" Perm {info.perm_index} {str(e)}",
            file=sys.stderr,
        )
        if debug:
            raise e
        return None


def get_reserved(domain: Scenario, r0: float, r1: float) -> tuple[float, float]:
    outcomes = list(domain.outcome_space.enumerate_or_sample())
    results, r = [], [r0, r1]
    for i in range(2):
        u = domain.ufuns[i]
        utils = sorted([u(_) for _ in outcomes], reverse=True)
        limit = int(r[0] * len(utils))
        results.append(utils[limit] - 0.0001)
    return tuple(results)


def get_strategy_name(
    sfullname: str,
) -> tuple[bool, type[Negotiator] | Callable[[], Negotiator], str, str]:
    has_adapter = sfullname.startswith(ADAPTERMARKER)
    name_without_adapter = sfullname.split("/")[-1]
    if name_without_adapter.startswith(GENIUSMARKER):
        parts = name_without_adapter.split(":")
        assert parts[0] == GENIUSMARKER
        classname, sn = parts[2], parts[1]
        if has_adapter:
            s = lambda: TAUNegotiatorAdapter(
                base=GeniusNegotiator(java_class_name=classname)
            )
        else:
            s = lambda: GeniusNegotiator(java_class_name=classname)
        return has_adapter, s, sfullname, sn
    s = get_class(name_without_adapter)
    sn = s.__name__
    return has_adapter, s, sfullname, sn


def check_rational(
    stats: dict[str, Any], path: Path, perm: Sequence[int], f0: float, f1: float
) -> bool:
    if "has_rational" in stats.keys():
        return stats["has_rational"]
    scenario = Scenario.from_genius_folder(
        path, ignore_discount=True, ignore_reserved=False
    )
    assert (
        scenario is not None
    ), f"Cannot determine if domain has rational for ({perm}, {f0}, {f1}) or find scenario at {str(path)}"
    ufuns = [scenario.ufuns[p] for p in perm]
    set_reservation(ufuns, f0, f1)
    fr = calc_rational_fraction(
        ufuns, tuple(scenario.outcome_space.enumerate_or_sample())
    )
    return fr > 1e-12


def isadapted(x: str):
    return "/" not in x


def run_all(
    domains: dict[str, Scenario],
    n_trials: int,
    time_limit: int,
    max_allowed_time: int,
    n_rounds: float,
    f0s: list[float],
    f1s: list[float],
    file_name: Path,
    done_paths: list[Path],
    strategies: dict[str, tuple[str, ...]],
    serial: bool,
    ignore_irrational: bool,
    max_cores: int,
    mechanisms: dict[str, Type[Mechanism], ...] = MECHANISMS,  # type: ignore
    strategy_params: dict[str, dict[str, Any] | None] | None = None,
    title: str = "Negotiations",
    order=True,
    reversed=False,
    pure_tau: bool = False,
    impure_tau: bool = False,
    pure_ao: bool = False,
    impure_ao: bool = False,
    ufun_permutations: bool = True,
    norun: bool = False,
    update_stats: bool = False,
    save_agreement: bool = True,
    verbosity: int = 1,
    debug: bool = False,
    extra_checks: bool = False,
    reserved_value_on_failure: bool = False,
    n_retries: int = 3,
    break_on_exception=False,
    fast_start=True,
    ignored_combinations=IGNORED_COMBINATIONS,
    timelimit_final_retry=False,
    round_robin: bool = False,
    save_fig=False,
    only2d=True,
    n_steps_to_register_failure: int = -1,
    f_steps_to_register_failure: float = -1,
):
    if verbosity > 0:
        print(f"Getting completed runs from:\n{done_paths}")
    n_total = 0
    strategies_ao = strategies["AO"]
    strategies_tau = strategies["TAU"]
    puritydict = dict(TAU=(pure_tau, impure_tau), AO=(pure_ao, impure_ao))

    if round_robin:
        ntau = max(1, len(strategies_tau) - 1)
    elif pure_tau and impure_tau:
        ntau = len(strategies_tau) ** 2
    elif pure_tau:
        ntau = len(strategies_tau)
    else:  # impure_tau
        ntau = len(strategies_tau) ** 2 - len(strategies_tau)

    if round_robin:
        nao = max(1, len(strategies_ao) - 1)
    elif pure_ao and impure_ao:
        nao = len(strategies_ao) ** 2
    elif pure_ao:
        nao = len(strategies_ao)
    else:  # impure_ao
        nao = len(strategies_ao) ** 2 - len(strategies_ao)
    n_per_ao = int(time_limit > 0) + int(n_rounds > 0)
    per_scenario = len(domains) * n_trials * max(1, len(f0s) * len(f1s) + 1)
    n_estimated_total = (n_per_ao * nao + ntau) * per_scenario
    print(
        f"Estimated {n_estimated_total} total runs: ({nao=} * {n_per_ao=} + {ntau=}) * {len(domains)=} * {n_trials=} * {max(1, len(f0s) * len(f1s) + 1)=} ({per_scenario=})"
    )

    if strategy_params is None:
        strategy_params = dict()
    runs = []

    remove_repeated_header(file_name, verbose=verbosity > 1, override=True)
    remove_invalid_lines(file_name, verbose=verbosity > 1, override=True)
    correct_field_types(file_name, verbose=verbosity > 1, override=True)
    if verbosity > 0:
        print("Reading found ...", end="", flush=True)
    found, _, prevs = read_found(
        [file_name] + done_paths,
        update_stats=update_stats,
        f0s=f0s,
        f1s=f1s,
        verbose=verbosity > 1,
    )
    if len(found):
        print(
            f"[green]DONE found {len(found)} keys with {sum(found.values())} runs [/green]",
            flush=True,
        )
    else:
        print(f"[yellow]DONE found {len(found)} keys with 0 runs [/yellow]", flush=True)

    for dn, d in track(
        domains.items(), total=len(domains), description="Reading Stats"
    ):
        allstats = read_all_stats(Path(dn))
        if allstats is None:
            continue
        fractions = list(product(f0s, f1s))
        fractions = [(-1, -1)] + fractions
        if allstats is not None:
            try:
                n_outcomes = allstats["n_outcomes"]
            except Exception as e:
                try:
                    n_outcomes = int(Path(dn).name[:7])
                except Exception as e:
                    estr = "\n\t".join(
                        str(e).split("\n")[: N_EXCEPTION_LINES if verbosity else 1]
                    )
                    print(
                        f"[bold red]Cannot read n_outcomes for [/bold red]{dn}[red]{estr}[/red]",
                        file=sys.stderr,
                    )
                    if break_on_exception:
                        raise e
                    continue
        else:
            n_outcomes = int(Path(dn).name[:7])
        n_steps = int(n_rounds * n_outcomes)
        # TAU strategies take a maximum of n_outcomes+1 rounds. If the number of rounds
        # is less than that, increase it by 1 (mostly to make --rounds=1.0 actually mean n_outcomes + 1)
        if n_steps:
            n_steps += 1

        if ufun_permutations:
            ufun_perms = list(PERMUTAION_METHOD(range(len(d.ufuns))))
        else:
            ufun_perms = [tuple(range(len(d.ufuns)))]
        nxt = dict(zip(found.keys(), itertools.repeat(0)))

        for perm_indx, perm in enumerate(ufun_perms):
            # rotate utility functions according to the perm
            # d.ufuns = [d.ufuns[_] for _ in perm]
            for mn, m in mechanisms.items():
                if issubclass(m, SAOMechanism):
                    params = []
                    if time_limit > 0:
                        params.append(
                            dict(
                                time_limit=time_limit,
                                n_steps=None,
                                name="AOt",
                                extra_callbacks=True,
                            )
                        )
                    if n_steps > 0:
                        params.append(
                            dict(
                                n_steps=n_steps,
                                hidden_time_limit=HIDDEN_TIME_LIMIT,
                                time_limit=None,
                                name="AOr",
                                extra_callbacks=True,
                            )
                        )
                else:
                    params = []
                    params.append(
                        dict(
                            name="TAU0",
                            n_steps=n_steps if n_steps else n_outcomes + 1,
                            extra_callbacks=True,
                            parallel=False,
                        )
                    )
                for p in params:
                    new_mn: str = p.pop("name")  # type: ignore
                    pure, impure = puritydict[mn]
                    _st = strategies[mn]
                    if isinstance(_st, str):
                        _st = (_st,)
                    if not _st:
                        combinations = []
                    elif round_robin:
                        combinations = (
                            zip(_st[:-1], _st[1:], strict=True)
                            if len(_st) > 1
                            else [(_st[0], _st[0])]
                        )
                    elif pure and impure:
                        combinations = itertools.product(_st, _st)
                    elif pure:
                        combinations = [(_, _) for _ in _st]
                    else:
                        combinations = list(
                            (a, b) for a, b in itertools.product(_st, _st) if a != b
                        )

                    for combination in combinations:
                        adapt1, _, s1fname, sn = get_strategy_name(combination[0])
                        adapt2, _, s2fname, sn2 = get_strategy_name(combination[1])
                        sname_full = get_full_name(sn, sn2)
                        for f0, f1 in fractions:
                            if is_ignored(
                                new_mn,
                                sn,
                                sn2,
                                Path(dn).name,
                                dn,
                                f0,
                                f1,
                                patterns=ignored_combinations,
                            ):
                                if verbosity > 0:
                                    print(
                                        f"[magenta]Ignoring {sn} v {sn2} on {Path(dn).name} ({f0}, {f1})[/magenta]"
                                    )
                                continue
                            stats_key = key_(perm_indx, f0, f1)
                            if not fast_start:
                                try:
                                    dstats = allstats[stats_key]
                                except KeyError:
                                    print(
                                        f"\t[red]stats data exists but cannot find stats for {(dn.split('/')[-1], perm_indx, f0, f1)} with key {stats_key} [/red]",
                                        flush=True,
                                        file=sys.stderr,
                                    )
                                    continue
                            else:
                                dstats = None

                            if dstats is not None:
                                if ignore_irrational and not dstats.get(
                                    IRKEY,
                                    check_rational(dstats, Path(dn), perm, f0, f1),
                                ):
                                    if verbosity > 0:
                                        print(
                                            f"\t[cyan]Ignoring Domain with no rational outcomes [/cyan] {(dn.split('/')[-1], f0, f1)}",
                                            flush=True,
                                        )
                                    continue
                            # this_time_limit = p.get("time_limit", "nan")
                            # this_n_rounds = p.get("n_rounds", "nan")
                            key = (
                                new_mn,
                                sname_full,
                                dn.split("/")[-1][7:],
                                stats_key,
                                # safe_key_(this_time_limit),
                                # safe_key_(this_n_rounds),
                            )
                            if update_stats:
                                n_remaining = found[key]
                            else:
                                n_remaining = n_trials - found[key]

                            n_total += n_trials
                            if n_remaining < 1:
                                continue

                            for t in range(n_remaining):
                                runs.append(
                                    RunInfo(
                                        perm=perm,
                                        key=key,
                                        domain_path=Path(dn),
                                        mechanism_name=new_mn,
                                        strategy_name=sn,
                                        domain_name=Path(dn).name,
                                        mechanism=get_full_type_name(m),
                                        first_strategy=s1fname,
                                        path=Path(dn),
                                        domain_info=dstats,
                                        trial=t,
                                        mechanism_params=p,
                                        f0=f0,
                                        f1=f1,
                                        first_strategy_params=strategy_params.get(  # type: ignore
                                            s1fname, dict()
                                        ),
                                        second_strategy=s2fname,
                                        # second_strategy=get_negotiator_type_name(s2, second_strategy_full_name),  # type: ignore
                                        second_strategy_name=sn2,
                                        second_params=strategy_params.get(  # type: ignore
                                            s2fname, dict()
                                        ),
                                        perm_index=perm_indx,
                                        previous_results=prevs[  # type: ignore
                                            tuple(list(key) + [nxt[key]])
                                        ]
                                        if update_stats
                                        else dict(),
                                        has_adapter1=adapt1,
                                        has_adapter2=adapt2,
                                    )
                                )
                                if update_stats:
                                    nxt[key] += 1
    del found
    runs = list(runs)
    n = len(runs)
    print(f"Summary of {title}", flush=True)
    print(f"Should run {len(runs)} tasks (of {n_total})", flush=True)
    print(
        f"Domains: {len(domains)}, Trials: {n_trials}\nAOP Strategies: {len(strategies['AO'])}"
        f", TAU Strategies: {len(strategies['TAU'])}, AOP combinations: {nao}, TAU Combinations: {ntau}\n"
        f"Timelimits: {int(time_limit > 0)}, Rounds: {int(n_rounds > 0)}, "
        f"f0s: {len(f0s)}, f1s: {len(f1s)}, F trials: {max(1, len(f0s) * len(f1s) + 1)}",
        flush=True,
    )
    if n < 1:
        print(
            f"[bold green]---------------- Nothing to run for '{title}' ------------------ [/bold green]"
        )
        return
    try:
        sleep(10)
    except:
        pass
    if order:
        runs = sorted(
            runs, key=lambda x: (x.domain_name, x.strategy_name, x.mechanism_name)
        )
        if reversed:
            runs.reverse()
    else:
        shuffle(runs)
    if norun:
        print(f"Summary of {title}", flush=True)
        print(f"Should run {len(runs)} tasks (of {n_total})", flush=True)
        print(
            f"Domains: {len(domains)}, Trials: {n_trials}\nAOP Strategies: {len(strategies['AO'])}"
            f", TAU Strategies: {len(strategies['TAU'])}, AOP combinations: {nao}, TAU Combinations: {ntau}\n"
            f"Timelimits: {int(time_limit > 0)}, Rounds: {int(n_rounds > 0)}, "
            f"f0s: {len(f0s)}, f1s: {len(f1s)}, F trials: {max(1, len(f0s) * len(f1s) + 1)}",
            flush=True,
        )
        print("You passed --norun. Will not run anything", flush=True)
        print(
            f"Will NOT run the {len(runs)} sessions on {set(_.path.name for _ in runs)}",
            flush=True,
        )
        print("-----------------------------------------------------")
        return
    print(f"Submitting {len(runs)} tasks (of {n_total})", flush=True)
    strt = perf_counter()
    desc = f"Stats: {title}" if update_stats else title
    print(
        f"[bold yellow]---------------- Starting {title} ({serial=},{max_cores=}) ------------------ [/bold yellow]"
    )
    if serial or max_cores < 0:
        for info in track(runs, total=len(runs), description=desc):
            run_once(
                info,
                file_name,
                n_total,
                strt,
                update_stats,
                save_agreement,
                debug=debug,
                verbosity=verbosity,
                extra_checks=extra_checks,
                max_allowed_time=max_allowed_time,
                reserved_value_on_failure=reserved_value_on_failure,
                n_retries=n_retries,
                break_on_exception=break_on_exception,
                ignore_irrational=ignore_irrational,
                timelimit_final_retry=timelimit_final_retry,
                save_fig=save_fig,
                only2d=only2d,
                n_steps_to_register_failure=n_steps_to_register_failure,
                f_steps_to_register_failure=f_steps_to_register_failure,
            )
    else:
        futures = []
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for info in runs:
                futures.append(
                    pool.submit(
                        run_once,
                        info,
                        file_name,
                        n_total,
                        strt,
                        update_stats,
                        save_agreement,
                        debug=debug,
                        verbosity=verbosity,
                        extra_checks=extra_checks,
                        max_allowed_time=max_allowed_time,
                        reserved_value_on_failure=reserved_value_on_failure,
                        n_retries=n_retries,
                        break_on_exception=break_on_exception,
                        ignore_irrational=ignore_irrational,
                        timelimit_final_retry=timelimit_final_retry,
                        save_fig=save_fig,
                        only2d=only2d,
                        n_steps_to_register_failure=n_steps_to_register_failure,
                        f_steps_to_register_failure=f_steps_to_register_failure,
                    )
                )
            print("Running ...", flush=True)
            for f in track(as_completed(futures), total=len(futures), description=desc):
                f.result()
    print(f"[bold blue]---------------- DONE {title} ------------------ [/bold blue]")


def main(
    base: Path = SCENRIOS_PATH,
    trials: int = NTRIALS,
    outcomelimit: int = sys.maxsize,
    minoutcomes: int = 0,
    ngroups: int = 1,
    group: int = 0,
    ndomains: int = sys.maxsize,
    domain: str = "",
    timelimit: int = SAO_TIME_LIMIT,
    max_allowed_time: int = 0,
    rounds: float = SAO_ROUNDS,
    f0: list[float] = F0,
    f1: list[float] = F1,
    path: Path = RESULTS_FILE_NAME,
    unique_file_name: bool = True,
    rand_result_file: bool = False,
    append: bool = True,
    ao: list[str] = [
        "micro",
        "asp",
        "atlas3",
        "agentk",
        "caduceus",
        "hardheaded",
        "cuhk",
        "ntft",
        "gg",
        "conceder",
        "linear",
    ],
    tau: list[str] = [
        "cab",
        "war",
        "micro",
        "asp",
        "atlas3",
        "agentk",
        "caduceus",
        "hardheaded",
        "cuhk",
        "ntft",
        "gg",
        "conceder",
        "linear",
        # "car",
        # "wab",
        # "wan",
        # "can",
    ],
    aoyear: int = -1,
    max_cores: int = 0,
    year: list[int] = (2013,),  # type: ignore
    title: str = "Negotiations",
    serial: bool = False,
    order: bool = True,
    reversed: bool = False,
    ignore_irrational: bool = True,
    pure: bool = None,  # type: ignore
    impure: bool = None,  # type: ignore
    pure_tau: bool = True,
    impure_tau: bool = True,
    peryear_tau: bool = True,
    pure_ao: bool = True,
    impure_ao: bool = True,
    allcombinations: bool = False,
    ufun_permutations: bool = True,
    override: bool = False,
    fast_start: bool = False,
    small: bool = False,
    norun: bool = False,
    verbose: int = 1,
    debug: bool = False,
    update_stats: bool = False,
    extra_checks: bool = False,
    save_agreement: bool = True,
    reserved_value_on_failure: bool = False,
    done: list[Path] = [],  # type: ignore
    n_retries: int = 3,
    timelimit_final_retry: bool = False,
    finalists_only: bool = False,
    genius10: bool = True,
    winners_only: bool = False,
    extra_peryear_agents: bool = True,
    break_on_exception: bool = False,
    ignore_warnings: bool = False,
    ignore_common_failures: bool = False,
    round_robin: bool = False,
    save_fig: bool = False,
    only2d: bool = True,
    steps_to_register_failure: int = -1,
    relative_steps_to_register_failure: float = -1,
    add_cab_war: bool = True,
):
    """Runs an experiment with any strategies for TAU or AOP."""
    if allcombinations:
        pure = impure = True
    if pure is not None:
        pure_tau = pure_ao = pure
    if impure is not None:
        impure_tau = impure_ao = impure
    if pure_tau and pure_ao:
        pure = True
    if impure_tau and impure_ao:
        impure = True
    if ignore_warnings:
        warnings.filterwarnings("ignore")
    print(f"Starting Experiment '{title}' at {datetime.now()}", flush=True)
    if small:
        if len(ao) > 1:
            ao = [ao[0]]
        if len(tau) > 1:
            tau = [tau[0]]
        ndomains = 1
        trials = 1
        timelimit = 0
        rounds = 10
        serial = True
        unique_file_name = False
        path = Path() / "small.csv"

    yearlbl = ".".join([str(_)[2:] for _ in year]) if year else ""
    if unique_file_name:
        # rand_result_file = True
        if pure and impure:
            path = (
                path.parent
                / f"{path.stem}-{gethostname()}-allcombinations{yearlbl}.csv"
            )
        elif pure:
            path = path.parent / (f"{path.stem}-{gethostname()}-pure{yearlbl}.csv")
        elif impure:
            path = path.parent / (f"{path.stem}-{gethostname()}-impure{yearlbl}.csv")
        elif pure_tau and impure_tau:
            path = (
                path.parent
                / f"{path.stem}-{gethostname()}-allcombinations-tau{yearlbl}.csv"
            )
        elif pure:
            path = path.parent / (f"{path.stem}-{gethostname()}-pure-tau{yearlbl}.csv")
        elif impure:
            path = path.parent / (
                f"{path.stem}-{gethostname()}-impure-tau{yearlbl}.csv"
            )
        elif pure_ao and impure_ao:
            path = (
                path.parent
                / f"{path.stem}-{gethostname()}-allcombinations-ao{yearlbl}.csv"
            )
        elif pure:
            path = path.parent / (f"{path.stem}-{gethostname()}-pure-ao{yearlbl}.csv")
        elif impure:
            path = path.parent / (f"{path.stem}-{gethostname()}-impure-ao{yearlbl}.csv")
        else:
            raise ValueError(
                f"You cannot pass --no-pure and --no-impure together for both TAU and AO"
            )
    if not append:
        path = path.parent / (
            unique_name(path.stem, add_time=True, rand_digits=3, sep=".") + ".csv"
        )
    if rand_result_file:
        path = path.parent / f"{path.stem}{random.randint(1, 20)}{path.suffix}"

    print(f"Will save results to: {path}")
    f0 = [_ for _ in f0 if _ > 0]
    f1 = [_ for _ in f1 if _ > 0]
    clean_results(path, override=True, verbose=verbose > 1, f0s=f0, f1s=f1)
    if override and path.exists():
        print("Removing old runs")
        path.unlink(missing_ok=True)

    files = []
    if year is None or year[0] == 0:
        year = ALLYEARS
        print(f"Will work for years {year}")
    base = base.absolute()
    if verbose:
        print(f"Using scenarios in {base}")
    if base.name.startswith("y2"):
        files += list(
            sorted(
                [
                    _
                    for _ in base.glob("*")
                    if _.is_dir() and not _.name.startswith("y2")
                ]
            )
        )
    for current in year:
        if current <= 0:
            base_path = base
            break
        else:
            base_path = base / f"y{current}"
        if verbose > 1:
            print(f"Found: {[_.name for _ in base_path.glob('*')]} at {base_path}")
        files += list(sorted([_ for _ in base_path.glob("*")]))
    files = list(set([_ for _ in files if _.is_dir()]))
    if verbose > 1:
        print(
            f"Found {len(files)} files at {base}. Will work with maximum of {ndomains} domains "
            f"of outcomes between {minoutcomes} and {outcomelimit}",
            end="",
        )
        if ngroups:
            print(f"(group {group} out of {ngroups})")
        else:
            print()
    if ndomains is not None and ndomains < len(files):
        files = files[:ndomains]

    if outcomelimit is not None and outcomelimit:
        files = [_ for _ in files if int(_.name[:7]) <= outcomelimit]
    if minoutcomes is not None:
        files = [_ for _ in files if int(_.name[:7]) >= minoutcomes]
    if domain:
        files = [_ for _ in files if _.name.endswith(domain)]
    if ngroups > 0 and group > -1 and len(files) > 0:
        assert ngroups > group or ngroups < len(
            files
        ), f"{ngroups=} must be larger than {group=} and smaller than {len(files)=}"
        files = sorted(files)
        nfiles = len(files)
        per_group = int(nfiles // ngroups)
        remaining = nfiles % ngroups
        assert per_group >= 1
        # mygroup = range(
        #     remaining + per_group * group if group > 0 else 0,
        #     remaining + per_group * (group + 1),
        # )
        mygroup = (remaining + group, nfiles, ngroups)
        extension = (0, remaining) if group == 0 else (0, 0)
        print(
            f"Will use {max(0, mygroup[1] - mygroup[0]) + max(0, extension[1] - extension[0])} files of {nfiles} (i.e. group {group} {ngroups} = {mygroup} (extension {extension}))"
        )
        assert mygroup[1] > mygroup[0], f"Invalid range of files {mygroup=}"
        files = [files[_] for _ in range(*extension)] + [
            files[_] for _ in range(*mygroup)
        ]
        print(f"Will use {len(files)} files: {mygroup=}")

    all_domains = {
        name: s
        for name, s in (
            (
                str(f),
                Scenario.from_genius_folder(
                    f, ignore_discount=True, ignore_reserved=False
                ),
            )
            for f in track(files, total=len(files), description="Loading Domains")
            if f.is_dir()
        )
        if s is not None
    }
    print(f"Will use {len(all_domains)} domains")
    strategy_params = dict()
    if "none" in ao:
        aostrategies = tuple()
    else:
        aostrategies = tuple(full_name(_) for _ in ao)
    if aoyear is not None and aoyear >= 2010:
        if extra_peryear_agents:
            extra = get_all_anac_agents(aoyear, finalists_only, winners_only, genius10)
        else:
            extra = []
        if verbose > 0:
            print(
                f"Will use {len(extra)} Geinus Negotiators from year {year}:\n{[_.split(':')[-1] for _ in extra]}"
            )
            if len(aostrategies):
                print(
                    f"Will use {len(aostrategies)} Negotiators [red]NOT[/red] from year {year}:\n{[_.split(':')[-1] for _ in aostrategies]}"
                )

        aostrategies = tuple(list(aostrategies) + extra)

    if "none" in tau:
        taustrategies = tuple()
    else:
        # use strategies of the given year adding WAR and CAB to them
        if aoyear is not None and aoyear >= 2010 and peryear_tau:
            extra = [
                f"{ADAPTERMARKER}/{_}"
                for _ in get_all_anac_agents(
                    aoyear, finalists_only, winners_only, genius10
                )
            ]
            if add_cab_war:
                taustrategies = tuple(
                    extra + [full_tau_name(_) for _ in ["war", "cab"]]
                )
                if verbose > 0:
                    print(
                        f"Will use the {len(taustrategies)} strategies on [blue]TAU[/blue] for domains of year {aoyear}:\n{[_.split(':')[1] for _ in extra] + ['WAR', 'CAB']}"
                    )
            else:
                taustrategies = tuple(extra)
                if verbose > 0:
                    print(
                        f"Will use the {len(taustrategies)} strategies on [blue]TAU[/blue] for domains of year {aoyear}:\n{[_.split(':')[1] for _ in extra]}"
                    )

        else:
            taustrategies = tuple(full_tau_name(_) for _ in tau)
    strategies: dict[str, tuple[str, ...]] = dict(  # type: ignore
        AO=aostrategies,
        TAU=taustrategies,
    )
    print(f"Will save results to {str(path)}")
    run_all(
        domains=all_domains,
        n_trials=trials,
        time_limit=timelimit,
        max_allowed_time=max_allowed_time,
        n_rounds=rounds,
        serial=serial,
        f0s=f0,
        f1s=f1,
        file_name=path,
        mechanisms=MECHANISMS,
        strategies=strategies,
        ignore_irrational=ignore_irrational,
        max_cores=max_cores,
        order=order,
        reversed=reversed,
        strategy_params=strategy_params,
        pure_ao=pure_ao,
        impure_ao=impure_ao,
        pure_tau=pure_tau,
        impure_tau=impure_tau,
        ufun_permutations=ufun_permutations,
        norun=norun,
        update_stats=update_stats,
        save_agreement=save_agreement,
        title=title,
        verbosity=verbose,
        debug=debug,
        extra_checks=extra_checks,
        reserved_value_on_failure=reserved_value_on_failure,
        done_paths=done,
        n_retries=n_retries,
        break_on_exception=break_on_exception,
        fast_start=fast_start,
        ignored_combinations=IGNORED_COMBINATIONS if ignore_common_failures else dict(),
        timelimit_final_retry=timelimit_final_retry,
        round_robin=round_robin,
        save_fig=save_fig,
        only2d=only2d,
        n_steps_to_register_failure=steps_to_register_failure,
        f_steps_to_register_failure=relative_steps_to_register_failure,
    )


if __name__ == "__main__":
    typer.run(main)
