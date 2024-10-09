import csv
import subprocess
import sys
import warnings

# import warnings
from collections import deque
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from socket import gethostname
from time import perf_counter
from typing import Any, Callable, Iterable, Literal, Sequence

from pandas.errors import DtypeWarning, SettingWithCopyWarning

warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import yaml
from attr import asdict
from negmas.genius.ginfo import get_anac_agents
from negmas.helpers import get_class, humanize_time
from negmas.helpers.inout import dump, get_full_type_name, is_nonzero_file, load
from negmas.inout import Scenario
from negmas.negotiators import Negotiator
from negmas.outcomes import Outcome
from negmas.outcomes.outcome_space import CartesianOutcomeSpace
from negmas.preferences import BaseUtilityFunction, UtilityFunction
from negmas.preferences.ops import (
    ScenarioStats,
    calc_outcome_distances,
    calc_outcome_optimality,
    calc_reserved_value,
    calc_scenario_stats,
    estimate_max_dist,
    is_rational,
    itertools,
    make_rank_ufun,
)
from pandas.api.types import is_float_dtype
from pandas.compat import os
from pandas.core.frame import functools
from pandas.errors import EmptyDataError
from rich import print
from scipy.stats import median_test, ttest_ind, ttest_rel, wilcoxon

STRATEGY_NAME_MAP = {
    "KLH": "HardHeaded",
    "Groupn": "AgentNeo",
    "AgentBuyogMain": "AgentBuyog",
    "kawaii": "Kawaii",
    "Pnegotiator": "PNegotiator",
    "AgentHP2main": "AgentHP2",
    "BRAMAgent": "BramAgent",
    "BRAMAgent2": "BramAgent2",
    "ParsAgent": (2016, "ParsAgent2"),
}
DOMAIN_NAME_MAP = {
    "0000384IS_BT_Acquisition": "0000384Acquisition",
    "0000004group7-movie": "0000004Movie",
    "0000018group3-bank_robbery": "0000018BankRobbery",
    "0000216group11-car_purchase": "0000216CarPurchase",
    "0000240group5-car_domain": "0000240CarDomain",
    "0000243group2-new_sporthal": "0000243NewSporthal",
    "0000448group4-zoning_plan": "0000448ZoningPlan",
    "0000864group10-building_construction": "0000864BuildingConstruction",
    "0000972group6-tram": "0000972Tram",
    "0001024group8-holiday": "0001024Holiday",
    "0001200group2-dinner": "0001200Dinner",
    "0001600group9-vacation": "0001600Vacation",
    "0002250group1-university": "0002250University",
    "0002304group12-symposium": "0002304Symposium",
    "0023040group2-politics": "0023040Politics",
    "0240000group9-killer_robot": "0240000KillerRobot",
    "0000008nGent": "0000008ElectricVehicle",
    "0000009Atlas3": "0000009TriangularFight",
    "0000025Farma": "0000025TwF",
    "0000080Caduceus": "0000080SmartGrid",
    "0000100ClockWork": "0000100SmartGridDomain",
    "0000240SYAgent": "0000240JapanTrip2016",
    "0000320AgentSmith": "0000320DomainAce",
    "0000612parsAgent2": "0000612PEnergy",
    "0000625AgentLightSmartGrid": "0000625SmartEnergeGridSmall",
    "0000984Grandma": "0000984SmartEnergyGridMedium",
    "0001280myAgent": "0001280KDomain",
    "0004250YXAgent": "0004250SmartEnergyGridLarge",
    "0007200Maxoops": "0007200WindFarm",
    "0035840Terra": "0035840NewDomain",
    "0390625parsCat": "0390625EnergyLarge",
    "IS_BT_Acquisition": "Acquisition",
    "group7-movie": "Movie",
    "group3-bank_robbery": "BankRobbery",
    "group11-car_purchase": "CarPurchase",
    "group5-car_domain": "CarDomain",
    "group2-new_sporthal": "NewSporthal",
    "group4-zoning_plan": "ZoningPlan",
    "group10-building_construction": "BuildingConstruction",
    "group6-tram": "Tram",
    "group8-holiday": "Holiday",
    "group2-dinner": "Dinner",
    "group9-vacation": "Vacation",
    "group1-university": "University",
    "group12-symposium": "Symposium",
    "group2-politics": "Politics",
    "group9-killer_robot": "KillerRobot",
    "nGent": "ElectricVehicle",
    "Atlas3": "TriangularFight",
    "Farma": "TwF",
    "Caduceus": "SmartGrid",
    "ClockWork": "SmartGridDomain",
    "SYAgent": "JapanTrip2016",
    "AgentSmith": "DomainAce",
    "parsAgent2": "PEnergy",
    "AgentLightSmartGrid": "SmartEnergeGridSmall",
    "Grandma": "SmartEnergyGridMedium",
    "myAgent": "KDomain",
    "YXAgent": "SmartEnergyGridLarge",
    "Maxoops": "WindFarm",
    "Terra": "NewDomain",
    "parsCat": "EnergyLarge",
}
EPS = 1e-3

GENIUSMARKER = "genius"
TAUVARIANTS = [
    "WAN",
    "CAN",
    "FAN",
    "WAB",
    "FAB",
    "CAR",
    "FAR",
    "WANNegotiator",
    "CANNegotiator",
    "FANNegotiator",
    "WABNegotiator",
    "FABNegotiator",
    "CARNegotiator",
    "FARNegotiator",
]
TAUNATIVE = [
    "WAN",
    "CAN",
    "FAN",
    "WAB",
    "CAB",
    "FAB",
    "WAR",
    "CAR",
    "FAR",
    "WANNegotiator",
    "CANNegotiator",
    "FANNegotiator",
    "WABNegotiator",
    "CABNegotiator",
    "FABNegotiator",
    "WARNegotiator",
    "CARNegotiator",
    "FARNegotiator",
]
BASE_FOLDER = Path(__file__).parent.parent.parent
SCENARIOS_FOLDER = BASE_FOLDER / "scenarios"
FIGS_FOLDER = BASE_FOLDER / "figs"
RANK_SCENARIOS_FOLDER = BASE_FOLDER / "rank_scenarios"
RESULTS_FOLDER = BASE_FOLDER / "results"
STATS_FOLDER = BASE_FOLDER / "stats"
TABLES_FOLDER = BASE_FOLDER / "tables"

CONDITIONS = [
    "mechanism_name",
    "strategy_name",
    "first_strategy_name",
    "second_strategy_name",
    "domain_name",
    "f0",
    "f1",
    "perm_indx",
    "time_limit",
    "n_rounds",
]
URESOLUTION = 3

F0 = [1.0]
F1 = [0.1, 0.5, 0.9, 1.0]

ONLY_ZERO_RESERVE = ["AgentK", "AgentGG", "NiceTfT", "Caduceus"]

STATS_AGENT_INDIVIDUAL = [
    "Utility",
    "Advantage",
    "Rank",
    "O_Advantage",
    "IRUB",
    "Privacy",
    "Uniqueness",
    "AgentScore",
]
STATS_FINAL_INDIVIDUAL = [
    "ScoreWithIR",
    "FinalScore",
    "DesignerScore",
    "OutcomeScore",
    "Score",
    "IRUB",
    "Privacy",
    "Uniqueness",
    "N_Fairness",
    "O_N_Fairness",
    "K_Fairness",
    "O_K_Fairness",
    "RK_Fairness",
    "O_RK_Fairness",
    "Completeness",
    "Optimality",
    "Fairness",
    "Welfare",
    "Rounds",
    "Time",
]
STATS_AGENT = [
    "Utility",
    "Rank",
    # "IRUB",
    "Privacy",
    "Uniqueness",
]
STATS_FINAL = [
    "Completeness",
    "Optimality",
    "Fairness",
    "Welfare",
    # "IRUB",
    "Privacy",
    "Uniqueness",
    # "Rounds",
    # "Time",
]

STATS_ORDINAL = [
    "Completeness",
    "O_Optimality",
    "Fairness",
    "O_RK_Fairness",
    "O_Welfare",
    "O_N_Fairness",
    "Rank1",
    "Rank2",
    "O_Advantage1",
    "O_Advantage2",
    # "O_R. Welfare",
]

STATS_CARDINAL = [
    "Completeness",
    "Optimality",
    "RK_Fairness",
    "N_Fairness",
    "Welfare",
    "Utility1",
    "Utility2",
    "Advantage1",
    "Advantage2",
]
STATS_TIMING = [
    "Rounds",
    "Time",
]
STATS = [
    "Advantage1",
    "Advantage2",
    "Completeness",
    "DesignerScore",
    "Fairness",
    "FinalScore",
    "ScoreWithIR",
    "IRUB",
    "IRUB1",
    "IRUB2",
    "Privacy",
    "Privacy1",
    "Privacy2",
    "K_Fairness",
    "N_Fairness",
    "O_Advantage1",
    "O_Advantage2",
    "O_K_Fairness",
    "O_N_Fairness",
    "O_Optimality",
    "O_RK_Fairness",
    "O_Welfare",
    "Optimality",
    "OutcomeScore",
    "RK_Fairness",
    "Rank1",
    "Rank2",
    "Rounds",
    "Score",
    "Time",
    "Uniqueness",
    "Uniqueness1",
    "Uniqueness2",
    "Utility1",
    "Utility2",
    "Welfare",
]
NONSTATS = ["Condition", "RationalFraction"]
MAX_TRIALS = 1
MIN_TRIALS = 1
PRECISION = 3

BASEKEY = "_base"
STATSFILE = "stats.json"
BASIC_STATS_FILE = "basic_stats.json"
STATSBASE = "stats"
FRKEY = "fraction_rational"
IRKEY = "has_rational"

maxInt = sys.maxsize

DTYPES = {
    "agreement": "string",
    "agreement_ranks": "string",
    "agreement_utils": "string",
    "domain_name": "category",
    "f0": "float32",
    "f1": "float32",
    "failed": "bool",
    "first_strategy_name": "category",
    "full_time": "float32",
    "kalai_dist": "float32",
    "kalai_optimality": "float32",
    "max_pareto_welfare": "float32",
    "max_welfare": "float32",
    "max_welfare_optimality": "float32",
    "mechanism_name": "category",
    "modified_kalai_dist": "float32",
    "modified_kalai_optimality": "float32",
    "n_outcomes": "Int32",
    "n_rounds": "Int32",
    "nash_dist": "float32",
    "nash_optimality": "float32",
    "noffers": "string",
    "noffers0": "Int32",
    "noffers1": "Int32",
    "nreceived": "string",
    "nreceived0": "Int32",
    "nreceived1": "Int32",
    "nseen": "Int32",
    "opposition": "float32",
    "ordinal_kalai_dist": "float32",
    "ordinal_kalai_optimality": "float32",
    "ordinal_max_welfare": "float32",
    "ordinal_max_welfare_optimality": "float32",
    "ordinal_modified_kalai_dist": "float32",
    "ordinal_modified_kalai_optimality": "float32",
    "ordinal_nash_dist": "float32",
    "ordinal_nash_optimality": "float32",
    "ordinal_opposition": "float32",
    "ordinal_pareto_dist": "float32",
    "ordinal_pareto_optimality": "float32",
    "pareto_dist": "float32",
    "pareto_optimality": "float32",
    "perm_indx": "Int32",
    "rank_ranges": "string",
    "reserved": "string",
    "reserved0": "float32",
    "reserved1": "float32",
    "resrank0": "float32",
    "resrank1": "float32",
    "run_time": "float32",
    "second_strategy_name": "category",
    "second_strategy_params": "string",
    "steps": "float32",
    "strategy_name": "category",
    "strategy_params": "string",
    "succeeded": "bool",
    "time": "float32",
    "utility_ranges": "string",
    "welfare": "float32",
    "welfare_rank": "float32",
    "year": "Int32",
}
DTYPESFINAL = {
    "Domain": "category",
    "Mechanism": "category",
    "Protocol": "category",
    "Condition": "category",
    "Strategy": "category",
    "Strategy1": "category",
    "Strategy2": "category",
    "Type": "category",
    "Bilateral": "bool",
    "Controlled": "bool",
    "HasRational": "bool",
    "Advantage1": "float32",
    "Advantage2": "float32",
    "Completeness": "float32",
    "DesignerScore": "float32",
    "Fairness": "float32",
    "FinalScore": "float32",
    "ScoreWithIR": "float32",
    "IRUB": "float32",
    "IRUB1": "float32",
    "IRUB2": "float32",
    "Privacy": "float32",
    "Privacy1": "float32",
    "Privacy2": "float32",
    "K_Fairness": "float32",
    "N_Fairness": "float32",
    "O_Advantage1": "float32",
    "O_Advantage2": "float32",
    "O_K_Fairness": "float32",
    "O_N_Fairness": "float32",
    "O_Optimality": "float32",
    "O_RK_Fairness": "float32",
    "O_Welfare": "float32",
    "Optimality": "float32",
    "OutcomeScore": "float32",
    "RK_Fairness": "float32",
    "Rank1": "float32",
    "Rank2": "float32",
    "RationalFraction": "float32",
    "Rational1": "float32",
    "Rational2": "float32",
    "RelativeRounds": "float32",
    "Score": "float32",
    "Time": "float32",
    "Uniqueness": "float32",
    "Uniqueness1": "float32",
    "Uniqueness2": "float32",
    "Utility1": "float32",
    "Utility2": "float32",
    "Welfare": "float32",
    "Rounds": "float64",
    "time_limit": "Int32",
    "Agents": "int32",
    "Outcomes": "int32",
    "Year": "Int32",
    "MaxRounds": "Int32",
    "AgentScore": "float32",
    "Difficulty": "category",
    "Size": "category",
    "SizeOrder": "int32",
}
DTYPESAGENTS = {
    "Mechanism": "category",
    "RationalFraction": "float32",
    "Completeness": "float32",
    "Bilateral": "bool",
    "Domain": "category",
    "Outcomes": "int32",
    "Controlled": "bool",
    "Condition": "category",
    "HasRational": "bool",
    "Time": "float32",
    "Rounds": "float32",
    "MaxRounds": "float32",
    "time_limit": "float32",
    "Strategy": "category",
    "Rational": "float32",
    "IRUB": "float32",
    "Privacy": "float32",
    "Uniqueness": "float32",
    "Rank": "float32",
    "O_Advantage": "float32",
    "Utility": "float32",
    "Advantage": "float32",
    "StrategyPartner": "category",
    "RationalPartner": "float32",
    "IRUBPartner": "float32",
    "PrivacyPartner": "float32",
    "UniquenessPartner": "float32",
    "RankPartner": "float32",
    "O_AdvantagePartner": "float32",
    "UtilityPartner": "float32",
    "AdvantagePartner": "float32",
    "Starting": "bool",
    "Protocol": "category",
    "Type": "category",
    "Difficulty": "category",
    "Size": "category",
    "SizeOrder": "int32",
}
FINALREMOVE = [
    # "agreement",
    # "agreement_ranks",
    # "agreement_utils",
    "full_time",
    "kalai_dist",
    "max_pareto_welfare",
    "max_welfare",
    "modified_kalai_dist",
    "nash_dist",
    "noffers0",
    "noffers1",
    "noffers",
    "nreceived",
    "nreceived0",
    "nreceived1",
    "nseen",
    "opposition",
    "ordinal_kalai_dist",
    "ordinal_max_welfare",
    "ordinal_modified_kalai_dist",
    "ordinal_nash_dist",
    "ordinal_opposition",
    "ordinal_pareto_dist",
    "pareto_dist",
    "perm_indx",
    "rank0",
    "rank1",
    "rank_ranges",
    "reserved",
    # "reserved0",
    # "reserved1",
    "resrank0",
    "resrank1",
    # "run_time",
    "second_strategy_params",
    # "steps",
    "strategy_params",
    # "succeeded",
    "util0",
    "util1",
    "utility_ranges",
    "welfare",
    "welfare_rank",
]

while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)


def remove_invalid_lines(
    path: Path, override: bool = True, verbose: bool = True
) -> int:
    if not path.exists():
        return 0
    with open(path, "r") as f:
        lines = f.readlines()
    if not lines:
        return 0
    n0 = len(lines[0].split(","))
    if verbose:
        print(f"Found {n0} fields in the first line")
    to_remove = []
    numeric_columns = [
        k for k, v in DTYPES.items() if v not in ("category", "string", "bool")
    ]
    # [
    #     "year",
    #     "n_outcomes",
    #     "opposition",
    #     "time_limit",
    #     "n_rounds",
    #     "f0",
    #     "f1",
    #     "reserved0",
    #     "reserved1",
    #     "resrank0",
    #     "resrank1",
    #     "perm_indx",
    #     "time",
    #     "run_time",
    #     "pareto_dist",
    #     "nash_dist",
    #     "kalai_dist",
    #     "modified_kalai_dist",
    #     "max_welfare",
    #     "ordinal_pareto_dist",
    #     "ordinal_nash_dist",
    #     "ordinal_kalai_dist",
    #     "ordinal_modified_kalai_dist",
    #     "ordinal_max_welfare",
    #     "pareto_optimality",
    #     "nash_optimality",
    #     "kalai_optimality",
    #     "modified_kalai_optimality",
    #     "max_welfare_optimality",
    #     "ordinal_pareto_optimality",
    #     "ordinal_nash_optimality",
    #     "ordinal_kalai_optimality",
    #     "ordinal_modified_kalai_optimality",
    #     "ordinal_max_welfare_optimality",
    #     "full_time",
    #     "steps",
    #     "ordinal_opposition",
    # ]
    with open(path, "r") as f:
        for i, fields in enumerate(csv.reader(f)):
            if i == 0:
                n = len(fields)
                assert n == n0, f"First line has {n} fields but we estimated {n0}"
                columns = fields
                existing_numeric = set(numeric_columns).intersection(set(columns))
                non_existing = set(DTYPES.keys()).difference(set(columns))
                if len(non_existing):
                    print(f"[red]Columns {non_existing} were not found[/red] in {path}")
                numeric_indices = [columns.index(_) for _ in existing_numeric]
                continue
            m = len(fields)
            if m != n0:
                if verbose:
                    print(
                        f"[red] Found {m} fields in the line {i} (expected {n0})[/red]"
                    )
                to_remove.append(i)
                continue
            for indx in numeric_indices:  # type: ignore
                try:
                    _ = float(fields[indx]) if fields[indx].strip() else 0.0
                except:
                    if verbose:
                        print(f"[red] Field {indx} ({columns[indx]}) has value {fields[indx]} which is not numeric[/red]")  # type: ignore
                    to_remove.append(i)

    if not to_remove:
        return 0
    to_remove = set(to_remove)
    path.parent.mkdir(parents=True, exist_ok=True)
    if override:
        lines = [_ for i, _ in enumerate(lines) if i not in to_remove]
        with open(path, "w") as f:
            f.writelines(lines)
    else:
        with open(path.parent / f"{path.stem}-invalid.txt", "w") as f:
            f.writelines([f"{i}\n" for i in to_remove])
    return len(to_remove)


# def remove_incomplete(
#     path: Path | None = None,
#     data: pd.DataFrame | None = None,  # type: ignore
#     min_trials: int = 0,
#     count_failed: bool = True,
#     verbose: bool = True,
# ) -> pd.DataFrame | None:
#     assert not (path is None and data is None)
#     if data is None:
#         assert path is not None
#         if not path.exists():
#             return None
#         try:
#             data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
#         except EmptyDataError:
#             return None
#     badcols = set(["index"] + [f"level_{i}" for i in range(10)])
#
#     # we cannot group by floats so we need to convert things to int and convert them back
#     conditions = [
#         "mechanism_name",
#         "domain_name",
#         "f0",
#         "f1",
#         "perm_indx",
#         "time_limit",
#         "n_rounds",
#     ]
#     n_records = len(data)
#     if verbose:
#         print(f"Found {n_records} records ... ", flush=True, end="")
#     allcols = [_ for _ in data.columns if _ not in badcols]
#     data: pd.DataFrame
#     data = data[allcols]  # type: ignore
#     if not count_failed:
#         # data = data.loc[~data.failed, :]
#         data = remove_failed_records(data, explicit=True, implicit=False)
#     # I intentionally ignore time_limit and n_rounds here otherwise I will have to treat each mechanism independently
#     for col in conditions:
#         if data[col].dtype != float:
#             continue
#         data[f"_orig{col}"] = data[col]
#         data[col] = (URESOLUTION * data[col]).fillna(-1000).astype(int)
#
#     def _process(data, title=""):
#         strategies = set(data.strategy_name.unique())
#         n_strategies = len(strategies)
#         counts_first = data[conditions + ["first_strategy_name"]].gropuby(conditions)["first_strategy_name"].count().to_dict()
#         counts_second = data[conditions + ["second_strategy_name"]].gropuby(conditions)["second_strategy_name"].count()
#         counts = defaultdict(int)
#         for d in (counts_first, counts_second):
#             for k, v in d.items():
#                 counts[k] += v
#         counts = {k: v for k, v in counts if v > 0}
#         n_found = len(counts)
#         if n_found == n_strategies:
#             return
#         if title and verbose:
#             print(title)
#             print(counts)
#
#
#     # convert condition columns back to float if needed
#     for col in conditions:
#         if f"_orig{col}" in data.columns:
#             data[col] = data[f"_orig{col}"]
#             data.drop(f"_orig{col}", axis=1, inplace=True)
#     return data


def remove_extra(
    path: Path | None = None,
    data: pd.DataFrame | None = None,  # type: ignore
    max_trials: int = 0,
    min_trials: int = 0,
    remove_failed: bool = True,
    remove_implicit_failures: bool = False,
    outcomelimit: int = 0,
    verbose: bool = True,
    minoutcomes: int = 0,
) -> pd.DataFrame | None:
    assert not (path is None and data is None)
    if data is None:
        assert path is not None
        if not path.exists():
            return None
        try:
            data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
            mm = len(data)
            if outcomelimit:
                data = data.loc[data.n_outcomes <= outcomelimit, :]
                if verbose and mm > len(data):
                    print(
                        f"  Kept {len(data)} records with outcome-limit of {outcomelimit} out of ({mm}) [Removed [red]{mm - len(data)}[/red]]"
                    )
            mm = len(data)
            if minoutcomes:
                data = data.loc[data.n_outcomes >= minoutcomes, :]
                if verbose and mm > len(data):
                    print(
                        f"  Kept {len(data)} records with minimum outcomes of {minoutcomes} out of ({mm}) [Removed [red]{mm - len(data)}[/red]]"
                    )
        except EmptyDataError:
            return None
        except:
            print(
                f"[yellow]Failed reading {path} for remove-extra ... trying to correct[/yellow]"
            )
            remove_repeated_header(path, override=True, verbose=verbose)
            remove_invalid_lines(path, override=True, verbose=verbose)
            try:
                correct_field_types(path, verbose=verbose, override=True)
            except Exception as e:
                print(f"[red]Failed in correcting data types for {path}[/red]")
            try:
                data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
                print(f"[green]Corrected {path} in remove-extra.[/green]")
            except EmptyDataError:
                return None
            except Exception as e:
                print(e)
                try:
                    data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
                    print(
                        f"[yellow]Read without types for {path} in clean-frame.[/yellow]"
                    )
                except Exception as e:
                    print(
                        f"[red]Failed in correcting {path} for clean-frame[/red]: {e}"
                    )
                    return None
    badcols = set(["index"] + [f"level_{i}" for i in range(10)])
    existing = {c: set(data[c].unique()) for c in []}
    existing_warn = {
        c: set(data[c].unique())
        for c in ["mechanism_name", "strategy_name", "domain_name"]
    }

    def assert_no_major_deletion(x: pd.DataFrame):
        failures = dict()
        for k, e in existing.items():
            f = e.difference(set(x[k].unique()))
            if f:
                failures[k] = f
        if failures:
            raise ValueError(f"Completely removed values for\n {failures}")
        failures = dict()
        for k, e in existing_warn.items():
            f = e.difference(set(x[k].unique()))
            if f:
                failures[k] = f
        if verbose and failures:
            print(f"[yellow]Completely removed values for\n {failures}[/yellow]")

    # we cannot group by floats so we need to convert things to int and convert them back
    for col in CONDITIONS:
        if not pd.api.types.is_float_dtype(data[col].dtype) and col not in (
            "n_rounds,"
        ):
            continue
        if verbose:
            print(f"Converting {col} from {data[col].dtype} to int")
        data[f"_orig{col}"] = data[col]
        data[col] = (URESOLUTION * data[col]).fillna(-1000).astype(np.int32)
    assert_no_major_deletion(data)
    n_records = len(data)
    if verbose:
        print(f"Found {n_records} records ... ", flush=True, end="")
    allcols = [_ for _ in data.columns if _ not in badcols]
    data: pd.DataFrame
    data = data[allcols]  # type: ignore
    data = remove_failed_records(
        data, explicit=remove_failed, implicit=remove_implicit_failures
    )
    assert_no_major_deletion(data)
    succeeded = data.loc[~data.failed, :]
    failed = data.loc[data.failed, :]

    def limit_trials(x: pd.DataFrame, n: int) -> pd.DataFrame:
        assert_no_major_deletion(x)
        y = x.groupby(CONDITIONS).tail(n).reset_index()
        assert_no_major_deletion(y)
        # if verbose:
        #     print(x.groupby(CONDITIONS).tail(n).count())
        if "index" in y.columns:
            y = y[[_ for _ in y.columns if _ != "index"]]
        if min_trials and len(y) and len(y) < min_trials:
            y = y.sample(min_trials, replace=True)
        assert_no_major_deletion(y)
        return y  # type: ignore

    if max_trials:
        succeeded = limit_trials(succeeded, max_trials)
        assert_no_major_deletion(succeeded)
    if not remove_failed:
        failed = limit_trials(failed, 1)
        succeeded = pd.concat((succeeded, failed))
    assert_no_major_deletion(succeeded)
    # convert condition columns back to float if needed
    succeeded = succeeded.copy()
    for col in CONDITIONS:
        if f"_orig{col}" in succeeded.columns:
            succeeded[col] = succeeded[f"_orig{col}"]
            succeeded.drop(f"_orig{col}", axis=1, inplace=True)
    assert_no_major_deletion(succeeded)
    if verbose:
        if n_records == len(succeeded):
            print(f" [green]all OK[/green]", flush=True)
        else:
            print(
                f"remove [red]{n_records - len(succeeded)}[/red] "
                f"extra/failed records keeping {len(succeeded)}",
                flush=True,
            )
    return succeeded


def is_genius_folder(d: Path, check_stats: bool = True):
    if not d.is_dir():
        return False
    if check_stats and not is_nonzero_file(d / "stats.json"):
        return False
    try:
        scenario = Scenario.from_genius_folder(d, ignore_discount=True)
    except:
        scenario = None
    return scenario is not None


def find_domains(
    year: int,
    outcomelimit: int = 0,
    base: Path = SCENARIOS_FOLDER,
    minoutcomes: int = 0,
) -> list[str]:
    if year:
        base /= f"y{year}"
    paths = [_ for _ in base.glob("**/*")]
    return [
        _.name
        for _ in paths
        if is_genius_folder(_)
        and (not outcomelimit or int(_.name[:7]) <= outcomelimit)
        and (not minoutcomes or int(_.name[:7]) >= minoutcomes)
    ]


def get_dirs(
    base: Path,
    outcomelimit: int = sys.maxsize,
    ndomains: int = sys.maxsize,
    order: bool = True,
    reversed: bool = False,
    minoutcomes: int = 0,
):
    base = Path(base).absolute()
    base = base.absolute()
    dirs = []
    alldirs = [base] + sorted(base.glob("**/*"))
    for dir in alldirs:
        if not dir.is_dir():
            continue
        if len(list(dir.glob("*.xml"))) < 3:
            continue
        try:
            n = int(dir.name[:7])
        except:
            n = None
        if n is not None and outcomelimit and n > outcomelimit:
            continue
        if n is not None and minoutcomes and n < minoutcomes:
            continue
        dirs.append(dir.absolute())
    if order:
        dirs = sorted(dirs, key=lambda x: x.name)
        if reversed:
            dirs.reverse()
    if len(dirs) > ndomains:
        dirs = dirs[:ndomains]
    return dirs


# def get_max_util(rational_outcomes_only=False):
#     results = dict()
#
#     paths = [Path() / "domains"]
#     for d in SCENRIOS_PATH.glob("*"):
#         if not d.is_dir():
#             continue
#         paths.append(d)
#     for d in paths:
#         for f in d.glob("*"):
#             if not f.is_dir():
#                 continue
#             stats = load(f / STATSFILE)
#             if not stats:
#                 print(
#                     f"[red]No stats for {f.absolute().relative_to(Path().absolute())}[/red]"
#                 )
#                 continue
#             maxs = stats["maxs"]
#             for i, perm in enumerate(PERMUTAION_METHOD(maxs)):
#                 results[(f.name, i)] = perm
#     return results


def get_ufuns_for(
    ufuns: Sequence[UtilityFunction], perm_indx: int = 0
) -> tuple[UtilityFunction, ...]:
    if perm_indx == 0:
        return tuple(ufuns)
    return list(PERMUTAION_METHOD(ufuns))[perm_indx]


# def get_rationality():
#     results = dict()
#     paths = [Path() / "domains"]
#     for d in SCENRIOS_PATH.glob("*"):
#         if not d.is_dir():
#             continue
#         paths.append(d)
#
#     for d in paths:
#         for f in d.glob("*"):
#             if not f.is_dir():
#                 continue
#             infos = load(f / "info.json")
#             for info in infos:
#                 results[(f.name, info["f0int"], info["f1int"])] = info["has_rational"]
#     return results


# def get_rational_fraction():
#     results = dict()
#     paths = [Path() / "domains"]
#     for d in SCENRIOS_PATH.glob("*"):
#         if not d.is_dir():
#             continue
#         paths.append(d)
#
#     for d in paths:
#         for f in d.glob("*"):
#             if not f.is_dir():
#                 continue
#             infos = load(f / "info.json")
#             for info in infos:
#                 if (f.name, *info["reserved"]) in results:
#                     ValueError(
#                         f"{(f.name, *info['reserved'])} already found in {str(f.absolute().relative_to(Path().absolute()))}"
#                     )
#                 results[(f.name, *info["reserved"])] = info["f_rational"]
#     return results


def get_ranks(ufun: UtilityFunction, outcomes: list[Outcome | None]) -> list[float]:
    assert ufun.outcome_space is not None
    assert ufun.outcome_space.is_discrete()
    alloutcomes = list(ufun.outcome_space.enumerate_or_sample())
    n = len(alloutcomes)
    vals: list[tuple[float, Outcome | None]]
    vals = [(ufun(_), _) for _ in alloutcomes]
    ordered = sorted(vals, reverse=True)
    # insert null into its place
    r = ufun.reserved_value
    loc = n
    if r is None:
        r = float("-inf")
    else:
        for k, (u, o) in enumerate(ordered):
            if u < r:
                loc = k
                break
    if loc == n:
        ordered.append((r, None))
    else:
        ordered.insert(loc, (r, None))

    ordered = list(zip(range(n, -1, -1), ordered, strict=True))
    # mark outcomes with equal utils with the same rank
    for i, (second, first) in enumerate(zip(ordered[1:], ordered[:-1], strict=True)):
        if abs(first[1][0] - second[1][0]) < 1e-10:
            ordered[i + 1] = (first[0], second[1])
    results = []
    for outcome in outcomes:
        for v in ordered:
            k, _ = v
            u, o = _
            if o == outcome:
                results.append(k / n)
                break
        else:
            raise ValueError(f"Could not find {outcome}")
    return results


def get_rank(ufun: UtilityFunction, outcome: Outcome | None) -> float:
    return get_ranks(ufun, [outcome])[0]


def adjust_tex(path: Path, pdf: bool = False):
    import warnings

    warnings.filterwarnings("ignore")
    path = Path(path)
    map = {
        "multicolumn{2}{l}": "multicolumn{2}{c}",
    }
    with open(path, "r") as f:
        lines = f.readlines()
    orient = "landscape"
    size = 1
    for i, line in enumerate(lines):
        line = line.replace("n_o", "n-o")
        lines[i] = line.replace("_", "").replace("n-o", "n_o")

    lines = (
        [
            f"\\documentclass[a{size}paper,{orient}]{{article}}\n"
            f"\\usepackage[a{size}paper,{orient}, margin=0in]{{geometry}}\n"
            f"\\usepackage{{booktabs}}\n"
            f"\\usepackage{{longtable}}\n"
            f"\\begin{{document}}\n"
        ]
        + lines
        + ["\\end{document}\n"]
    )
    newlines = []
    for line in lines:
        for k, v in map.items():
            line = line.replace(k, v)
        newlines.append(line.replace("tabular", "longtable"))
        if "std" in line:
            newlines.append("\\midrule\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving tex for {str(path)}", flush=True)
    with open(path, "w") as f:
        f.writelines(newlines)

    if not pdf:
        return
    print(f"Making PDF for {str(path)}", end="", flush=True)
    current = Path.cwd()
    os.chdir(path.parent)
    subprocess.run(["pdflatex", str(path.name)], capture_output=True)
    subprocess.run(["pdflatex", str(path.name)], capture_output=True)
    subprocess.run(["pdflatex", str(path.name)], capture_output=True)
    os.chdir(current)
    print(" DONE", flush=True)


def clean_data(
    files: list[Path],
    output: Path,
    max_trials: int,
    failures: bool = False,
    implicit_failures: bool = True,
) -> pd.DataFrame:
    x = []
    for file_path in files:
        clean_results(file_path)
        try:
            x.append(pd.read_csv(file_path, dtype=DTYPES))  # type: ignore
        except EmptyDataError:
            continue
    data = pd.concat(x, ignore_index=True)
    data: pd.DataFrame  # type: ignore
    data = data.remove_failed_records(
        data, implicit=not implicit_failures, explicit=not failures
    )

    # we cannot group by floats so we need to convert things to int and convert them back
    if max_trials > 0:
        data = remove_extra(  # type: ignore
            data=data, max_trials=max_trials, remove_failed=False, verbose=True
        )
    data.to_csv(output, index=False)
    return data


def get_year(file_path: Path) -> int:
    try:
        i = 2000 + int(file_path.stem[-2:])
        # if i == 2023:
        #     i = 2022
        return i
    except:
        return -1


def keepn(x, stat, largest, n: int, keep_all_equal=True):
    x = x.sort_values(stat, ascending=not largest)
    if len(x) < n:
        return x
    if not keep_all_equal:
        if largest:
            return x.head(n)
        return x.head(n)
    value = x.iloc[n - 1][stat]
    if largest:
        return x.loc[x[stat] >= value]
    return x.loc[x[stat] <= value]


def filter_topn(
    data: pd.DataFrame,
    topn: int,
    stat: str,
    groups: str = "Protocol",
    field: str = "Strategy",
    largest: bool = True,
    include: Sequence[str] = ("WAR", "CAB"),
    exclude: dict[str, Sequence[str]] = dict(
        TAU=(
            "MiCRO",
            "Boulware",
            "Atlas3",
            "CAN",
            "WAN",
            "CAR",
            "WAB",
            "CANNegotiator",
            "WANNegotiator",
            "CARNegotiator",
            "WABNegotiator",
        ),
        TAU0=(
            "MiCRO",
            "Boulware",
            "Atlas3",
            "CAN",
            "WAN",
            "CAR",
            "WAB",
            "CANNegotiator",
            "WANNegotiator",
            "CARNegotiator",
            "WABNegotiator",
        ),
    ),
    exclude_impure: Sequence[str] = ("TAU", "TAU0"),
    reverse: bool = False,
    keep_all_equal: bool = True,
) -> pd.DataFrame:
    for mech in exclude_impure:
        data = data.loc[
            (data["Mechanism"] != mech) | (~data["Strategy"].str.contains("-")), :
        ]
    gg = [groups] if isinstance(groups, str) else list(groups)
    ff = [field] if isinstance(field, str) else list(field)
    g = gg + ff
    addedtimelimit = False
    strategies = data.Strategy.unique()
    tlcol = "time_limit" if "time_limit" in data.columns else "Time"
    if "Mechanism" not in gg:
        addedtimelimit = True
        time_limited = (data[tlcol] > 0).astype(str)
        data["Strategy"] = (
            data["Strategy"].astype("string") + ":" + time_limited.astype("string")
        ).astype("category")
    # we cannot group by floats so we need to convert things to int and convert them back
    for col in g:
        if data[col].dtype != float:
            continue
        data[f"_orig{col}"] = data[col]
        data[col] = (URESOLUTION * data[col]).fillna(-1000).astype(int)
    df = data[[groups, field, stat]].groupby(g).mean().reset_index()

    x = df.groupby(gg).apply(
        functools.partial(
            keepn, stat=stat, largest=largest, n=topn, keep_all_equal=keep_all_equal
        )
    )
    x = x.sort_values(ascending=not largest, by=stat)[g]
    if addedtimelimit:
        x["Strategy"] = x.Strategy.astype(str).str.split(":").str[0].astype("category")
        data["Strategy"] = data.Strategy.str.split(":").str[0].astype("category")
    for col in g:
        if f"_orig{col}" in x.columns:
            x[col] = x[f"_orig{col}"]
            x.drop(f"_orig{col}", axis=1, inplace=True)
    assert not set(data.Strategy.unique()).difference(
        strategies
    ), f"Some new strategies appeared\nNew {data.Strategy.unique()}\nOld {strategies}"
    condition = (data[field].isin(include)) | (
        (data[groups].isin(x[groups].unique())) & (data[field].isin(x[field].unique()))
    )
    for k, v in exclude.items():
        condition = condition & ((data[groups] != k) | (~data[field].isin(v)))
    if reverse:
        return data.loc[~condition, :]
    return data.loc[condition, :]


def filter_dominated_out(
    data: pd.DataFrame,
    stats: Sequence[str],
    groups: str = "Protocol",
    field: str = "Strategy",
    largest: bool = True,
    include: Sequence[str] = ("WAR", "CAB"),
    exclude: dict[str, Sequence[str]] = dict(
        TAU=(
            "MiCRO",
            "Boulware",
            "Atlas3",
            "CAN",
            "WAN",
            "CAR",
            "WAB",
            "CANNegotiator",
            "WANNegotiator",
            "CARNegotiator",
            "WABNegotiator",
        ),
        TAU0=(
            "MiCRO",
            "Boulware",
            "Atlas3",
            "CAN",
            "WAN",
            "CAR",
            "WAB",
            "CANNegotiator",
            "WANNegotiator",
            "CARNegotiator",
            "WABNegotiator",
        ),
    ),
    exclude_impure: Sequence[str] = ("TAU", "TAU0"),
    reverse: bool = False,
) -> pd.DataFrame:
    for mech in exclude_impure:
        data = data.loc[
            (data["Mechanism"] != mech) | (~data["Strategy"].str.contains("-")), :
        ]
    if len(stats) == 1:
        return filter_topn(
            data,
            1,
            stats[0],
            groups,
            field,
            largest,
            include,
            exclude,
            exclude_impure=exclude_impure,
            reverse=False,
        )
    selected = set(include)
    for stat in stats:
        current = filter_topn(
            data,
            1,
            stat,
            groups,
            field,
            largest,
            include,
            exclude,
            exclude_impure=exclude_impure,
            reverse=False,
        )[field].unique()
        for s in current:
            selected.add(s)
    condition = data[field].isin(selected)
    if reverse:
        return data.loc[~condition, :]
    return data.loc[condition, :]


def _get_value(i: int, lbl: str, x) -> float:
    if i < 0:
        try:
            utils = eval(str(x))
            return sum(utils[1:]) / (len(utils) - 1)
        except:
            print(f"failed in reading {x} in {lbl}")
            return float("nan")

    try:
        return eval(str(x))[i] if x else np.nan
    except:
        print(f"failed in reading {x} in {lbl}")
        return float("nan")


def read_data(
    files: list[Path] = [],
    failures: bool = False,
    rounds: bool = True,
    timelimit: bool = True,
    agreements_only: bool = False,
    *,
    max_trials: int = MAX_TRIALS,
    min_trials: int = MIN_TRIALS,
    rational_only=False,
    pure_only=False,
    impure_only=False,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    include_first: list[str] | None = None,
    include_second: list[str] | None = None,
    exclude_first: list[str] | None = None,
    exclude_second: list[str] | None = None,
    tau_exclude: list[str] | None = None,
    permutations: bool = True,
    f1: list[float] | None = None,
    year: int = 0,
    verbose: bool = False,
    addyear: bool = True,
    add_maxs: bool = True,
    original_reserved_values: bool = True,
    controlled_reserved_values: bool = True,
    bilateral: bool = True,
    multilateral: bool = True,
    filter_dominated: bool = False,
    designer_domination_colunmns: Sequence[str] = (
        "K_Fairness",
        "O_K_Fairness",
        "RK_Fairness",
        "O_RK_Fairness",
        "N_Fairness",
        "O_N_Fairness",
        "Welfare",
        "Optimality",
        "O_Optimality",
    ),
    agent_domination_columns: Sequence[str] = ("Utility", "Rank", "Advantage"),
    ignore_second_agent: bool = False,
    implicit_failures: bool = True,
    remove_incomplete_domains: bool = False,
    remove_incomplete_params: dict[str, Any] | None = None,
    mechanisms: Sequence[str] | None = None,
    exclude_mechanisms: Sequence[str] | None = None,
    explicit_types: bool = True,
    remove_incompatible_reserves: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _ = add_maxs
    if include is None:
        include = []
    if exclude is None:
        exclude = []
    if include_first is None:
        include_first = []
    if include_second is None:
        include_second = []
    if exclude_first is None:
        exclude_first = []
    if exclude_second is None:
        exclude_second = []
    if include:
        include_first += include
        include_second += include
    if exclude:
        exclude_first += exclude
        exclude_second += exclude

    x = []
    new_files = []
    fileyear = 0
    for file_path in files:
        if file_path.is_dir():
            for new_file in file_path.glob("*.csv"):
                if not new_file.is_file():
                    continue
                fileyear = get_year(new_file)
                if year > 0 and fileyear > 0 and year != fileyear:
                    continue
                print(
                    f"Adding {file_path.name}/{str(new_file.absolute().relative_to(file_path.absolute()))}"
                )
                new_files.append(new_file)
        else:
            new_files.append(file_path)
    files = new_files
    for file_path in files:
        if not file_path.is_file():
            print(f"[yellow]{str(file_path)} does not exist or is not a file [/yellow]")
            continue
        nr = remove_invalid_lines(file_path, override=True, verbose=verbose)
        if nr:
            print(
                f"[red]Removed {nr} lines that were not valid from {str(file_path)}[/red]"
            )
        clean_results(file_path)
        fileyear = get_year(file_path)
        try:
            if explicit_types:
                current = pd.read_csv(file_path, dtype=DTYPES)  # type: ignore
            else:
                current = pd.read_csv(file_path)
        except EmptyDataError:
            continue

        current, nr = clean_frame(
            data=current,
            valid=dict(f0=F0 + [-1], f1=F1 + [-1]),
            override=True,
            path=file_path,
            verbose=verbose,
        )
        if current is None:
            print(f"[red]Cannot clean frame read from {str(file_path)}[/red]")
        if nr:
            print(f"[red]Removed {nr} invalid records from {str(file_path)}[/red]")
        if current is not None:
            if addyear and fileyear > 0 and "year" not in current.columns:
                current["year"] = fileyear
                if verbose:
                    print(f"[green]{str(file_path)} (year {fileyear}) added [/green]")
            x.append(current)
        if verbose:
            print(f"[green]{str(file_path)} read [/green]")
    if not x:
        print(f"[red]No data to use[/red]")
        raise ValueError("No data to use")
    data = pd.concat(x, ignore_index=True)
    print(f"Found {len(data)} total records")
    if addyear:
        assert "year" in data.columns, f"year is not in the read columns {data.columns}"
    nd, nr = len(data), 0
    data = remove_failed_records(data, explicit=False, implicit=False)
    assert len(data) == nd, f"No records should have been removed"
    if remove_incomplete_domains:
        _params = dict(
            ignore_impure=False,
            ignore_strategies=TAUVARIANTS,
        )
        if remove_incomplete_params is None:
            remove_incomplete_params = dict()
        _params.update(remove_incomplete_params)
        if pure_only:
            _params["ignore_impure"] = True
        data, incomplete, nr = remove_incomplete(data, verbose=verbose, **_params)
        if verbose:
            print(f"Domains and missing runs:\n\t{incomplete}")
    if verbose and nr > 0:
        print(
            f"Removed {nr} data points from {nd} because they were from domains for which some strategies never succeeded"
        )
    nd = len(data)
    if mechanisms:
        data = data.loc[data.mechanism_name.isin(mechanisms), :]
        if verbose and (nd - len(data)):
            print(
                f"Removed mechanisms except {mechanisms}: {len(data)} remains of {nd}"
            )
    nd = len(data)
    if exclude_mechanisms:
        data = data.loc[~data.mechanism_name.isin(exclude_mechanisms), :]
        if verbose and (nd - len(data)):
            print(
                f"Removed mechanisms {exclude_mechanisms}: {len(data)} remains of {nd}"
            )
    data["RationalFraction"] = data[["f0", "f1"]].min(axis=1)
    data["Controlled"] = (~data["RationalFraction"].isna()) & (
        data["RationalFraction"] > -0.05
    )
    if max_trials > 0:
        data = remove_extra(  # type: ignore
            data=data,
            max_trials=max_trials,
            min_trials=min_trials,
            remove_failed=True,
            verbose=verbose,
        )
    data = remove_failed_records(data, implicit=implicit_failures, explicit=failures)
    if not original_reserved_values:
        data = data.loc[data.Controlled, :]
    if not controlled_reserved_values:
        data = data.loc[~data.Controlled, :]
    if f1:
        data = data.loc[data.f1.isin(f1), :]
    if not rounds:
        data = data.loc[data.n_rounds.isnan(), :]
    if not timelimit:
        data = data.loc[data.steps.isnan(), :]
    if not permutations:
        data = data.loc[data.perm_indx == 0, :]

    if agreements_only:
        data = data.loc[data.succeeded, :]
    if pure_only:
        data = data.loc[data.first_strategy_name == data.second_strategy_name, :]
    if impure_only:
        data = data.loc[data.first_strategy_name != data.second_strategy_name, :]
    if include_first:
        data = data.loc[data.first_strategy_name.isin(include_first), :]
    if include_second:
        data = data.loc[data.second_strategy_name.isin(include_second), :]
    if exclude_first:
        data = data.loc[~data.first_strategy_name.isin(exclude_first), :]
    if tau_exclude:
        data = data.loc[
            (~data.first_strategy_name.isin(tau_exclude))
            | (data.mechanism_name.str.startswith("AO")),
            :,
        ]
    if exclude_second:
        data = data.loc[~data.second_strategy_name.isin(exclude_second), :]
    data["Agents"] = data["reserved"].apply(lambda x: len(eval(x)))
    data["Bilateral"] = data["Agents"] == 2
    if not bilateral:
        data = data.loc[~data.Bilateral, :]
    if not multilateral:
        data = data.loc[data.Bilateral, :]

    data = data.loc[data.mechanism_name != "AU0", :]
    try:
        data.year = data.year.astype(int)
    except:
        pass

    data: pd.DataFrame
    data.rename(
        columns={
            "mechanism_name": "Mechanism",
            "strategy_name": "Strategy",
            "first_strategy_name": "Strategy1",
            "second_strategy_name": "Strategy2",
            "domain_name": "Domain",
            "n_rounds": "MaxRounds",
            "time": "Time",
            "year": "Year",
            "n_outcomes": "Outcomes",
            "pareto_optimality": "Optimality",
            "ordinal_pareto_optimality": "O_Optimality",
            "nash_optimality": "N_Fairness",
            "ordinal_nash_optimality": "O_N_Fairness",
            "kalai_optimality": "K_Fairness",
            "ordinal_modified_kalai_optimality": "O_RK_Fairness",
            "modified_kalai_optimality": "RK_Fairness",
            "ordinal_kalai_optimality": "O_K_Fairness",
            "max_welfare_optimality": "Welfare",
            "ordinal_max_welfare_optimality": "O_Welfare",
            "f0": "Rational1",
            "f1": "Rational2",
        },
        inplace=True,
    )
    data["IRUB1"] = data["noffers0"] / data["Outcomes"]
    data["IRUB2"] = data["noffers1"] / data["Outcomes"]
    data["IRUB"] = 0.5 * (data["IRUB1"] + data["IRUB2"])
    data["Privacy1"] = 1 - data.IRUB1
    data["Privacy2"] = 1 - data.IRUB2
    data["Privacy"] = 1 - data.IRUB
    data["Uniqueness1"] = data["noffers0"] / data.steps
    data["Uniqueness2"] = data["noffers1"] / data.steps
    data["Uniqueness"] = 0.5 * (data["Uniqueness1"] + data["Uniqueness2"])
    data["Fairness"] = data[[_ for _ in data.columns if "Fairness" in _]].max(axis=1)
    data.loc[data.Year == 2023, "Year"] = 2022
    data["RelativeRounds"] = data["MaxRounds"] / data["Outcomes"]
    data["RelativeRounds"] = data["RelativeRounds"].fillna(0)
    data["RelativeRounds"] = data["RelativeRounds"].astype(int)
    data["Rounds"] = data.steps / data["MaxRounds"]
    # data.loc[(data["Mechanism"] == "AOr"), "Mechanism"] = data.loc[
    #     (data["Mechanism"] == "AOr"), "Mechanism"
    # ] + data.loc[(data["Mechanism"] == "AOr"), "RelativeRounds"].astype(str)
    data["Mechanism"] = (
        data.Mechanism.str.replace("AOr1", "AOP($n_o$)")  # type: ignore
        .str.replace("AOr", "AOP($n_o$)")  # type: ignore
        .str.replace("AOt", "AOP(3min)")
        .str.replace("TAU0", "TAU")
        .str.replace("AUinf", "TAUinf")
        .str.replace("TAUinf", "TAUinf")
    )
    typemap = dict(
        # Baseline=["Boulware", "Linear", "Conceder", "NiceTfT"],
        WAR=["WAR"],
        CAB=["CAB"],
        # MiCRO=["MiCRO"],
    )
    for ext in ("", "1", "2"):
        data["Strategy" + ext] = (
            data["Strategy" + ext]
            .str.replace("Negotiator", "")
            .str.replace("The", "TheNegotiator")
            .str.replace("CUHKAgent", "CUHK")
            .str.replace("Aspiration", "Boulware")
            .str.replace("NiceTitForTat", "NiceTfT")
            .str.replace("ConcederTB", "Conceder")
            .str.replace("LinearTB", "Linear")
        )
    for ext in ("1", "2"):
        col = "Type" + ext
        data[col] = data["Strategy" + ext]
        allmapped = list(itertools.chain(*typemap.values()))
        # print(allmapped
        data.loc[~data[col].isin(allmapped), col] = "SOTA"
        for k, v in typemap.items():
            data.loc[data[col].isin(v), col] = k
    data["Type"] = data["Type1"] + "-" + data["Type2"]
    data["Condition"] = data.Mechanism + "+" + data.Strategy
    data["Completeness"] = data.succeeded.astype(int)  # type: ignore
    data.loc[(data["RationalFraction"] < -0.1), "RationalFraction"] = float("nan")
    if filter_dominated and len(data):
        data = filter_dominated_out(
            data,
            stats=designer_domination_colunmns,
            groups="Mechanism",
            largest=True,
            exclude=dict(),
            reverse=False,
        )
    data.loc[(data["MaxRounds"].isna()), "Rounds"] = data.steps / data["Outcomes"]
    # data.drop(columns=["Mechanism", "Strategy"], inplace=True)
    # data.drop("Strategy1", axis=1)
    # data.drop("Strategy2", axis=1)
    data["HasRational"] = True

    data["rank0"] = data["agreement_ranks"].apply(
        functools.partial(_get_value, 0, "rank0")
    )
    data["rank1"] = data["agreement_ranks"].apply(
        functools.partial(_get_value, 1, "rank1")
    )
    if "resrank" in data.columns and ("resrank0" not in data.columns):
        data["resrank0"] = data.resrank.apply(
            functools.partial(_get_value, 0, "resrank0")
        )
    if "resrank" in data.columns and ("resrank1" not in data.columns):
        data["resrank1"] = data.resrank.apply(
            functools.partial(_get_value, 1, "resrank1")
        )
    data["util0"] = data.agreement_utils.apply(
        functools.partial(_get_value, 0, "util0")
    )
    data["util1"] = data.agreement_utils.apply(
        functools.partial(_get_value, 1, "util1")
    )
    data["reserved0"] = data.reserved.apply(
        functools.partial(_get_value, 0, "reserved0")
    )
    data["reserved1"] = data.reserved.apply(
        functools.partial(_get_value, 1, "reserved1")
    )
    data["Rank1"] = data.rank0
    data["Rank2"] = data.rank1
    data["max1"] = 1.0
    data["max0"] = 1.0
    data["O_Advantage1"] = (data.rank0 - data.resrank0) / (1 - data.resrank0)
    data["O_Advantage2"] = (data.rank1 - data.resrank1) / (1 - data.resrank1)
    data["Utility1"] = data.util0 / data.max0
    data["Utility2"] = data.util1 / data.max1
    data["Advantage1"] = (data.util0 - data.reserved0) / (data.max0 - data.reserved0)
    data["Advantage2"] = (data.util1 - data.reserved1) / (data.max1 - data.reserved1)
    data["Advantage"] = (data.Advantage1 + data.Advantage2) / 2
    # to do use average of all others for2 items and add bilateral bool column
    if rational_only:
        data = data.loc[data["HasRational"], :]
    data["AgentScore"] = (data["Advantage"] + data["Privacy"]) / (2)
    data["RationalFraction"] = data["RationalFraction"].round(1)

    data = refresh_values(data)

    data["Difficulty"] = (
        data["RationalFraction"]
        .transform(
            lambda x: "Original"
            if np.isnan(x) or x < 0
            else (
                "Hard"
                if 0 < x < 0.25
                else ("Medium" if x < 0.75 else ("Simple" if x < 0.95 else "Trivial"))
            )
        )
        .astype("category")
    )
    data["Size"] = pd.cut(
        data.Outcomes,
        bins=[1, 10, 100, 1000, 10000, 100000, 1000_000],
        labels=["~1", "~10", "~100", "~1,000", "~10,000", "~100,000"],
    )
    data["SizeOrder"] = pd.cut(
        data.Outcomes,
        bins=[1, 10, 100, 1000, 10000, 100000, 1000_000],
        labels=["1", "10", "100", "1000", "10000", "100000"],
    ).astype(  # type: ignore
        int
    )

    print(f"Will use {len(data)} data points")
    main_cols = [
        # "Strategy1",
        # "Strategy2",
        "Mechanism",
        # "Type",
        "RationalFraction",
        "Outcomes",
        "Completeness",
        "Bilateral",
        "Domain",
        "Year",
        "Controlled",
        "HasRational",
        "Time",
        "Rounds",
        "MaxRounds",
        "time_limit",
        "Difficulty",
        "Size",
        "SizeOrder",
        "Protocol",
        "succeeded",
    ]
    data["Score"] = data["Optimality"] * data["Completeness"]
    data["OutcomeScore"] = data["Score"] * data["Fairness"]
    data["DesignerScore"] = data["OutcomeScore"] * data["Welfare"]
    data["ScoreWithIR"] = data["DesignerScore"] * data.Privacy
    data["FinalScore"] = data["ScoreWithIR"] * data["Uniqueness"]
    data["Quality"] = data["AgentScore"] * data["DesignerScore"]
    data["Protocol"] = data["Mechanism"]
    for x in ("AOP", "TAU"):
        data.loc[data["Protocol"].str.startswith(x), "Protocol"] = x
    data = data.drop(columns=FINALREMOVE)
    if remove_incompatible_reserves:
        data = data.loc[
            (~data.Strategy2.isin(ONLY_ZERO_RESERVE))
            | (data.RationalFraction > 0.95)
            | (data.RationalFraction < -0.5)
            | (~data.Controlled),
            :,
        ]
    data = refresh_values(data)

    first = data[
        main_cols
        + [_ for _ in data.columns if "1" in _ and _ not in main_cols]
        + [_ for _ in data.columns if "2" in _ and _ not in main_cols]
    ].copy()
    first.reset_index()
    first.columns = [_.replace("1", "") for _ in first.columns]
    first.columns = [_.replace("2", "Partner") for _ in first.columns]
    first.rename(
        columns={
            "Strategy1": "Strategy",
            "Strategy2": "Partner",
            "Rational1": "$f_{self}$",
            "Rational2": "$f_{partner}$",
        },
        inplace=True,
    )
    # first["Condition"] = first.Mechanism + "+" + first.Strategy
    first["Starting"] = True
    first = first.reset_index(drop=True)
    if ignore_second_agent:
        data_agent = first
    else:
        second = data[
            main_cols
            + [_ for _ in data.columns if "2" in _ and _ not in main_cols]
            + [_ for _ in data.columns if "1" in _ and _ not in main_cols]
        ].copy()
        second.reset_index()
        second.columns = [_.replace("2", "") for _ in second.columns]
        second.columns = [_.replace("1", "Partner") for _ in second.columns]
        second.rename(
            columns={
                "Strategy1": "Partner",
                "Strategy2": "Strategy",
                "Rational1": "$f_{partner}$",
                "Rational2": "$f_{self}$",
            },
            inplace=True,
        )

        # second["Condition"] = second.Mechanism + "+" + second.Strategy
        second["Starting"] = False
        second = second.reset_index(drop=True)
        data_agent = pd.concat((first, second), ignore_index=True)
    data_agent["Condition"] = data_agent.Mechanism + "+" + data_agent.Strategy
    data_agent["AgentScore"] = (data_agent["Advantage"] + data_agent["Privacy"]) / (2)
    if filter_dominated and len(data_agent):
        data_agent = filter_dominated_out(
            data_agent,
            stats=agent_domination_columns,
            groups="Mechanism",
            largest=True,
            exclude=dict(),
            reverse=False,
        )

    data = data.astype(DTYPESFINAL)
    data_agent = data_agent.astype(DTYPESAGENTS)
    data = refresh_values(data)
    data_agent = refresh_values(data_agent)
    return data, data_agent


# def agent_view(data: pd.DataFrame) -> pd.DataFrame:
#     x1 = data.loc[:]


def make_latex_table(
    data: pd.DataFrame,
    file_name: str | Path,
    count: bool,
    perdomain: bool = False,
    precision: int = PRECISION,
    perf1: bool = True,
    what: Literal["ordinal", "cardinal", "timing", "all", "final", "agent"] = "all",
    condition_col="Condition",
) -> pd.DataFrame:
    if what == "ordinal":
        stats, lbl = STATS_ORDINAL, "ordinal"
    elif what == "final":
        stats, lbl = STATS_FINAL, ""
    elif what == "cardinal":
        stats, lbl = STATS_CARDINAL, "cardinal"
    elif what == "timing":
        stats, lbl = STATS_TIMING, "timing"
    elif what == "all":
        stats, lbl = STATS, "everything"
    elif what == "agent":
        stats, lbl = STATS_AGENT, "agent"
    else:
        raise ValueError(f"Unknown table type {what}")
    file_name = str(file_name)
    if file_name.endswith(".tex"):
        file_name = file_name[:-4]
    file_name += f"{f'-{lbl}' if lbl else ''}.tex"
    data = data.sort_values(condition_col)  # type: ignore
    groups = [condition_col]
    if perdomain:
        groups.append("Domain")
    if perf1:
        groups = ["RationalFraction"] + groups
        data["RationalFraction"] = (
            (data["RationalFraction"].fillna(1.0) * 1000).round().astype(int)
        )

    calculations = {k: ["mean", "std"] for k in stats}
    if "Completeness" in calculations.keys():
        calculations["Completeness"] = ["mean"]
    if count:
        calculations[stats[0]] = ["size"] + calculations[stats[0]]
    results: pd.DataFrame
    results = data.groupby(groups).agg(calculations)  # type: ignore
    # results = results.reset_index()
    results = results.fillna(1)
    # results = results.set_index(groups)
    if len(results.columns) > len(results):
        results = results.transpose()
    Path(file_name).parent.mkdir(exist_ok=True, parents=True)
    results = results.reset_index()
    results.to_csv(file_name.replace("tex", "csv"), index=False)
    results.to_latex(
        file_name,
        index=True,
        escape=False,
        float_format=f"{{:0.{precision}f}}".format if precision > 0 else None,
    )
    adjust_tex(Path(file_name))
    return results


def do_all_tests(
    data: pd.DataFrame,
    insignificant: bool,
    allstats: bool,
    path: Path,
    basename: str = "exp1",
    significant: bool = True,
    exceptions: bool = True,
    stats: list[str] = STATS,
    precision: int = PRECISION,
):
    for type in ("ttest", "wilcoxon"):
        results = dict()
        file_name = str(
            path
            / f"tbl-{basename}-{type}{'-all' if allstats else ''}{'-insig' if insignificant else ''}"
        )
        for stat in stats:
            results[stat] = factorial_test(
                data,
                stat,
                type,
                insignificant,
                allstats,
                significant=significant,
                exceptions=exceptions,
                tbl_name=file_name,
                precision=precision,
            )
        file_name = str(
            path
            / f"ttest-{basename}-{type}{'-all' if allstats else ''}{'-insig' if insignificant else ''}.json"
        )
        dump(results, file_name)


def factorial_test(
    data: pd.DataFrame,
    stat: str,
    type: str,
    insignificant: bool = True,
    significant: bool = True,
    allstats: bool = False,
    exceptions: bool = True,
    tbl_name: str | None = None,
    ignore_nan_ps: bool = False,
    precision: int = PRECISION,
    alternative: str = "two-sided",
    print_na: bool = True,
    stop_on_nas: bool = False,
    fill_nas: bool = False,
    drop_nas: bool = False,
    condition_col: str = "Condition",
    tbl_index: bool = True,
    pivot_index: list[str] | tuple[str] | str = "Domain",
    aggfunc: str = "",
):
    """
    Args:
        data: data read through read_data()
        stat: The stat column (e.g. Optimality)
        type: Test type. One of "ttest", "ttest_ind", "wilcoxon", "median"
        insignificant: Save statistically insignificant results
        allstats: If given, all strategy combination comparisons will be done. If not given, one side must be a TAU strategy
        exceptions: raise exceptions
        tbl_name: if given, will save the statistics (not the statistical test results) in the given file
        ignore_nan_ps: Ignore nans in p-values
        precision: The precision used when saving statistics
        alternative: two-sided or one-sided test
        stop_on_nas: Stop processing and return an empty result if NAs were found in the data
        print_na: Print conditions with NAs
        fill_nas: If given NaNs will be filled with zeros
        drop_nas: If given NaNs will be dropped from the data
        tbl_index: If given, will save the index of the latex table (tbl_name must not be None)
        pivot_index: the index to pivot with when creating the latex table
        agg_func: the aggregation function when creating the pivot_table. If empty, a suitable value for the test will be created
    """
    method = dict(
        ttest=ttest_rel, ttest_ind=ttest_ind, wilcoxon=wilcoxon, median=median_test
    )[type]
    params = dict(
        ttest=dict(alternative=alternative),
        ttest_ind=dict(alternative=alternative),
        wilcoxon=dict(alternative=alternative),
        median=dict(ties="below"),
    )[type]
    aggfunc = dict(ttest="mean", ttest_ind="mean", wilcoxon="mean", median="median")[
        type
    ]
    results = dict()
    if isinstance(pivot_index, str):
        pivot_index = [pivot_index]
    else:
        pivot_index = list(pivot_index)
    if len(pivot_index) == 1:
        CONDNAME = pivot_index[0]
    else:
        CONDNAME = "Cond"
        data[CONDNAME] = data[pivot_index[0]].astype(str)
        for v in pivot_index[1:]:
            data[CONDNAME] += "_" + data[v].astype(str)
        data[CONDNAME] = data[CONDNAME].astype("category")
    non_stats = list(
        set([_ if _ != "Condition" else condition_col for _ in NONSTATS])
    ) + [CONDNAME]
    data = data.loc[:, non_stats + [stat]]
    tbl = pd.pivot_table(
        data, index=CONDNAME, columns=condition_col, values=stat, aggfunc=aggfunc
    )
    if print_na:
        nantbl = tbl.isna().sum()
        nantbl = nantbl[nantbl > 0]
        if len(nantbl) > 0:
            print(nantbl)
            if stop_on_nas:
                return []
    if drop_nas:
        tbl = tbl.dropna()
    default = dict(zip(STATS, itertools.repeat(0)))
    default.update(dict(zip(STATS_AGENT, itertools.repeat(0))))
    if fill_nas:
        tbl = tbl.fillna(default.get(stat, 0))
    if tbl is None:
        return []
    if tbl_name is not None:
        tbl_name = str(tbl_name)
        if tbl_name.endswith("csv"):
            tbl_name = tbl_name[:-4]
            tbl_fname = Path(tbl_name).name
            tbl_path = Path(tbl_name).absolute()
            tbl_path = tbl_path.parent / "_".join(pivot_index)
            tbl_fname = f"{tbl_fname}/{aggfunc}-{stat.replace(' ','').replace('.', '')}"
            tbl_fname += ".csv"
            final_name = tbl_path / tbl_fname
            Path(final_name).parent.mkdir(exist_ok=True, parents=True)
            tbl.to_csv(final_name)
        else:
            if tbl_name.endswith(".tex"):
                tbl_name = tbl_name[:-4]
            tbl_fname = Path(tbl_name).name
            tbl_path = Path(tbl_name).absolute()
            tbl_path = tbl_path.parent / "_".join(pivot_index)
            tbl_fname = f"{tbl_fname}/{aggfunc}-{stat.replace(' ','').replace('.', '')}"
            tbl_fname += ".tex"
            final_name = tbl_path / tbl_fname
            Path(final_name).parent.mkdir(exist_ok=True, parents=True)
            styler = tbl.style.highlight_max(
                axis=0,
                props="bfseries: ;",
                # float_format=f"{{:0.{precision}f}}".format if precision > 0 else None,
            )
            if precision > 0:
                styler.format(f"{{:0.{precision}f}}")

            # index=tbl_index,
            styler.to_latex(
                final_name,
                hrules=True,
                # escape=False,
            )
    correction = len(tbl.columns) - 1
    combinations = (
        (a, b)
        for a, b in product(tbl.columns, tbl.columns)
        if a < b and (allstats or "TAU" in a or "TAU" in b)
    )
    combinations = [(b, a) if b > a else (a, b) for a, b in combinations]
    for a, b in combinations:
        if a == b:
            continue
        # key = f"{a} vs. {b}" if allstats else b
        key = f"{a} vs. {b}"
        try:
            r = method(tbl[a], tbl[b], nan_policy="omit", **params)  # type: ignore
            t, p = r.statistic, r.pvalue  # type: ignore
            if ignore_nan_ps and np.isnan(p):
                p = float("inf")
            if not insignificant and p >= 0.05 / correction:
                continue
            if not significant and p < 0.05 / correction:
                continue
            results[key] = (t, p)
        except Exception as e:
            if exceptions:
                results[key] = str(e)
    return results


def correct_field_types(
    path: Path, override: bool = True, verbose: bool = False
) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    data = None  # type: ignore
    try:
        data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
        if verbose:
            print(f"[green]All datatypes correct ({path})[/green]")
        return 0
    except EmptyDataError:
        return 0
    except Exception as e:
        data: pd.DataFrame = pd.read_csv(path)  # type: ignore
        n = len(data)
        for k, v, d in (("failed", "bool", True), ("succeeded", "bool", False)):
            if str(data.dtypes[k]) == v:
                continue
            data.loc[data[k].isnull(), k] = f"{d}"
            data[k] = data[k].astype(v)
            assert str(data.dtypes[k]) == v, f"{data.dtypes[k]}"
        for k, v in DTYPES.items():
            try:
                if v == "string":
                    data.loc[data[k].isna(), k] = ""
                    data.loc[data[k].isnull(), k] = ""
                    data = data.astype("string")
                    continue
            except Exception as e:
                print(
                    f"[red]Cannot correct datatype for field {k} (should be {v}) {path}[/red]: {e}"
                )
                pass
    if override and data is not None and len(data) == n:
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
    try:
        data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
        print(f"[yellow]Can read without datatypes for {path}[/yellow]")
    except Exception as e:
        print(f"[red]Failed in correcting datatypes for {path}[/red]: {e}")
        return 0
    return 0


def remove_repeated_header(
    path: Path, override: bool = True, verbose: bool = False
) -> int:
    if not path.exists():
        return 0
    with open(path, "r") as f:
        lines = f.readlines()
    n_lines = len(lines)
    if verbose:
        print(f"Found {n_lines} lines", flush=True, end="")
    if not lines:
        return 0
    first = lines[0]
    corrected = [first]
    changed = False
    for l in lines[1:]:
        if l[:20] == first[:20]:
            changed = True
            continue
        corrected.append(l)
    if not changed:
        if verbose:
            print(f" [green]all clean[/green]", flush=True)
        return 0
    nr = n_lines - len(corrected)
    if override:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.writelines(corrected)
    if verbose:
        print(
            f" ... removed [red]{nr}[/red] repeated header lines keeping {len(corrected) - 1} records.",
            flush=True,
        )
    return nr


def clean_results(
    path: Path,
    *,
    override: bool = True,
    verbose: bool = False,
    f0s: list[float] = F0,
    f1s: list[float] = F1,
):
    try:
        remove_repeated_header(path, override=override, verbose=verbose)
        remove_invalid_lines(path, override=override, verbose=verbose)
        correct_field_types(path, verbose=verbose, override=override)
        clean_frame(
            path=path,
            data=None,
            verbose=verbose,
            override=override,
            valid=dict(f0=list(set(list(f0s) + [-1])), f1=list(set(list(f1s) + [-1]))),
        )
    except Exception as e:
        print(f"[red]Cannot clean {path}[/red]: {e}")


def clean_frame(
    path: Path | None = None,
    data: pd.DataFrame | None = None,  # type: ignore
    verbose: bool = False,
    override: bool = False,
    valid: dict[str, Sequence] = dict(),
    notnan: Sequence[str] = ("f0", "f1", "agreement_ranks", "agreement_utils"),
    exclude_domains: Sequence[str] = tuple(),
    exclude_mechanisms: Sequence[str] = tuple(),
    exclude_strategies: Sequence[str] = tuple(),
) -> tuple[pd.DataFrame | None, int]:
    nt = 0
    if data is None and path is not None:
        try:
            data = pd.read_csv(path, dtype=DTYPES)  # type: ignore
        except EmptyDataError:
            pass
        except FileNotFoundError:
            pass
        except:
            print(
                f"[yellow]Failed reading {path} for clean-frame ... trying to correct[/yellow]"
            )
            remove_repeated_header(path, override=True, verbose=verbose)
            remove_invalid_lines(path, override=True, verbose=verbose)
            correct_field_types(path, override=True, verbose=verbose)
            try:
                data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
                print(f"[green]Corrected {path} in clean-frame.[/green]")
            except EmptyDataError:
                pass
            except Exception as e:
                print(e)
                try:
                    data: pd.DataFrame = pd.read_csv(path, dtype=DTYPES)  # type: ignore
                    print(
                        f"[yellow]Read without types for {path} in clean-frame.[/yellow]"
                    )
                except Exception as e:
                    print(
                        f"[red]Failed in correcting {path} for clean-frame[/red]: {e}"
                    )

    if data is None:
        return data, 0
    data: pd.DataFrame  # type: ignore
    if len(data) == 0:
        return data, 0

    mm = len(data)
    for domain in exclude_domains:
        data = data.loc[~data.domain_name.str.contains(domain), :]
        if verbose and mm > len(data):
            print(
                f"\tremoved [red] {mm - len(data)} [/red] records with domain {domain}"
            )
    for mechanism in exclude_mechanisms:
        data = data.loc[~data.mechanism_name.str.contains(mechanism), :]
        if verbose and mm > len(data):
            print(
                f"\tremoved [red] {mm - len(data)} [/red] records with mechanism {mechanism}"
            )

    for strategy in exclude_strategies:
        data = data.loc[~data.strategy_name.str.contains(strategy), :]
        if verbose and mm > len(data):
            print(
                f"\tremoved [red] {mm - len(data)} [/red] records with strategy {strategy}"
            )
    for col in notnan:
        n = len(data.loc[data[col].isna(), col])
        if n > 0:
            nt += n
            if verbose:
                print(f"[red] removing {n} records with nan in {col} [/red]")
            nbefore = len(data)
            data = data.loc[~data[col].isna(), :]
            assert len(data) == nbefore - n
    for col, v in valid.items():
        if not v:
            continue
        if isinstance(v[0], str):
            cond = ~data[col].isin(v)
        else:
            cond = np.abs(data[col] - v[0]) > 1e-3
            for _ in v[1:]:
                cond &= np.abs(data[col] - _) > 1e-3
        x = data.loc[cond, col]
        n = len(x)
        if n > 0:
            nt += n
            nbefore = len(data)
            if verbose:
                print(
                    f"[red] removing {n} records {col} values not in {v}: {x.unique()} [/red]"
                )
            data = data.loc[~cond, :]
            assert len(data) == nbefore - n
    if verbose and nt:
        print(
            f"[red] Removed {nt} records in total from {str(path)} keeping {len(data)} records ({nt / (nt + len(data))}) [/red]"
        )
    if override:
        data.to_csv(path, index=False)
    return data, nt


def count_runs(file_name: Path) -> int:
    if not file_name.exists():
        return 0
    with open(file_name, "r") as f:
        lines = f.readlines()
    header = lines[0].split(",")
    try:
        failed_field = header.index("failed")

        def is_failed(i: int, line: str) -> bool:
            if len(line.split(",")) < failed_field + 1:
                print(
                    f"[yellow]Warning: Line {i} of {str(file_name)} has {len(line.split(','))} fields. Expected more than{failed_field}[/yellow]"
                )
                return True
            return line.split(",")[failed_field] == "True"

    except ValueError:

        def is_failed(i: int, line: str) -> bool:
            _ = i, line
            return False

    return len(
        [
            _
            for i, _ in enumerate(lines)
            if not _.startswith("mechanism") and not is_failed(i, _)
        ]
    )


# def add_pareto(
#     dinfo: dict[str, Any],
#     d: Scenario,
#     perm_index: int = 0,
#     f0: float | None = None,
#     f1: float | None = None,
# ) -> dict[str, Any]:
#     ufuns = get_ufuns_for(d.ufuns, perm_index)
#     if f0 is not None and f0 >= 0:
#         ufuns[0].reserved_value = calc_reserved_value(ufuns[0], f0, nmin=1, finite=True)
#     # it is enough to set the reserved value of the second ufun only
#     if f1 is not None and f1 >= 0:
#         ufuns[1].reserved_value = calc_reserved_value(ufuns[1], f1, nmin=1, finite=True)
#     if f0 is not None or f1 is not None:
#         if len(ufuns) > 2:
#             for u in ufuns:
#                 u.reserved_value = 0.0
#     rvalues = tuple(u.reserved_value for u in ufuns)
#     stats_dict = {k: v for k, v in dinfo.items() if k not in ("stats", "ordinal_stats")}
#     stats_dict.update(
#         dict(
#             perm_index=perm_index,
#             f0=f0,
#             f1=f1,
#             r0=ufuns[0].reserved_value,
#             r1=ufuns[1].reserved_value,
#             reserved_values=rvalues,
#         )
#     )
#     stats = ScenarioStats(**dinfo["stats"]).restrict(
#         ufuns=ufuns, reserved_values=rvalues
#     )
#     stats_rank = ScenarioStats(**dinfo["ordinal_stats"]).restrict(
#         ufuns=ufuns, reserved_values=rvalues
#     )
#     stats_dict.update(asdict(stats))
#     stats_dict.update(
#         {
#             k.replace("utils", "ranks")
#             .replace("outcomes", "ordinal_outcomes")
#             .replace("opposition", "ordinal_opposition"): v
#             for k, v in asdict(stats_rank).items()
#         }
#     )
#     dinfo.update(stats_dict)
#     return dinfo


def no_permutations(lst: list | tuple | Iterable) -> list[tuple]:
    return [tuple(lst)]


def rotations(lst: list | tuple | Iterable) -> list[tuple]:
    lst = list(lst)
    if not lst:
        return []
    d = deque(lst)
    r = []
    for _ in range(len(lst)):
        r.append(tuple(d))
        d.rotate()
    return r


def isserver():
    name = gethostname()
    return "daiba" in name


def safe_key_(x) -> str:
    if x is None:
        return "none"
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        if np.isinf(x):
            return "inf"
        if np.isneginf(x):
            return "-inf"
        if np.isnan(x):
            return "nan"
        return str(int(x * URESOLUTION + 0.5))
    return str(x)


def key_(perm, f0, f1):
    if f0 < 0 and f1 < 1:
        return BASEKEY
    return f"({int(perm)},{f0:0.{URESOLUTION}f},{f1:0.{URESOLUTION}f})"


def unpack_(key: str) -> tuple[int, float, float]:
    return eval(key)


def permute_state(state: ScenarioStats, perm: tuple[int]) -> ScenarioStats:
    state_dict = asdict(state)
    result = deepcopy(state)
    result.pareto_utils = tuple(tuple(a[_] for _ in perm) for a in state.pareto_utils)
    result.nash_utils = [tuple(a[_] for _ in perm) for a in state.nash_utils]
    result.kalai_utils = [tuple(a[_] for _ in perm) for a in state.kalai_utils]
    result.max_welfare_utils = [
        tuple(a[_] for _ in perm) for a in state.max_welfare_utils
    ]
    result.kalai_utils = [tuple(a[_] for _ in perm) for a in state.kalai_utils]
    result.utility_ranges = [tuple(a[_] for _ in perm) for a in state.utility_ranges]
    assert str(state_dict) == str(asdict(state))
    return result


def set_reservation(
    ufuns: Sequence[UtilityFunction], f0: float, f1: float
) -> tuple[UtilityFunction, ...]:
    if f0 >= 0:
        ufuns[0].reserved_value = calc_reserved_value(ufuns[0], f0, nmin=1, finite=True)
    if f1 >= 0:
        ufuns[1].reserved_value = calc_reserved_value(ufuns[1], f1, nmin=1, finite=True)
    if len(ufuns) > 2 and (f0 >= 0 or f1 >= 0):
        for u in ufuns[2:]:
            mn, _ = u.minmax()
            u.reserved_value = mn - EPS
    return tuple(ufuns)


def calc_rational_fraction(
    ufuns: Sequence[UtilityFunction], outcomes: Sequence[Outcome]
):
    if not outcomes:
        return 1.0
    nr = 0
    for o in outcomes:
        if is_rational(ufuns, o):
            nr += 1
    return nr / len(outcomes)


@contextmanager
def print_timing(txt: str, level: int = 0, single_line: bool = False, verbose=True):
    if not verbose:
        yield
        return

    if level:
        print("\t" * level, end="")
    print(
        f"{txt}: {datetime.now()}",
        flush=True,
        end="" if single_line else "\n",
    )
    _start = perf_counter()
    yield
    if not verbose:
        return
    if single_line:
        print(
            f"DONE in {humanize_time(perf_counter() - _start, show_ms=True, show_us=True)}",
            flush=True,
        )
    else:
        print(
            f"{txt} DONE in {humanize_time(perf_counter() - _start, show_ms=True, show_us=True)}",
            flush=True,
        )


def get_stats(
    os: CartesianOutcomeSpace,
    ufuns: list[UtilityFunction] | tuple[UtilityFunction, ...],
    dir: Path,
    base: Path,
    verbose: bool,
    f0s: list[float],
    f1s: list[float],
    override: bool = True,
    check: bool = False,
):
    base = Path(base).absolute()
    dir = Path(dir).absolute()
    if verbose:
        print(
            f"[yellow]Adding Stats to {str(dir.relative_to(base))}[/yellow]", flush=True
        )
    path = str(dir.absolute())
    is_anac = True
    year = None
    if is_anac:
        if "scenarios/" in path:
            year = int(path.split("scenarios/")[-1].split("/")[0][1:])
        else:
            year = 2013
    n_outcomes = os.cardinality

    result = dict(
        name=dir.name,
        path=str(dir),
        continuous=n_outcomes == float("inf"),
        n_outcomes=n_outcomes,
        n_ufuns=len(ufuns),
        is_anac=is_anac,
        year=year,
        # reserved_values=reserved_values,
    )

    def _extend_stats(
        stats: ScenarioStats,
        perm: tuple[int],
        ufuns: Sequence[UtilityFunction],
        mins: Sequence[float],
        maxs: Sequence[float],
        f0: float,
        f1: float,
        outcomes: Sequence[Outcome],
    ):
        # reorder ufuns, mins and maxs
        mins = [mins[_] for _ in perm]
        maxs = [maxs[_] for _ in perm]
        ufuns = [ufuns[_] for _ in perm]
        # save reserved values
        reserved = tuple(_.reserved_value for _ in ufuns)
        # we set reserved values according to the new order
        set_reservation(ufuns, f0, f1)
        fr = calc_rational_fraction(ufuns, outcomes)
        # permute the state accordingly
        with print_timing("Calculating scenario stats", level=2, verbose=verbose):
            stats = calc_scenario_stats(ufuns, outcomes=outcomes)
        with print_timing("Calculating distances", level=2, verbose=verbose):
            dists = calc_outcome_distances(
                tuple(_.reserved_value for _ in ufuns), stats
            )
        with print_timing("Calculating outcome optimality", level=2, verbose=verbose):
            optim = calc_outcome_optimality(dists, stats, estimate_max_dist(ufuns))
        # restore reserved values
        for u, r in zip(ufuns, reserved):
            u.reserved_value = r
        return dict(
            mins=mins,
            maxs=maxs,
            stats=asdict(stats),
            fr=fr,
            ir=fr > 1e-6,
            dists=asdict(dists),
            optim=asdict(optim),
        )

    def _insert_stats(f0, f1, result, ufuns, override):
        # find where to insert the stats
        key = key_(0, f0, f1)
        assert ufuns[0].outcome_space is not None
        # save reserved values
        reserved = tuple(_.reserved_value for _ in ufuns)
        outcomes = list(ufuns[0].outcome_space.enumerate_or_sample())
        if not override and key in result:
            print(f"\t[blue]Reusing {key}[/blue] ", end="")
            ordinal_ufuns = [make_rank_ufun(u) for u in ufuns]
            stats, ordinal_stats = result[key]["cardinal"], result[key]["ordinal"]
            mins, maxs = result[key]["mins"], result[key]["maxs"]
            ordinal_mins, ordinal_maxs = (
                result[key]["ordinal_mins"],
                result[key]["ordinal_maxs"],
            )
        else:
            # change reserved values as needed
            with print_timing(
                "Calculating rational fractions", level=2, verbose=verbose
            ):
                ufuns = set_reservation(ufuns, f0, f1)
                fraction_rational = calc_rational_fraction(ufuns, outcomes)
            # calculate stats for the ufuns
            with print_timing(
                f"Calculating stats ({f0=}, {f1=})", level=2, verbose=verbose
            ):
                stats = calc_scenario_stats(ufuns, outcomes=outcomes)
            with print_timing(
                f"Calculating distances ({f0=}, {f1=})", level=2, verbose=verbose
            ):
                dists = calc_outcome_distances(
                    tuple(_.reserved_value for _ in ufuns), stats
                )
            with print_timing(
                f"Calculating optimality ({f0=}, {f1=})", level=2, verbose=verbose
            ):
                optim = calc_outcome_optimality(dists, stats, estimate_max_dist(ufuns))
            maxs = [_[1] for _ in stats.utility_ranges]
            mins = [_[0] for _ in stats.utility_ranges]
            result[key] = dict(
                perm=tuple(range(len(ufuns))),
                perm_index=0,
                maxs=maxs,
                mins=mins,
                cardinal=asdict(stats),
                cardinal_reserved_dists=asdict(dists),
                cardinal_reserved_optim=asdict(optim),
                reserved_values=tuple(_.reserved_value for _ in ufuns),
                f0=f0,
                f1=f1,
                fraction_rational=fraction_rational,
                is_rational=fraction_rational > 1e-6,
            )
            # convert to ranks ufuns (remove utility information keeping only relative order)
            with print_timing("Making rank ufuns", level=2, verbose=verbose):
                ordinal_ufuns = [make_rank_ufun(u) for u in ufuns]
            with print_timing(
                f"Calculating rational fraction (ordinal, {f0=}, {f1=})",
                level=2,
                verbose=verbose,
            ):
                ordinal_ufuns = set_reservation(ordinal_ufuns, f0, f1)
                ordinal_fraction_rational = calc_rational_fraction(
                    ordinal_ufuns, outcomes
                )
            if check:
                assert (
                    fraction_rational == ordinal_fraction_rational
                ), f"{fraction_rational=}, {ordinal_fraction_rational=}"
            # calculate stats for the ufuns after removing cardinal information
            with print_timing(
                f"Calculating scenario stats (ordinal)", level=2, verbose=verbose
            ):
                ordinal_stats = calc_scenario_stats(ordinal_ufuns, outcomes=outcomes)
            with print_timing(
                "Calculating distances (ordinal)", level=2, verbose=verbose
            ):
                dists = calc_outcome_distances(
                    tuple(_.reserved_value for _ in ordinal_ufuns), ordinal_stats
                )
            with print_timing(
                "Calculating optimality (ordinal)", level=2, verbose=verbose
            ):
                optim = calc_outcome_optimality(
                    dists, ordinal_stats, estimate_max_dist(ordinal_ufuns)
                )
            ordinal_maxs = [_[1] for _ in ordinal_stats.utility_ranges]
            ordinal_mins = [_[0] for _ in ordinal_stats.utility_ranges]
            result[key].update(
                dict(
                    ordinal_ordinal_maxs=ordinal_maxs,
                    ordinal_ordinal_mins=ordinal_mins,
                    ordinal=asdict(ordinal_stats),
                    ordinal_reserved_dists=asdict(dists),
                    ordinal_reserved_optim=asdict(optim),
                    ordinal_reserved_values=tuple(_.reserved_value for _ in ufuns),
                    ordinal_fraction_rational=ordinal_fraction_rational,
                    ordinal_is_rational=ordinal_fraction_rational > 1e-6,
                )
            )

        # now do the rotations
        for i, perm in enumerate(PERMUTAION_METHOD(range(len(ufuns)))):
            key = key_(i, f0, f1)
            if not override and key in result:
                print(f"\t[blue]Reusing {key}[/blue] ", end="")
                continue
            # save reserved values
            with print_timing(
                f"Extending stats for perm {i}", level=3, verbose=verbose
            ):
                reserved = tuple(_.reserved_value for _ in ufuns)
                new_stats = _extend_stats(
                    stats, perm, ufuns, mins, maxs, f0, f1, outcomes
                )
                # restore reserved values
                for u, r in zip(ufuns, reserved):
                    u.reserved_value = r
                result[key] = dict(perm_index=i, perm=perm, f0=f0, f1=f1)
                result[key]["reserved_values"] = tuple(_.reserved_value for _ in ufuns)
                result[key]["cardinal"] = new_stats["stats"]
                result[key]["cardinal_reserved_dists"] = new_stats["dists"]
                result[key]["cardinal_reserved_optim"] = new_stats["optim"]
                result[key]["mins"] = new_stats["mins"]
                result[key]["maxs"] = new_stats["maxs"]
                result[key][FRKEY] = new_stats["fr"]
                result[key][IRKEY] = new_stats["ir"]
        for i, perm in enumerate(PERMUTAION_METHOD(range(len(ufuns)))):
            key = key_(i, f0, f1)
            if not override and key in result:
                continue
            # save reserved values
            reserved = tuple(_.reserved_value for _ in ordinal_ufuns)
            with print_timing(
                f"Extending stats for perm {i} ({key=})", level=3, verbose=verbose
            ):
                new_stats = _extend_stats(
                    ordinal_stats,
                    perm,
                    ordinal_ufuns,
                    ordinal_mins,
                    ordinal_maxs,
                    f0,
                    f1,
                    outcomes,
                )
                # restore reserved values
                for u, r in zip(ordinal_ufuns, reserved):
                    u.reserved_value = r
                # the key will be there from the previous loop
                result[key]["ordinal_reserved_values"] = tuple(
                    _.reserved_value for _ in ordinal_ufuns
                )
                result[key]["ordinal"] = new_stats["stats"]
                result[key]["ordinal_reserved_dists"] = new_stats["dists"]
                result[key]["ordinal_reserved_optim"] = new_stats["optim"]
                result[key]["ordinal_mins"] = new_stats["mins"]
                result[key]["ordinal_maxs"] = new_stats["maxs"]
                result[key]["ordinal_fraction_rational"] = new_stats["fr"]
                result[key]["ordinal_is_rational"] = new_stats["ir"]
                if check:
                    assert (
                        new_stats["fr"] == result[key][FRKEY]
                    ), f"{new_stats['fr']=}, {result[key][f'{FRKEY}']=}"
        # restore reserved values
        for u, r in zip(ufuns, reserved):
            u.reserved_value = r
        return result

    # saving base stats (for original reserved values)
    with print_timing("Calculating base stats ", verbose=verbose, level=1):
        result = _insert_stats(-1, -1, result, ufuns, override)
    # Resetting reserved_values
    for u in ufuns:
        u.reserved_value = float(u.min()) - 1e-3
    # saving stats for every combination of fractions
    for f0, f1 in itertools.product(f0s, f1s):
        with print_timing(
            f"calculating stats for {f0=}, {f1=}", verbose=verbose, level=1
        ):
            result = _insert_stats(f0, f1, result, ufuns, override)
    # saving stats for all other reserved values
    # json.dump(result, open(dir /  STATSFILE, "w"), indent=2, sort_keys=True)
    with print_timing(f"saving all stats ", verbose=verbose):
        save_stats(result, dir, override=override)
    if verbose:
        print(f"[green]Added Stats to {str(dir.relative_to(base))}[/green]", flush=True)
    return result


def save_stats(stats: dict, path: Path, override: bool = True):
    stasts_path = path / STATSFILE
    if not stasts_path.exists() or override:
        dump(stats, stasts_path, compact=False, sort_keys=True)
    base = path / STATS_FOLDER
    base.mkdir(parents=True, exist_ok=True)
    d = dict()
    for k, v in stats.items():
        if isinstance(v, dict):
            stasts_path = base / adjustname(k)
            if not stasts_path.exists() or override:
                dump(v, stasts_path, compact=False, sort_keys=True)
            continue
        d[k] = v
    if not d:
        return
    dump(d, base / BASIC_STATS_FILE, compact=False, sort_keys=True)


def load_scenario(path: Path) -> Scenario:
    files = list(path.glob("*.yml")) + list(path.glob("*.yaml"))
    files = sorted(files)
    dicts = [yaml.load(open(_, "r"), yaml.Loader) for _ in files]
    classes = [get_class(f"negmas.{_['type']}") for _ in dicts]
    dicts = [{k: v for k, v in _.items() if k not in ("type",)} for _ in dicts]
    ufun_indices = [
        i for i in range(len(classes)) if issubclass(classes[i], BaseUtilityFunction)
    ]
    tmp = [
        i
        for i in range(len(classes))
        if not issubclass(classes[i], BaseUtilityFunction)
    ]
    assert len(tmp) == 1, f"Cannot find an outcome space {classes}"
    os_index = tmp[0]
    osdict = dicts[os_index]
    issuedicts = osdict["issues"]
    issue_types = [get_class(f"negmas.outcomes.{_['type']}") for _ in issuedicts]
    issuedicts = [
        {k: v for k, v in _.items() if k not in ("type",)} for _ in issuedicts
    ]
    osdict["issues"] = [t(**d) for t, d in zip(issue_types, issuedicts)]
    os = classes[os_index](**osdict)
    assert issubclass(
        classes[os_index], CartesianOutcomeSpace
    ), f"Not an outcome space or a ufun ({classes[os_index]=})"
    ufuns = [classes[i](**dicts[i], outcome_space=os) for i in ufun_indices]
    return Scenario(
        agenda=os, ufuns=tuple(ufuns), mechanism_type=None, mechanism_params=dict()
    )


def remove_failed_records(
    data: pd.DataFrame,
    explicit: bool = True,
    implicit: bool = False,
) -> pd.DataFrame:
    """Remove runs marked as failed."""
    data.failed = data.failed.fillna(True).astype(bool)
    data.succeeded = data.succeeded.fillna(False).astype(bool)
    if explicit:
        data = data.loc[~data.failed, :]
    if implicit:
        data = data.loc[
            (~data.run_time.isna())
            | (~data.time.isna())
            | (data.steps > 0)
            | (~data.agreement.isna())
            | data.succeeded,
            :,
        ]
    return data


def remove_runs(
    data: pd.DataFrame,
    mechanisms: Sequence[str],
    allowed: Sequence[str] = [],
    disallowed: Sequence[str] = [],
    src: Path | None = None,
):
    """Removes all runs that of given mechanisms for which not all negotiators
    are in the allowed list.

    If allowed and disallowed are empty, everything is allowed.
    """
    mechanism_col = (
        "mechanism_name" if "mechanism_name" in data.columns else "Mechanism"
    )
    fsname = (
        "first_strategy_name" if "first_strategy_name" in data.columns else "Strategy1"
    )
    ssname = (
        "second_strategy_name"
        if "second_strategy_name" in data.columns
        else "Strategy2"
    )

    strategies = set(data[fsname].unique()).union(set(data[ssname].unique()))
    not_allowed = set(disallowed)
    if allowed:
        not_allowed = not_allowed.union(strategies.difference(set(allowed)))
    not_allowed = not_allowed.intersection(strategies)
    removed = data.loc[
        (~data[mechanism_col].isin(mechanisms))
        | (
            (data[mechanism_col].isin(mechanisms))
            & (~data[fsname].isin(not_allowed))
            & (~data[ssname].isin(not_allowed))
        ),
        :,
    ]
    nremoved = len(data) - len(removed)
    if nremoved:
        print(
            f"[blue]Will remove all runs with {not_allowed} from {src} ({nremoved} records)[/blue]"
        )
        data = removed
    return data


def remove_incomplete(
    data: pd.DataFrame,
    max_incomplete: int = 0,
    ignore: Sequence[tuple[str, str]] = tuple(),
    ignore_mechanisms: Sequence[str] = tuple(),
    ignore_strategies: Sequence[str] = tuple(),
    ignore_pure: bool = False,
    ignore_impure: bool = False,
    ignore_impure_for: Sequence[str] = tuple(),
    ignore_impure_strategies: Sequence[str] = tuple(),
    ignore_pure_strategies_for: dict[str, list[str]] | None = None,
    ignore_impure_strategies_for: dict[str, list[str]] | None = None,
    ignore_all_but_strategies_for: dict[str, list[str]] | None = None,
    use_base_mechanism_name: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, list], int]:
    """Removes domains with incomplete runs.

    Args:
        max_incomplete: Maximum number of strategies not running any any domain to be allowed. The default is zero removing any such domain
        ignore: Sequence[tuple[str, str]] = tuple(),
        ignore_mechanisms: Sequence[str] = tuple(),
        ignore_strategies: Sequence[str] = tuple(),
        ignore_impure_for: Sequence[str] = tuple(),
        ignore_impure_strategies: Sequence[str] = tuple(),
        ignore_pure_strategies_for: dict[str, list[str]] | None = None,
        ignore_impure_strategies_for: dict[str, list[str]] | None = None,
        ignore_all_but_strategies_for
    """
    if use_base_mechanism_name:
        data = data.copy()
        data.mechanism_name = data.mechanism_name.str.lower()
        data.loc[data.mechanism_name.str.contains("ao"), "mechanism_name"] = "AO"
        data.loc[data.mechanism_name.str.contains("tau"), "mechanism_name"] = "TAU"
        data.loc[data.mechanism_name.str.contains("au0"), "mechanism_name"] = "TAU"
    x = data.groupby(["domain_name", "mechanism_name", "strategy_name"])[
        "succeeded"
    ].count()
    n = len(data)
    y = x.unstack(0)
    y = ~y.isna()
    y = y.astype(int)
    not_completed: dict[str, list[tuple[str, str]]] = dict()
    for c in y.columns:
        not_completed[c] = y.index[y[c] == 0].to_list()  # type: ignore
    domains = [_ for _ in not_completed.keys()]
    if verbose:
        print(f"Found {len(domains)} domains: {domains}")
    if ignore_impure_strategies:
        not_completed = {
            k: [
                _
                for _ in v
                if not (
                    "-" in _[1]
                    and any(
                        any(y.lower() in x.lower() for x in ignore_impure_strategies)
                        for y in _[1].split("-")
                    )
                )
            ]
            for k, v in not_completed.items()
        }
    if ignore_impure:
        not_completed = {
            k: [_ for _ in v if "-" not in _[1]] for k, v in not_completed.items()
        }
    if ignore_pure:
        not_completed = {
            k: [_ for _ in v if "-" in _[1]] for k, v in not_completed.items()
        }
    if ignore_impure_for:
        not_completed = {
            k: [
                _
                for _ in v
                if not (
                    "-" in _[1]
                    and any(_[0].lower() in x.lower() for x in ignore_impure_for)
                )
            ]
            for k, v in not_completed.items()
        }
    if ignore_all_but_strategies_for:
        for mech, strategies in ignore_all_but_strategies_for.items():
            not_completed = {
                k: [
                    _
                    for _ in v
                    if _[0] != mech
                    or any(_[1].lower().startswith(x.lower()) for x in strategies)
                ]
                for k, v in not_completed.items()
            }
    if ignore_pure_strategies_for:
        for mech, strategies in ignore_pure_strategies_for.items():
            not_completed = {
                k: [
                    _
                    for _ in v
                    if not (
                        _[0] == mech
                        and "-" not in _[1]
                        and any(x.lower() in _[1].lower() for x in strategies)
                    )
                ]
                for k, v in not_completed.items()
            }
    if ignore_impure_strategies_for:
        for mech, strategies in ignore_impure_strategies_for.items():
            not_completed = {
                k: [
                    _
                    for _ in v
                    if not (
                        _[0] == mech
                        and "-" in _[1]
                        and any(
                            any(y.lower() in x.lower() for x in strategies)
                            for y in _[1].split("-")
                        )
                    )
                ]
                for k, v in not_completed.items()
            }
    for x in ignore_mechanisms:
        not_completed = {
            k: [_ for _ in lst if x != _[0]] for k, lst in not_completed.items()
        }
    for x in ignore_strategies:
        not_completed = {
            k: [_ for _ in lst if x != _[1]] for k, lst in not_completed.items()
        }
    ignore_set = set(ignore)
    not_completed = {
        k: [_ for _ in v if _ not in ignore_set] for k, v in not_completed.items()
    }

    not_completed = {k: v for k, v in not_completed.items() if len(v) > max_incomplete}
    if not_completed:
        domains = list(not_completed.keys())
        if verbose:
            print(f"Will remove {len(domains)} domains: {domains}")
        data = data.loc[~data.domain_name.isin(domains), :]

    return data, not_completed, n - len(data)


def get_all_anac_agents(
    year: int, finalists_only: bool, winners_only: bool, genius10: bool = False
) -> list[str]:
    if year < 2010:
        return []

    def adjust(name: str) -> str:
        return name.replace("_", "").replace("-", "_")

    return [
        f"{GENIUSMARKER}:{adjust(name)}:{classname}"
        for name, classname in get_anac_agents(
            year=year,
            finalists_only=finalists_only,
            winners_only=winners_only,
            genius10=genius10,
        )
    ]


def get_genius_proper_class_name(s: str) -> str:
    assert s.startswith(GENIUSMARKER)
    return s.split(":")[-1]


def get_negotiator_type_name(
    x: type[Negotiator] | Callable[[], Negotiator], fullname: str
) -> str:
    try:
        issubclass(x, Negotiator)  # type: ignore
        return get_full_type_name(x)
    except TypeError:
        return fullname


def adjustname(s: str):
    "Removes first and last letters (parantheses) and replace ',', '_'"
    return f"{s.replace('(', '').replace(')', '')}.json"


def readjustname(s: str):
    "Inverts adjustname"
    if s.startswith("_"):
        return s.replace(".json", "")
    if s == BASIC_STATS_FILE:
        return s.replace(".json", "")
    return f"({s.replace('.json', '')})"


def load_stats_from_folder(folder: Path) -> dict:
    d = dict()
    for f in folder.glob("*.json"):
        if f.name == BASIC_STATS_FILE:
            d.update(load(f))
            continue
        d[readjustname(f.name)] = load(f)
    return d


def read_stats_of(domain_folder: Path, perm: int, f0, f1) -> dict[str, Any]:
    stats_key = key_(perm, f0, f1)
    stats_file_path = domain_folder / STATSBASE / adjustname(stats_key)
    if stats_file_path.exists():
        return load(stats_file_path)
    allstats = load(domain_folder / STATSFILE)
    return allstats[stats_key]


def read_all_stats(domain_folder: Path) -> dict[str, Any] | None:
    stats_path = domain_folder / STATSFILE
    stats_folder = domain_folder / STATSBASE
    allstats = None
    if stats_folder.exists():
        try:
            allstats = load_stats_from_folder(stats_folder)
        except Exception as e:
            estr = str(e).split("\n")[0]
            print(
                f"[bold red]Cannot read stats for [/bold red]{domain_folder}[red]{estr}[/red]",
                file=sys.stderr,
            )
            return None
    else:
        try:
            allstats = load(stats_path)
        except Exception as e:
            estr = str(e).split("\n")[0]
            print(
                f"[bold red]Cannot read stats for [/bold red]{domain_folder}[red]{estr}[/red]",
                file=sys.stderr,
            )
            return None
    return allstats


def find_runs_with_negative_advantage(
    f: Path, verbose: bool = False, override: bool = False
) -> pd.DataFrame:
    data = pd.read_csv(f, dtype=DTYPES)  # type: ignore
    data["util0"] = data.agreement_utils.apply(
        functools.partial(_get_value, 0, "util0")
    )
    data["util1"] = data.agreement_utils.apply(
        functools.partial(_get_value, 1, "util1")
    )
    data["reserved0"] = data.reserved.apply(
        functools.partial(_get_value, 0, "reserved0")
    )
    data["reserved1"] = data.reserved.apply(
        functools.partial(_get_value, 1, "reserved1")
    )
    data = refresh_values(data, categorical=False)
    data["Advantage1"] = (data.util0 - data.reserved0) / (1.0 - data.reserved0)
    data["Advantage2"] = (data.util1 - data.reserved1) / (1.0 - data.reserved1)
    x = data.loc[
        ~((data.Advantage1 >= -1e-6) & (data.Advantage2 >= -1e-6)),
        [
            "mechanism_name",
            "strategy_name",
            "year",
            "domain_name",
            "Advantage1",
            "Advantage2",
            "agreement_utils",
            "reserved",
            "reserved0",
            "reserved1",
            "f0",
            "f1",
            "perm_indx",
            "first_strategy_name",
            "second_strategy_name",
            "agreement",
        ],
    ]

    if not len(x):
        return pd.DataFrame([])

    print(f"[red]Negative advantage in {len(x)} cases[/red]")
    if verbose:
        for col in ["mechanism_name", "domain_name", "strategy_name", "f0", "f1"]:
            summary = x.groupby([col]).size()
            summary = summary[summary > 0]
            if len(summary) < 1:
                continue
            print(f"Negative advantage records by {col}")
            print(summary)
        x = x[
            [
                "mechanism_name",
                "year",
                "domain_name",
                "f0",
                "f1",
                "first_strategy_name",
                "second_strategy_name",
                "Advantage1",
                "Advantage2",
                "agreement_utils",
                "reserved",
                "agreement",
            ]
        ].reset_index(drop=True)
        print(x)
    if override:
        data = data.loc[((data.Advantage1 >= -1e-6) & (data.Advantage2 >= -1e-6)), :]
        data = data.drop(columns=["Advantage1", "Advantage2", "util0", "util1"])
        data = refresh_values(data, categorical=False)
        data.to_csv(f, index=False)
    return data


def refresh_values(data: pd.DataFrame, categorical=True, floating=True) -> pd.DataFrame:
    if categorical:
        for k in data.columns:
            if data[k].dtype != "category":
                continue
            data[k] = data[k].cat.remove_unused_categories()
    if floating:
        for k in data.columns:
            if not is_float_dtype(data[k].dtype):
                continue
            data[k].mask(data[k].abs() < 1e-6, 0.0, inplace=True)
    return data


# This is a special path for running on our servers. Should not affect anyone else
SCENRIOS_PATH = SCENARIOS_FOLDER
DOMAINS_PATH = BASE_FOLDER / "domains"
RESULTS_PATH = BASE_FOLDER / "results"
for p in (SCENRIOS_PATH, RESULTS_PATH):
    p.mkdir(parents=True, exist_ok=True)
PERMUTAION_METHOD = no_permutations
