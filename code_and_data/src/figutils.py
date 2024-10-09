import warnings
from pathlib import Path

import pandas as pd
import seaborn as sns
from helpers.utils import TAUNATIVE, read_data, refresh_values
from pandas.errors import DtypeWarning, SettingWithCopyWarning
from rich import print

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 2)

warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BILATERAL_YEARS = [2010, 2011, 2012, 2013, 2014, 2019, 2020, 20021, 2022]
MULTILATERAL_YEARS = [2015, 2016, 2017, 2018]
BASE = Path(__file__).absolute().parent.parent / "figs" / "ijcai"
DATA_LOC = Path(__file__).absolute().parent.parent / "serverclean"
SCENARIOS_LOC = Path(__file__).absolute().parent.parent / "scenarios"
VALID_DATASETS_ORDERED = [
    "2010",
    "2011",
    "2012",
    "2013",
    "2015",
    "2016",
    "2017",
    "2018",
    "2010finalists",
    "2015finalists",
    "2016finalists",
    "2017finalists",
    "2018finalists",
    "final",
]
VALID_DATASETS = set(VALID_DATASETS_ORDERED)
DIFFICULTY = "Rational Fraction"
OPPOSITION = "Opposition Level"
STRATEGY_TYPE = "Strategy Type"
MECHANISM_TYPE = "Session Type"
MIXED_NAME = "AOP-TAU"
IRRATIONAL_SECOND = ("AgentK", "NiceTfT", "AgentGG", "Caduceus")
TAU_BASE = "TAU "
TAU_ADAPTED_LABEL = f"{TAU_BASE}(Adapted)"
TAU_MIXED_LABEL = f"{TAU_BASE}(Mixed)"
TAU_PURE_LABEL = f"{TAU_BASE}(Pure)"
TAU_NATIVE_LABEL = f"{TAU_BASE}(CAB-WAR)"


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


def read_and_adjust(
    dataset: str = "final",
    years: tuple[int, ...] = tuple(),
    no_adapters: bool = False,
    remove_mixed: bool = True,
    separate_pure: bool = True,
    separate_cabwar: bool = False,
    remove_cab_vs_war: bool = False,
    rename_cab: bool = False,
    tau_pure: bool = False,
    remove_war: bool = False,
    with_agreements_only: bool = False,
    remove_negative_advantage: bool = False,
    no_cabwar: bool = False,
    remove_aop: bool = False,
    remove_incomplete_domains: bool = True,
    implicit_failures: bool = False,
    failures: bool = False,
    verbose: bool = False,
    multilateral: bool = True,
    bilateral: bool = True,
    privacy_factor: float | None = None,
    add_opposition: bool = True,
    correct: bool = True,
    overwrite: bool = False,
    experiment_results_path: Path = DATA_LOC,
    scenarios_path: Path = SCENARIOS_LOC,
):
    experiment_results_path = Path(experiment_results_path)
    scenarios_path = Path(scenarios_path)
    if not experiment_results_path.exists():
        print(f"Experiment results path given {experiment_results_path} does not exist")
        return None, None, None, None
    if not scenarios_path.exists() and add_opposition:
        print(f"Scenarios path given {scenarios_path} does not exist")
        return None, None, None, None
    if dataset not in VALID_DATASETS:
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    peryear = not dataset.startswith("final")
    controlled_only, original_only = not peryear, peryear
    remove_irrational_agents = peryear
    filename = (
        f"{dataset}/{dataset}.csv"
        if dataset.startswith("final")
        else f"y{dataset}/y{dataset}.csv"
    )
    figbasename = filename.split("/")[-1].replace(".csv", "")
    if "final" in filename and figbasename.startswith("y"):
        figbasename = f"final/only{figbasename}"
    files = [Path(experiment_results_path) / filename]
    data, agents = read_data(
        files,
        implicit_failures=implicit_failures,
        failures=failures,
        remove_incomplete_domains=remove_incomplete_domains,
        overwrite=overwrite,
        correct=correct,
        verbose=verbose,
    )
    for c in ["Strategy", "Strategy1", "Strategy2"]:
        data[c] = data[c].astype(str)
    data.loc[data.Strategy1 == data.Strategy2, "Strategy"] = data.loc[
        data.Strategy1 == data.Strategy2, "Strategy1"
    ]
    for c in ["Strategy", "Strategy1", "Strategy2"]:
        data[c] = data[c].astype("category")
    data = data.rename(columns=dict(Difficulty=DIFFICULTY))
    agents = agents.rename(columns=dict(Difficulty=DIFFICULTY))
    if privacy_factor is not None:
        if privacy_factor < 1e-5:
            data.AgentScore = data.Advantage
            agents.AgentScore = agents.Advantage
        else:
            data.AgentScore = (data.Advantage + privacy_factor * data.Privacy) / (
                privacy_factor + 1
            )
            agents.AgentScore = (agents.Advantage + privacy_factor * agents.Privacy) / (
                privacy_factor + 1
            )

    base_name = f"{files[0].stem}/" if files[0].stem != "final" else "final/"
    n_original, n_agents_original = len(data), len(agents)
    if not multilateral:
        data = data.loc[~data.Year.isin(MULTILATERAL_YEARS), :]
    if not bilateral:
        data = data.loc[~data.Year.isin(BILATERAL_YEARS), :]
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
    if remove_irrational_agents:
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
    if remove_aop:
        data = data.loc[~data.Mechanism.str.startswith("AO"), :]
        agents = agents.loc[~agents.Mechanism.str.startswith("AO"), :]
    if rename_cab:
        for col in (
            "Strategy",
            "Strategy1",
            "Strategy2",
            "StrategyPartner",
            "Condition",
        ):
            n = len(data)
            if n == 0:
                continue
            for df in (data, agents):
                if col not in df.columns:
                    continue
                df[col] = df[col].astype(str)
                n_before_x = len(df.loc[df[col].str.contains("CAB"), col])
                if n_before_x < 1:
                    df[col] = df[col].astype("category")
                    continue
                x = df.loc[df[col].str.contains("CAB"), col].str.replace("CAB", "SCS")
                assert (
                    len(x) == n_before_x
                ), f"{col} collapsed when trying to rename CAB"
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
    if verbose:
        print(f"Before Refreshing: {len(data)}, Agents: {len(agents)}")
    data = refresh_values(data)
    agents = refresh_values(agents)
    if verbose:
        print(f"After Refreshing: {len(data)}, Agents: {len(agents)}")
    for col in ("Type", "Strategy", "Strategy1", "Strategy2", "Strategy2"):
        data[col] = data[col].astype("category")
    data.Type = data.Type.cat.remove_unused_categories()
    data.Strategy = data.Strategy.cat.remove_unused_categories()
    data.Strategy1 = data.Strategy1.cat.remove_unused_categories()
    data.Strategy2 = data.Strategy2.cat.remove_unused_categories()
    data["Speed"] = 1 / data["Time"]
    agents["Speed"] = 1 / agents["Time"]

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
        dd.to_csv(Path.cwd().parent.parent / "negative_advantage.csv", index=False)
    if remove_negative_advantage:
        data = data.loc[(data.Advantage1 >= 0) & (data.Advantage2 >= 0), :]

    data["Speed (Relative)"] = 1 / data["Time"]
    agents["Speed (Relative)"] = 1 / agents["Time"]

    if remove_cab_vs_war:
        for col in ("Strategy1", "Strategy2"):
            data[col] = data[col].astype(str)
        for col in ("Strategy", "StrategyPartner"):
            agents[col] = agents[col].astype(str)
        data = data.loc[
            ~((data.Strategy1.isin(("CAB", "SCS"))) & (data.Strategy2 == "WAR")), :
        ]
        data = data.loc[
            ~((data.Strategy2.isin(("CAB", "SCS"))) & (data.Strategy1 == "WAR")), :
        ]
        agents = agents.loc[
            ~(
                (agents.Strategy.isin(("CAB", "SCS")))
                & (agents.StrategyPartner == "WAR")
            ),
            :,
        ]
        agents = agents.loc[
            ~(
                (agents.StrategyPartner.isin(("CAB", "SCS")))
                & (agents.Strategy == "WAR")
            ),
            :,
        ]
        for col in ("Strategy1", "Strategy2"):
            data[col] = data[col].astype("category").cat.remove_unused_categories()
        for col in ("Strategy", "StrategyPartner"):
            agents[col] = agents[col].astype("category").cat.remove_unused_categories()

    data[STRATEGY_TYPE] = MIXED_NAME
    data.loc[
        (data.Strategy1.isin(TAUNATIVE)) & (data.Strategy2.isin(TAUNATIVE)),
        STRATEGY_TYPE,
    ] = "TAU Native"
    data.loc[
        (~data.Strategy1.isin(TAUNATIVE)) & (~data.Strategy2.isin(TAUNATIVE)),
        STRATEGY_TYPE,
    ] = "AOP Native"
    data[STRATEGY_TYPE] = data[STRATEGY_TYPE].astype("category")

    agents[STRATEGY_TYPE] = MIXED_NAME
    agents.loc[(agents.Strategy.isin(TAUNATIVE)), STRATEGY_TYPE] = "TAU Native"
    agents.loc[(~agents.Strategy.isin(TAUNATIVE)), STRATEGY_TYPE] = "AOP Native"
    agents[STRATEGY_TYPE] = agents[STRATEGY_TYPE].astype("category")
    assert MIXED_NAME not in agents[STRATEGY_TYPE]

    if remove_mixed:
        if verbose:
            print(f"Removing mixed runs for TAU")
        data = data.loc[data[STRATEGY_TYPE] != MIXED_NAME, :]
        agents = agents.loc[
            (agents.Mechanism != "TAU")
            | (
                (
                    (agents.Strategy.isin(TAUNATIVE))
                    & (agents.StrategyPartner.isin(TAUNATIVE))
                )
                | (
                    (~agents.Strategy.isin(TAUNATIVE))
                    & (~agents.StrategyPartner.isin(TAUNATIVE))
                )
            ),
            :,
        ]

    tau_native_label = TAU_NATIVE_LABEL
    if rename_cab:
        tau_native_label = tau_native_label.replace("CAB", "SCS")

    data[MECHANISM_TYPE] = data.Mechanism.astype(str)
    data.loc[
        (data[STRATEGY_TYPE] == "AOP Native") & (data.Mechanism == "TAU"),
        MECHANISM_TYPE,
    ] = TAU_ADAPTED_LABEL
    data.loc[
        (data[STRATEGY_TYPE] == "TAU Native") & (data.Mechanism == "TAU"),
        MECHANISM_TYPE,
    ] = tau_native_label

    agents[MECHANISM_TYPE] = agents.Mechanism.astype(str)
    agents.loc[
        (
            (agents.Mechanism == "TAU")
            & (agents.Strategy.isin(TAUNATIVE))
            & (agents.StrategyPartner.isin(TAUNATIVE))
        ),
        MECHANISM_TYPE,
    ] = tau_native_label
    agents.loc[
        (
            (agents.Mechanism == "TAU")
            & (~agents.Strategy.isin(TAUNATIVE))
            & (~agents.StrategyPartner.isin(TAUNATIVE))
        ),
        MECHANISM_TYPE,
    ] = TAU_ADAPTED_LABEL

    if remove_mixed:
        data = data.loc[~(data[MECHANISM_TYPE] == "TAU"), :]
        agents = agents.loc[~(agents[MECHANISM_TYPE] == "TAU"), :]
    else:
        data.loc[(data[MECHANISM_TYPE] == "TAU"), MECHANISM_TYPE] = TAU_MIXED_LABEL
        agents.loc[(agents[MECHANISM_TYPE] == "TAU"), MECHANISM_TYPE] = TAU_MIXED_LABEL

    if separate_pure or separate_cabwar:
        agents.loc[
            (agents[MECHANISM_TYPE] == tau_native_label)
            & (agents.Strategy == agents.StrategyPartner),
            MECHANISM_TYPE,
        ] = TAU_PURE_LABEL
        data.loc[
            (data[MECHANISM_TYPE] == tau_native_label)
            & (data.Strategy1 == data.Strategy2),
            MECHANISM_TYPE,
        ] = TAU_PURE_LABEL
    if separate_cabwar:
        for x in ("CAB", "SCS", "WAR"):
            agents.loc[
                (agents[MECHANISM_TYPE] == TAU_PURE_LABEL)
                & (agents.Strategy.str.startswith(x)),
                MECHANISM_TYPE,
            ] = f"{TAU_BASE}({x})"
            data.loc[
                (data[MECHANISM_TYPE] == TAU_PURE_LABEL)
                & (data.Strategy1.str.startswith(x)),
                MECHANISM_TYPE,
            ] = f"{TAU_BASE}({x})"

    data[MECHANISM_TYPE] = data[MECHANISM_TYPE].astype("category")
    agents[MECHANISM_TYPE] = agents[MECHANISM_TYPE].astype("category")

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
    data[MECHANISM_TYPE] = data[MECHANISM_TYPE].cat.remove_unused_categories()
    # if verbose:
    #     print(data.groupby(STRATEGY_TYPE).count())
    #     print(data.groupby(MECHANISM_TYPE).count())

    # print(data.loc[data.Mechanism == "TAU", "Strategy"].unique())

    data["Advantage"] = (data.Advantage1 + data.Advantage2) / 2
    data["Optimality x Completeness"] = data.Score
    data["Overall Score"] = data.Advantage * data.DesignerScore * data.Speed
    data["Score"] = data.Advantage * data.DesignerScore
    data["Designer"] = data["DesignerScore"]
    data["Agent"] = data["Advantage"]

    print(
        f"Read {n_original} negotiation records ({n_agents_original} agent specific records)\n"
        f"Will use {len(data)} negotiation records ({len(agents)} agent specific records)\n"
        f"{len(data.Domain.unique())} domains, {len(data.Strategy.unique())} strategy combinations"
        f", {len(agents.Strategy.unique())} strategies\n"
    )
    data["Quality"] = data["DesignerScore"] * data["AgentScore"]

    return agents, data, base_name, filename
