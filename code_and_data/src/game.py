import itertools
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from time import perf_counter

import nashpy as npy
import numpy as np
import pandas as pd
import pygambit as pyg
import seaborn as sns
import typer
from figutils import (
    DATA_LOC,
    SCENARIOS_LOC,
    VALID_DATASETS,
    VALID_DATASETS_ORDERED,
    read_and_adjust,
)
from helpers.utils import TAUNATIVE
from matplotlib import pyplot as plt
from negmas.helpers import unique_name
from negmas.helpers.inout import dump
from negmas.helpers.strings import humanize_time
from pandas.errors import DtypeWarning, SettingWithCopyWarning
from pygambit.nash import NashComputationResult
from rich import print
from rich.progress import track
from scipy.integrate import odeint

np.random.seed(10)
random.seed(10)

pd.set_option("display.max_columns", None)
pd.set_option("display.precision", 3)

warnings.filterwarnings("ignore", category=DtypeWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path(__file__).absolute().parent.parent / "figs" / "ecai" / "games"
EPS = 1e-9

PROPOSED_COLOR = "k"

GAMES = dict(
    everything=dict(
        remove_aop=False,
        no_adapters=False,
        remove_mixed=False,
        remove_war=False,
        no_cabwar=False,
    ),
    taunative=dict(
        remove_aop=True,
        no_adapters=True,
        remove_mixed=True,
        remove_war=False,
        no_cabwar=False,
    ),
    tau=dict(
        remove_aop=True,
        no_adapters=False,
        remove_mixed=False,
        remove_war=False,
        no_cabwar=False,
    ),
    adapters=dict(
        remove_aop=True,
        no_adapters=False,
        remove_mixed=True,
        remove_war=False,
        no_cabwar=True,
    ),
    aop=dict(
        remove_aop=False,
        no_adapters=True,
        remove_mixed=False,
        remove_war=False,
        no_cabwar=True,
    ),
)

TAU = 100


def dx(x, t, A, tau: int | float = TAU):
    f = np.dot(A, x)
    phi = np.dot(f, x)
    return TAU * x * (f - phi)


def iterated_elimination(row_payoffs, col_payoffs):
    """
    Performs iterated elimination of strictly dominated strategies in a non-zero sum game
    with potentially different numbers of row and column strategies.

    Args:
        row_payoffs: A NumPy array representing the row player's payoff matrix.
        col_payoffs: A NumPy array representing the column player's payoff matrix.
                    Both matrices must have the same number of columns.

    Returns:
        A tuple of two NumPy arrays representing the indices of undominated strategies for
        the row and column players, respectively.

    Raises:
        ValueError: If the payoff matrices have different numbers of columns.
    """

    m, n = row_payoffs.shape
    _, n_cols = col_payoffs.shape
    if n != n_cols:
        raise ValueError("Payoff matrices must have the same number of columns.")

    payoff_matrix = np.stack((row_payoffs, col_payoffs), axis=2)  # Combine matrices

    undominated_row_indices = np.arange(m)  # Initialize indices
    undominated_col_indices = np.arange(n)

    while True:
        # Check for dominated rows
        dominated_rows = np.all(
            payoff_matrix > payoff_matrix[np.newaxis, :, :], axis=(2, 1)
        )
        payoff_matrix = payoff_matrix[~dominated_rows]
        undominated_row_indices = undominated_row_indices[
            ~dominated_rows
        ]  # Update indices

        # Check for dominated columns
        dominated_cols = np.all(
            payoff_matrix.T > payoff_matrix.T[:, np.newaxis, :], axis=(2, 0)
        )
        payoff_matrix = payoff_matrix[:, ~dominated_cols]
        undominated_col_indices = undominated_col_indices[
            ~dominated_cols
        ]  # Update indices

        if not dominated_rows.any() and not dominated_cols.any():
            break

    return undominated_row_indices, undominated_col_indices


def pure_solver(
    game: npy.Game,
    row_strategies,
    col_strategies,
    undominated_row_strategies,
    undominated_col_strategies,
) -> list[tuple[np.ndarray, np.ndarray]]:
    equilibria = []
    for (i, _), (j, _) in itertools.product(
        enumerate(undominated_row_strategies), enumerate(undominated_col_strategies)
    ):
        s1 = np.zeros(len(row_strategies))
        s2 = np.zeros(len(col_strategies))
        s1[i], s2[j] = 1.0, 1.0
        br = game.is_best_response(s1, s2)
        if all(br):
            equilibria.append((s1, s2))
    return equilibria


def reformat_gambit_solution(solution: NashComputationResult):
    equilibria = []
    game = solution.game
    players = game.players
    for e in solution.equilibria:
        x = [[] for _ in players]
        for i, player in enumerate(players):
            strategies = player.strategies
            for strategy in strategies:
                x[i].append(float(e[player][strategy]))
        equilibria.append(x)
    return equilibria


GAMBIT_SOLVERS = dict(
    enumpure=pyg.nash.enumpure_solve,
    enummixed=pyg.nash.enummixed_solve,
    lcp=pyg.nash.lcp_solve,
    liap=pyg.nash.liap_solve,
    logit=pyg.nash.logit_solve,
    simpdiv=pyg.nash.simpdiv_solve,
    ipa=pyg.nash.ipa_solve,
    gnm=pyg.nash.gnm_solve,
)


def gambit_solver(game: npy.Game, method: str, **kwargs):
    payoff1, payoff2 = game.payoff_matrices
    pyg_game = pyg.Game.from_arrays(payoff1, payoff2)
    solution = GAMBIT_SOLVERS[method](pyg_game, **kwargs)
    return reformat_gambit_solution(solution)


def enumpure_solver(game: npy.Game) -> list[tuple[np.ndarray, np.ndarray]]:
    return gambit_solver(game, method="enumpure")


def enummixed_solver(
    game: npy.Game,
    rational: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return gambit_solver(game, method="enummixed", rational=rational)


def lcp_solver(game: npy.Game, **kwargs):
    return gambit_solver(game, method="lcp", **kwargs)


def liap_solver(game: npy.Game, **kwargs):
    return gambit_solver(game, method="liap", **kwargs)


def logit_solver(game: npy.Game, **kwargs):
    return gambit_solver(game, method="logit", **kwargs)


def simpdiv_solver(game: npy.Game, **kwargs):
    return gambit_solver(game, method="simpdiv", **kwargs)


def ipa_solver(game: npy.Game, **kwargs):
    return gambit_solver(game, method="ipa", **kwargs)


def gnm_solver(game: npy.Game, **kwargs):
    return gambit_solver(game, method="gnm", **kwargs)


SOLVERS = dict(
    enumpure=enumpure_solver,
    lcp=lcp_solver,
    liap=liap_solver,
    lh=npy.Game.lemke_howson_enumeration,
    support=npy.Game.support_enumeration,
    vertex=npy.Game.vertex_enumeration,
    enummixed=enummixed_solver,
    logit=logit_solver,
    simpdiv=simpdiv_solver,
    ipa=ipa_solver,
    gnm=gnm_solver,
)


def process(
    dataset: str,
    years: tuple[int, ...],
    simplify: bool,
    solver: str,
    per: str,
    game: str,
    rename_cab: bool,
    tau_pure: bool,
    auto_limit: int,
    separate_pure: bool,
    separate_cabwar: bool,
    with_agreements_only: bool,
    remove_negative_advantage: bool,
    remove_incomplete_domains: bool,
    remove_dominated_equilibria: bool,
    implicit_failures: bool,
    failures: bool,
    verbose: bool,
    name_stem,
    multilateral: bool,
    bilateral: bool,
    privacy_factor: float,
    add_opposition: bool,
    unknown_order: bool,
    fillna: float,
    experiment_results_path: Path,
    scenarios_path: Path,
    base: Path = BASE,
    egt: bool = True,
    equilibria: bool = True,
    move_cabwar_first: bool = False,
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = True,
    save_all_figs: bool = False,
    eps_egt: float = 1e-3,
    generations: int = 10_000_000,
    resolution: int = 100,
    override_figs: bool = True,
    ncol: int = 3,
    stat: str = "mean",
    fixation: bool = False,
    moran: bool = False,
    ficticious: bool = True,
    no_pure: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    if game == "taunative":
        generations = generations * 10
    runinfo = f"{dataset}: {game}, {stat}, {solver}, {per}"
    print(f"[blue]Working for dataset[/blue] {dataset} [yellow]({runinfo})[/yellow]")
    sns.set_context(
        "talk", rc={"legend.fontsize": 30, "xtick.labelsize": 24, "ytick.labelsize": 24}
    )
    info = dict(
        dataset=dataset,
        years=years,
        simplify=simplify,
        solver=solver,
        per=per,
        game=game,
        rename_cab=rename_cab,
        tau_pure=tau_pure,
        auto_limit=auto_limit,
        separate_pure=separate_pure,
        separate_cabwar=separate_cabwar,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        remove_incomplete_domains=remove_incomplete_domains,
        remove_dominated_equilibria=remove_dominated_equilibria,
        implicit_failures=implicit_failures,
        failures=failures,
        verbose=verbose,
        name_stem=name_stem,
        multilateral=multilateral,
        bilateral=bilateral,
        privacy_factor=privacy_factor,
        add_opposition=add_opposition,
        unknown_order=unknown_order,
        fillna=fillna,
        experiment_results_path=experiment_results_path,
        scenarios_path=scenarios_path,
        eps_egt=eps_egt,
        base=base,
        generations=generations,
        resolution=resolution,
    )
    agents, data, _, _ = read_and_adjust(
        dataset=dataset,
        years=years,
        rename_cab=rename_cab,
        separate_pure=separate_pure,
        separate_cabwar=separate_cabwar,
        tau_pure=tau_pure,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
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
        **GAMES[game],
    )
    assert (
        agents is not None and len(agents) > 0 and data is not None and len(data) > 0
    ), f"Found no data for {dataset}"
    tick = perf_counter()

    assert isinstance(agents, pd.DataFrame)
    for c in ("Strategy1", "Strategy2"):
        data[c] = data[c].astype("category")
        data[c] = data[c].cat.remove_unused_categories()
    if game in ("aop", "adapters"):
        for s in ("CAB", "WAR", "SCS"):
            for c in ("Strategy1", "Strategy2"):
                assert (
                    s not in data[c].unique()
                ), f"Found TAU native in {game}: {data[c].unique()}"
    if game == "taunative":
        for c in ("Strategy1", "Strategy2"):
            assert (
                len(set(data[c].unique()).difference(set(("CAB", "WAR", "SCS")))) == 0
            ), f"Found non-native TAU strategies in {game}: {data[c].unique()}"
    if game in ("tau", "adapters"):
        assert set(data["Mechanism"].unique()) == set(
            ["TAU"]
        ), f"Found AOP mechanisms in {game}: {data['Mechanism'].unique()}"
    if game == "aop":
        assert (
            len(set(["TAU"]).difference(set(data["Mechanism"].unique()))) == 1
        ), f"Found TAU mechanisms in {game}: {data['Mechanism'].unique()}"
    if name_stem:
        base = base / name_stem
    params = dict(
        solver=solver,
        dataset=dataset,
        game_name=game,
        remove_dominated_equilibria=remove_dominated_equilibria,
        auto_limit=auto_limit,
        fillna=fillna,
        runinfo=runinfo,
        no_pure=no_pure,
    )
    results, egt_results, fixation_results = [], [], []
    ficticious_results, moran_results = [], []

    egtparams = dict(
        generations=generations,
        resolution=resolution,
        override_figs=override_figs,
        ncol=ncol,
    )
    fictparams = dict(
        ncol=ncol,
    )
    if not per or per == "agg":
        mybase = base / "payoffs" / stat / game / dataset / "agg"
        game_info = make_payoffs(
            data,
            fillna,
            unknown_order,
            simplify,
            mybase,
            move_cabwar_first=move_cabwar_first,
            stat=stat,
            runinfo=runinfo,
            rename_cab=rename_cab,
        )
        if egt:
            try:
                p = do_egt(
                    game_info,
                    fig_name=f"{stat}_{game}_{dataset}_agg",
                    base=base / "egtfigs",
                    **egtparams,
                )
                selected_strategies = [i for i, x in enumerate(p) if x > eps_egt]
                egt_results.append(
                    dict(
                        dataset=dataset,
                        game=game,
                        probabilities=[p[_] for _ in selected_strategies],
                        strategies=[
                            game_info.row_strategies[_] for _ in selected_strategies
                        ],
                    )
                )
            except Exception as e:
                print(
                    f"[magenta]{runinfo}[/magenta]: [red]Failed EGT in {game}  for {dataset} per {per} (aggregate)[/red]: {e}"
                )
        try:
            if equilibria:
                results = run_for_dataset(**params, game_info=game_info)  # type: ignore
        except Exception as e:
            print(
                f"[magenta]{runinfo}[/magenta]: [red]Failed Nash in {game} for {dataset} per {per} (aggregate)[/red]: {e}"
            )
        if ficticious:
            ps = ficticious_play(
                game_info,
                fig_name=f"{stat}_{game}_{dataset}_agg",
                base=base / "ficticious",
                **fictparams,
            )
            selected_strategies = [
                [i for i, x in enumerate(p) if x > eps_egt] for p in ps
            ]
            ficticious_results.append(
                dict(
                    dataset=dataset,
                    game=game,
                    probabilities_first=[ps[0][_] for _ in selected_strategies[0]],
                    probabilities_second=[ps[1][_] for _ in selected_strategies[1]],
                    strategies_first=[
                        game_info.row_strategies[_] for _ in selected_strategies[0]
                    ],
                    strategies_second=[
                        game_info.row_strategies[_] for _ in selected_strategies[1]
                    ],
                )
            )
        if moran:
            selected = do_moran(game_info=game_info)
            print(f"[magenta]{runinfo}[/magenta]: Moran found {selected}")
            moran_results.append(
                {
                    "stat": stat,
                    "dataset": dataset,
                    "game": game,
                    "selected": selected,
                }
            )
        if fixation:
            p = fixation_prob(game_info=game_info)
            selection = dict()
            for k, vv in p.items():
                if vv < eps_egt:
                    continue
                selection[tuple(set([game_info.row_strategies[_] for _ in k]))] = vv

            print(f"[magenta]{runinfo}[/magenta]: Fixation found {selection}")
            fixation_results += [
                {
                    "stat": stat,
                    "dataset": dataset,
                    "game": game,
                    "strategies": k,
                    "prob": vv,
                }
                for k, vv in selection.items()
            ]

    else:
        vals = data[per].unique()
        for v in vals:
            d = data.loc[data[per] == v, :]
            if not len(d):  # type: ignore
                continue
            print(f"[magenta]{runinfo}[/magenta]: Working for {per} = {v}:")
            mybase = base / "payoffs" / stat / game / dataset / str(v)
            game_info = make_payoffs(
                d,
                fillna,
                unknown_order,
                simplify,
                mybase,
                move_cabwar_first=move_cabwar_first,
                stat=stat,
                runinfo=runinfo,
                rename_cab=rename_cab,
            )
            try:
                if equilibria:
                    r = run_for_dataset(**params, game_info=game_info)  # type: ignore
                    for _ in r:
                        _[per] = v
                    if len(r) < 1:
                        print(
                            f"[magenta]{runinfo}[/magenta]: [red]Did not find any results [red]"
                        )
                    results += r
            except Exception as e:
                print(
                    f"[magenta]{runinfo}[/magenta]: [red]Failed Nash in {game} for {dataset} per {per} for value {v}[/red]: {e}"
                )
            try:
                if egt:
                    p = do_egt(
                        game_info,
                        fig_name=f"{stat}_{game}_{dataset}_{v}",
                        base=base / "egtfigs" / per if save_all_figs else None,
                        **egtparams,
                    )
                    selected_strategies = [i for i, x in enumerate(p) if x > eps_egt]
                    egt_results.append(
                        {
                            "stat": stat,
                            "dataset": dataset,
                            "game": game,
                            "probabilities": [p[_] for _ in selected_strategies],
                            "strategies": [
                                game_info.row_strategies[_] for _ in selected_strategies
                            ],
                            per: v,
                        }
                    )
            except Exception as e:
                print(
                    f"[magenta]{runinfo}[/magenta]: [red]Failed EGT in {game} for {dataset} per {per} for value {v}[/red]: {e}"
                )
            if ficticious:
                ps = ficticious_play(
                    game_info,
                    fig_name=f"{stat}_{game}_{dataset}_{v}",
                    base=base / "ficticious" / per if save_all_figs else None,
                    **fictparams,
                )
                selected_strategies = [
                    [i for i, x in enumerate(p) if x > eps_egt] for p in ps
                ]
                ficticious_results.append(
                    {
                        "dataset": dataset,
                        "game": game,
                        "probabilities_first": [
                            ps[0][_] for _ in selected_strategies[0]
                        ],
                        "probabilities_second": [
                            ps[1][_] for _ in selected_strategies[1]
                        ],
                        "strategies_first": [
                            game_info.row_strategies[_] for _ in selected_strategies[0]
                        ],
                        "strategies_second": [
                            game_info.row_strategies[_] for _ in selected_strategies[1]
                        ],
                        per: v,
                    }
                )
            if moran:
                selected = do_moran(game_info=game_info)
                print(f"[magenta]{runinfo}[/magenta]: Moran found {selected} for {v}")
                moran_results.append(
                    {
                        "stat": stat,
                        "dataset": dataset,
                        "game": game,
                        "selected": selected,
                        per: v,
                    }
                )
            if fixation:
                p = fixation_prob(game_info=game_info)
                selection = dict()
                for k, vv in p.items():
                    if vv < eps_egt:
                        continue
                    selection[tuple(set([game_info.row_strategies[_] for _ in k]))] = vv

                fixation_results += [
                    {
                        "stat": stat,
                        "dataset": dataset,
                        "game": game,
                        "strategies": k,
                        "prob": vv,
                        per: v,
                    }
                    for k, vv in selection.items()
                ]
    df = pd.DataFrame.from_records(results)
    df_egt = pd.DataFrame.from_records(egt_results)
    df_fixation = pd.DataFrame.from_records(fixation_results)
    df_moran = pd.DataFrame.from_records(moran_results)
    df_ficticious = pd.DataFrame.from_records(ficticious_results)

    # print(df)
    print(
        f"[magenta]{runinfo}[/magenta]: Done Processing in {humanize_time(perf_counter() - tick)}\n"
        f"\tfound {len(df)} Nash equilibria"
        f"\tfound {len(df_egt)} ETG results\n"
        f"\tfound {len(df_fixation)} fixation results"
        f"\tfound {len(df_moran)} moran results\n"
        f"\tfound {len(df_ficticious)} ficcticious run results"
    )
    basename = (
        f"{stat}_{game}_{dataset}_{per}"
        if per and per != "agg"
        else f"{stat}_{game}_{dataset}_agg"
    )

    if df is not None and len(df):
        (base / "equilibria").mkdir(parents=True, exist_ok=True)
        df.to_csv(
            base / "equilibria" / f"{basename}_equilibria.csv",
            index_label="index",
        )
    if df_egt is not None and len(df_egt):
        (base / "egt").mkdir(parents=True, exist_ok=True)
        df_egt.to_csv(
            base / "egt" / f"{basename}_egt.csv",
            index_label="index",
        )
    if df_fixation is not None and len(df_fixation):
        (base / "fixation").mkdir(parents=True, exist_ok=True)
        df_fixation.to_csv(
            base / "fixation" / f"{basename}_fixation.csv",
            index_label="index",
        )
    if df_moran is not None and len(df_moran):
        (base / "moran").mkdir(parents=True, exist_ok=True)
        df_moran.to_csv(
            base / "moran" / f"{basename}_moran.csv",
            index_label="index",
        )
    if df_ficticious is not None and len(df_ficticious):
        (base / "ficticious").mkdir(parents=True, exist_ok=True)
        df_ficticious.to_csv(
            base / "ficticious" / f"{basename}_ficticious.csv",
            index_label="index",
        )
    return df, df_egt, info


@dataclass()
class GameInfo:
    payoff1: np.ndarray
    payoff2: np.ndarray
    row_strategies: list[str]
    col_strategies: list[str]
    undominated_row_strategies: list[str]
    undominated_col_strategies: list[str]


def savefig(name: str, base: Path, exts=("pdf",), override=True):
    name = name.replace(" ", "_")
    for ext in exts:
        path = base / f"{name}.{ext}"
        path.parent.mkdir(exist_ok=True, parents=True)
        if not override and path.exists():
            path = (
                path.parent
                / f'{unique_name(path.stem, add_time=True, add_host=False, rand_digits=1, sep="")}.{ext}'
            )
        plt.savefig(path, bbox_inches="tight")


def ficticious_play(
    game_info: GameInfo,
    base: Path | None,
    fig_name: str,
    iterations=1000_000,
    ncol: int = 4,
    log_scale: bool = True,
) -> tuple[list[float], list[float]]:
    print("[brown]Starting Fictitious Play[/brown]")
    assert all(
        a == b for a, b in zip(game_info.row_strategies, game_info.col_strategies)
    )
    counts = [
        _
        for _ in npy.Game(game_info.payoff1, game_info.payoff2).fictitious_play(
            iterations=iterations
        )
    ]
    probs = counts
    probs = np.asarray([[[_ / sum(a) for _ in a] for a in p] for p in probs])
    if base:
        for indx, indx_name in enumerate(("first", "second")):
            fig = plt.figure(figsize=(23, 8))
            n_colors = len(game_info.row_strategies)
            cm = plt.get_cmap("brg")
            ax = fig.add_subplot(111)
            colors_ = [cm(1.0 * i / n_colors) for i in range(n_colors)]
            ax.set_prop_cycle(color=colors_)
            style_cycle = [
                "dashdot",
                (0, (3, 5, 1, 5, 1, 5)),
                (0, (1, 10)),
                (5, (10, 3)),
                (0, (5, 10)),
                (0, (5, 5)),
                (0, (3, 5, 1, 5)),
                (0, (3, 10, 1, 10, 1, 10)),
                (0, (3, 1, 1, 1)),
                (0, (1, 1)),
            ]
            nxt, n_styles = 0, len(style_cycle)
            for i, s in enumerate(game_info.row_strategies):
                linestyle, linewidth, clr = "-", 2, PROPOSED_COLOR
                if s.startswith("AO") and "3" in s:
                    linestyle, linewidth, clr = (
                        style_cycle[nxt % n_styles],
                        1,
                        colors_[i % n_colors],
                    )
                elif s.startswith("AO") and "3" not in s:
                    linestyle, linewidth, clr = (
                        style_cycle[nxt % n_styles],
                        2,
                        colors_[i % n_colors],
                    )
                elif s.startswith("TAU") and not any(_ in s for _ in TAUNATIVE):
                    linestyle, linewidth, clr = (
                        (0, (3, 1, 1, 1, 1, 1)),
                        3,
                        colors_[i % n_colors],
                    )
                elif "CAB" in s or "SCS" in s:
                    linestyle, linewidth, clr = "dashed", 5, PROPOSED_COLOR
                nxt += 1

                plt.plot(
                    probs[:, indx, i],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=s,
                    color=clr,
                )
                if log_scale:
                    plt.xscale("log")
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=ncol)
            savefig(f"{fig_name}_{indx_name}", base, override=True)
    return probs[-1]  # type: ignore


def fixation_prob(game_info: GameInfo, n_per_strategy: int = 5, repetitions: int = 100):
    print("[yellow]Starting Fixation Prob calculation[/yellow]")
    n_strategies = len(game_info.row_strategies)
    population = []
    for i in range(n_strategies):
        population += [i] * n_per_strategy
    return npy.Game(game_info.payoff1, game_info.payoff2).fixation_probabilities(
        initial_population=tuple(population),
        repetitions=repetitions,
    )


def do_moran(
    game_info: GameInfo,
    n_per_strategy: int = 5,
    mutation: float | int = 0,
) -> list[str]:
    print("[brown]Starting Moran Process[/brown]")
    n_strategies = len(game_info.row_strategies)
    population = []
    for i in range(n_strategies):
        population += [i] * n_per_strategy
    result = list(
        npy.Game(game_info.payoff1, game_info.payoff2).moran_process(
            population, mutation_probability=mutation
        )
    )[-1].tolist()  # type: ignore
    result = list(set(result))  # type: ignore
    return [game_info.row_strategies[_] for _ in result]


def do_egt(
    game_info: GameInfo,
    base: Path | None,
    fig_name: str,
    y0: list[float] | None = None,
    generations: int = 10_000_000,
    resolution: int = 100,
    override_figs: bool = True,
    ncol: int = 4,
    log_scale=False,
    tau: float | int = TAU,
):
    print("[magenta]Starting Replicator Dynamics calculation[/magenta]")
    assert all(
        a == b for a, b in zip(game_info.row_strategies, game_info.col_strategies)
    )
    A = (game_info.payoff2 + game_info.payoff1) / 2
    xvals = np.linspace(0, resolution, generations)
    p = odeint(
        func=partial(dx, tau=tau),
        y0=np.asarray([1 / len(A)] * len(A)) if y0 is None else y0,
        t=xvals,
        args=(A,),
    )
    if base:
        fig = plt.figure(figsize=(23, 8))
        sns.set_context("poster")

        # Set the default font size for axes
        plt.rc("axes", labelsize=24, titlesize=28)
        n_colors = len(game_info.row_strategies)
        cm = plt.get_cmap("brg")
        ax = fig.add_subplot(111)
        colors_ = [cm(1.0 * i / n_colors) for i in range(n_colors)]
        ax.set_prop_cycle(color=colors_)
        style_cycle = [
            "dashdot",
            (0, (3, 5, 1, 5, 1, 5)),
            (0, (1, 10)),
            (5, (10, 3)),
            (0, (5, 10)),
            (0, (5, 5)),
            (0, (3, 5, 1, 5)),
            (0, (3, 10, 1, 10, 1, 10)),
            (0, (3, 1, 1, 1)),
            (0, (1, 1)),
        ]
        nxt, n_styles = 0, len(style_cycle)
        for i, s in enumerate(game_info.row_strategies):
            linestyle, linewidth, clr = "-", 2, PROPOSED_COLOR
            if s.startswith("AO") and "3" in s:
                linestyle, linewidth, clr = (
                    style_cycle[nxt % n_styles],
                    1,
                    colors_[i % n_colors],
                )
            elif s.startswith("AO") and "3" not in s:
                linestyle, linewidth, clr = (
                    style_cycle[nxt % n_styles],
                    2,
                    colors_[i % n_colors],
                )
            elif s.startswith("TAU") and not any(_ in s for _ in TAUNATIVE):
                linestyle, linewidth, clr = (
                    (0, (3, 1, 1, 1, 1, 1)),
                    3,
                    colors_[i % n_colors],
                )
            elif "CAB" in s or "SCS" in s:
                linestyle, linewidth, clr = "dashed", 5, PROPOSED_COLOR
            nxt += 1

            plt.plot(
                xvals / xvals.max(),
                p[:, i],
                linestyle=linestyle,
                linewidth=linewidth,
                label=s,
                color=clr,
            )
            if log_scale:
                plt.xscale("log")
        plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=ncol)
        plt.xlabel("Time (fraction of simulation time)")
        plt.ylabel("Fraction of the Population")
        savefig(fig_name, base, override=override_figs)
    return p[-1]


def make_payoffs(
    data,
    fillna,
    unknown_order,
    simplify,
    base,
    move_cabwar_first,
    runinfo,
    rename_cab,
    stat="mean",
):
    df = data[["Mechanism", "Strategy1", "Strategy2", "Advantage1", "Advantage2"]]
    df.Mechanism = df.Mechanism.astype(str)
    df.Strategy1 = df.Strategy1.astype(str)
    df.Strategy2 = df.Strategy2.astype(str)
    df = df.groupby(["Mechanism", "Strategy1", "Strategy2"]).apply(stat).reset_index()
    if len(df.Mechanism.unique()) > 1:
        df["Cond1"] = df["Mechanism"].astype(str) + "-" + df["Strategy1"].astype(str)
        df["Cond2"] = df["Mechanism"].astype(str) + "-" + df["Strategy2"].astype(str)
    else:
        df["Cond1"] = df["Strategy1"]
        df["Cond2"] = df["Strategy2"]
    df["Cond1"] = df["Cond1"].astype("category").cat.remove_unused_categories()
    df["Cond2"] = df["Cond2"].astype("category").cat.remove_unused_categories()
    a1 = df.pivot(index="Cond1", columns="Cond2", values="Advantage1")
    row_strategies = a1.index.tolist()
    col_strategies = a1.columns.tolist()
    assert all(
        [a == b for a, b in zip(row_strategies, col_strategies)]
    ), f"Row and Col strategies are not equal \n{row_strategies=}\n{col_strategies=}"
    indices = None
    if move_cabwar_first:
        indices = np.asarray(
            [
                i
                for i, _ in enumerate(row_strategies)
                if _.endswith("CAB") or _.endswith("SCS")
            ]
            + [i for i, _ in enumerate(row_strategies) if _.endswith("WAR")]
            + [
                i
                for i, _ in enumerate(row_strategies)
                if not any([_.endswith(x) for x in ("CAB", "SCS", "WAR")])
            ]
        )
    a2 = df.pivot(index="Cond1", columns="Cond2", values="Advantage2")
    row_strategies2 = a2.index.tolist()
    col_strategies2 = a2.columns.tolist()
    assert all(
        [a == b for a, b in zip(row_strategies, row_strategies2)]
    ), f"First and second row strategies are not equal \n{row_strategies=}\n{row_strategies2=}"
    assert all(
        [a == b for a, b in zip(col_strategies, col_strategies2)]
    ), f"First and second col strategies are not equal \n{col_strategies=}\n{col_strategies2=}"
    a1 = a1.fillna(fillna)
    a2 = a2.fillna(fillna)
    a1 = a1.values
    a2 = a2.values
    if indices is not None:
        a1 = a1[indices][:, indices]
        a2 = a2[indices][:, indices]
        row_strategies = [row_strategies[_] for _ in indices]
        col_strategies = [col_strategies[_] for _ in indices]
    if unknown_order:
        a1 = a2 = 0.5 * (a1 + a2)
    if simplify:
        undominated_row_indices, undominated_col_indices = iterated_elimination(a1, a2)
        undominated_row_strategies = [
            row_strategies[_] for _ in undominated_row_indices
        ]
        undominated_col_strategies = [
            col_strategies[_] for _ in undominated_col_indices
        ]
        print(
            f"[magenta]{runinfo}[/magenta]: {undominated_row_strategies=}\n{undominated_col_strategies=}"
        )
        payoff1 = a1[undominated_row_strategies, undominated_col_strategies]
        payoff2 = a2[undominated_row_strategies, undominated_col_strategies]
    else:
        payoff1, payoff2 = a1, a2
        undominated_row_strategies = row_strategies
        undominated_col_strategies = col_strategies
        undominated_row_indices = list(range(a1.shape[0]))
        undominated_col_indices = list(range(a1.shape[1]))
    (base / "payoffs").mkdir(exist_ok=True, parents=True)
    np.savetxt(base / "payoffs" / "first.csv", a1, delimiter=",")
    np.savetxt(base / "payoffs" / "second.csv", a2, delimiter=",")
    st = dict(row=row_strategies, col=col_strategies)
    dump(st, base / "payoffs" / "strategies.yaml")

    if rename_cab:
        row_strategies = [_.replace("CAB", "SCS") for _ in row_strategies]
        col_strategies = [_.replace("CAB", "SCS") for _ in col_strategies]
        undominated_row_strategies = [
            _.replace("CAB", "SCS") for _ in undominated_row_strategies
        ]
        undominated_col_strategies = [
            _.replace("CAB", "SCS") for _ in undominated_col_strategies
        ]
    return GameInfo(
        payoff1,
        payoff2,
        row_strategies,
        col_strategies,
        undominated_row_strategies,
        undominated_col_strategies,
    )


def incompatible_protocols(x: float, fillna: float) -> bool:
    return x <= fillna + 1e-3


def run_for_dataset(
    dataset: str,
    game_name: str,
    solver: str,
    remove_dominated_equilibria: bool,
    auto_limit: int,
    game_info: GameInfo,
    fillna: float,
    runinfo: str,
    no_pure: bool = False,
) -> list:
    payoff1, payoff2 = game_info.payoff1, game_info.payoff2
    row_strategies, col_strategies = game_info.row_strategies, game_info.row_strategies
    undominated_col_strategies = game_info.undominated_col_strategies
    undominated_row_strategies = game_info.undominated_row_strategies
    game = npy.Game(payoff1, payoff2)
    solutions = []
    equilibria = []

    def remove_repetitions(
        solutions: list[dict], remove_dominated_equilibria=remove_dominated_equilibria
    ) -> list[dict]:
        selected = []
        solutions.reverse()

        def match(a: dict, b: dict):
            if a["dataset"] != b["dataset"]:
                return False
            for c in ("first", "second"):
                d1, d2 = a[c], b[c]
                for k in itertools.chain(d1.keys(), d2.keys()):
                    if abs(d1.get(k, 0) - d2.get(k, 0)) > EPS:
                        return False
            return True

        for i, s in enumerate(solutions):
            if incompatible_protocols(s["u1"], fillna) or incompatible_protocols(
                s["u2"], fillna
            ):
                continue
            has_matches = False
            if remove_dominated_equilibria:
                for j in range(0, len(solutions)):
                    if j == i:
                        continue
                    s2 = solutions[j]
                    if s["u1"] < s2["u1"] and s["u2"] < s2["u2"]:
                        has_matches = True
                        break
                if has_matches:
                    continue
            for j in range(i + 1, len(solutions)):
                s2 = solutions[j]
                if match(s, s2):
                    has_matches = True
                    break
            if not has_matches:
                selected.append(s)

        selected.reverse()
        return selected

    print(
        f"[magenta]{runinfo}[/magenta]: We have {len(undominated_row_strategies)} row strategies and {len(undominated_col_strategies)} col strategies in the payoff matrices"
    )
    if (
        len(undominated_row_strategies) > auto_limit
        or len(undominated_col_strategies) > auto_limit
    ) and solver == "all":
        print(
            f"[magenta]{runinfo}[/magenta]: We have too many strategies (>{auto_limit}). Switching from all solvers to any solver"
        )
        solver = "any"
    pure_solver_lambda = partial(
        pure_solver,
        row_strategies=row_strategies,
        col_strategies=col_strategies,
        undominated_row_strategies=undominated_row_strategies,
        undominated_col_strategies=undominated_col_strategies,
    )
    if solver in ("none", "pure"):
        solvers = ["pure"]
    elif solver not in ("any", "nonpure", "all"):
        solvers = ["pure", solver] if not no_pure else [solver]
    else:
        solvers = ["pure"] + list(SOLVERS.keys()) if not no_pure else [solver]
    print(f"[magenta]{runinfo}[/magenta]: Using Solvers: {solvers}")
    used = set()
    for current_solver in solvers:
        if current_solver in used:
            continue
        used.add(current_solver)
        print(f"[magenta]{runinfo}[/magenta]: Attempting {current_solver}")
        method = SOLVERS.get(current_solver, pure_solver_lambda if "pure" else None)
        if not method:
            print(
                f"[magenta]{runinfo}[/magenta]: [yellow]warning[/yellow] Cannot find current_solver{current_solver}"
            )
            continue
        current_e = []
        try:
            current_e = list(method(game))
            equilibria += current_e
        except Exception as exp:
            print(
                f"[magenta]{runinfo}[/magenta]: [red]ERROR[/red] {current_solver} raise exception {exp}"
            )
        if len(current_e) < 1:
            print(
                f"\t[magenta]{runinfo}[/magenta]: [yellow]No equilibria for {current_solver}[/yellow]"
            )
            continue
        print(
            f"\t[magenta]{runinfo}[/magenta]: [green]{len(current_e)} equilibria for {current_solver}[/green]"
        )
        max_welfare = np.max(payoff1 + payoff2)
        for e in equilibria:
            u1, u2 = game[e[0], e[1]]
            s1 = dict(_ for _ in zip(undominated_row_strategies, e[0]) if _[1] > EPS)
            s2 = dict(_ for _ in zip(undominated_col_strategies, e[1]) if _[1] > EPS)
            pure = set(s1.keys()) == set(s2.keys())
            solutions.append(
                dict(
                    dataset=dataset,
                    game=game_name,
                    solver=current_solver,
                    first=s1,
                    second=s2,
                    pure=pure,
                    u1=u1,
                    u2=u2,
                    welfare=(u1 + u2) / (max_welfare if max_welfare else 1.0),
                    n_first_strategies=len(undominated_row_strategies),
                    n_second_strategies=len(undominated_col_strategies),
                )
            )

        if solver == "any" and len(equilibria) > 0:
            break
        if solver == "nonpure" and current_solver != "pure" and len(equilibria) > 0:
            break

    def sorter(x: dict):
        return (-x["welfare"], -int(x["pure"]))

    solutions = remove_repetitions(solutions)
    solutions = sorted(solutions, key=sorter)
    return solutions


def main(
    # dataset: Literal["final", "2010", "2011", "2012", "2013", "2015", "2016"] = "final",
    dataset: str = "all",
    game: str = "all",
    solver: str = "nonpure",
    auto_limit: int = 1000,
    simplify: bool = False,
    unknown_order: bool = False,
    per: str = "agg",
    fillna: float = 0,
    remove_dominated_equilibria: bool = True,
    year: list[int] = [],
    name_stem: str = "",
    rename_cab: bool = False,
    remove_incomplete_domains: bool = True,
    tau_pure: bool = False,
    separate_pure: bool = True,
    separate_cabwar: bool = True,
    with_agreements_only: bool = False,
    remove_negative_advantage: bool = False,
    implicit_failures: bool = False,
    failures: bool = False,
    multilateral: bool = True,
    bilateral: bool = True,
    add_opposition: bool = True,
    experiment_results_path: Path = DATA_LOC,
    privacy_factor: float = 0,
    scenarios_path=SCENARIOS_LOC,
    verbose: bool = False,
    max_cores: int = 0,
    egt: bool = True,
    fixation: bool = False,
    moran: bool = False,
    ficticious: bool = False,
    equilibria: bool = True,
    cabwar_first: bool = False,
    correct_reading_errors: bool = True,
    overwrite_data_files: bool = True,
    dry_run: bool = False,
    raise_exceptions: bool = False,
    save_all_figs: bool = False,
    eps_egt: float = 1e-3,
    generations: int = 10_000_000,
    resolution: int = 100,
    base: Path = BASE,
    stat: str = "mean",
    override_figs: bool = True,
    ncol: int = 4,
    add_pure: bool = True,
):
    print(f"Working for dataset {dataset}")
    print(f"Will get results from {experiment_results_path}")
    print(f"Will get scenario information from {scenarios_path}")
    if dataset not in VALID_DATASETS and dataset != "all":
        print(
            f"{dataset} is not a valid dataset: Valid values are: all, any of {VALID_DATASETS}"
        )
    params = dict(
        rename_cab=rename_cab,
        egt=egt,
        equilibria=equilibria,
        auto_limit=auto_limit,
        unknown_order=unknown_order,
        separate_pure=separate_pure,
        separate_cabwar=separate_cabwar,
        correct_reading_errors=correct_reading_errors,
        overwrite_data_files=overwrite_data_files,
        override_figs=override_figs,
        no_pure=not add_pure,
        fixation=fixation,
        moran=moran,
        ficticious=ficticious,
        tau_pure=tau_pure,
        remove_dominated_equilibria=remove_dominated_equilibria,
        ncol=ncol,
        fillna=fillna,
        with_agreements_only=with_agreements_only,
        remove_negative_advantage=remove_negative_advantage,
        remove_incomplete_domains=remove_incomplete_domains,
        implicit_failures=implicit_failures,
        failures=failures,
        name_stem=name_stem,
        years=tuple(year),
        multilateral=multilateral,
        bilateral=bilateral,
        privacy_factor=privacy_factor,
        add_opposition=add_opposition,
        experiment_results_path=experiment_results_path,
        scenarios_path=scenarios_path,
        verbose=verbose,
        simplify=simplify,
        solver=solver,
        game=game,
        per=per,
        base=base,
        move_cabwar_first=cabwar_first,
        save_all_figs=save_all_figs,
        eps_egt=eps_egt,
        generations=generations,
        resolution=resolution,
        stat=stat,
    )
    if game == "all":
        games = GAMES.keys()
    else:
        games = [game]
    if stat == "all":
        stats = ("mean", "median")
    else:
        stats = [stat]
    runs = []
    base.mkdir(exist_ok=True, parents=True)
    for s in stats:
        for game in games:
            params["game"] = game
            params["stat"] = s
            if dataset == "all":
                for d in VALID_DATASETS_ORDERED:
                    runs.append(dict(dataset=d, **params))  # type: ignore
            else:
                runs.append(dict(dataset=dataset, **params))  # type: ignore

    if dry_run:
        print(runs)
        exit(0)
    if max_cores < 0:
        print("[blue]Running Serially[/blue]")
        for info in track(runs, total=len(runs), description="Running Serially"):
            try:
                process(**info)
            except Exception as e:
                print(f"[red]ERROR[/red] Failed to run {e}")
                if raise_exceptions:
                    raise e
                continue
    else:
        futures = []
        cpus = min(cpu_count(), max_cores) if max_cores else cpu_count()
        with ProcessPoolExecutor(max_workers=cpus) as pool:
            for info in runs:
                futures.append(pool.submit(process, **info))

        for f in track(
            as_completed(futures), total=len(futures), description="Receiving Results"
        ):
            try:
                f.result()
            except Exception as e:
                print(f"[red]ERROR[/red] Failed to run a future {e}")
                if raise_exceptions:
                    raise e
                continue


if __name__ == "__main__":
    typer.run(main)
