#!/usr/bin/env python3

from pathlib import Path
from statistics import mean

import pandas as pd
import typer
from negmas.helpers.inout import load
from rich import print
from rich.progress import track
from utils import get_dirs, unpack_


def main(
    base: Path,
    dst: Path = None,  # type: ignore
    verbose: bool = False,
):
    if dst is None:
        dst = base / "allstats.csv"
    domains = get_dirs(base)
    records = []
    for domain in track(domains):
        if verbose:
            print(f"Processing {domain}")
        stats_dir = domain / "stats"
        if not stats_dir.exists():
            print(f"[red]{domain} has no stats [/domain]")
        basic = load(stats_dir / "basic_stats.json")
        for f in stats_dir.glob("*.json"):
            if f.name == "basic_stats.json":
                continue
            if f.stem == "_base":
                perm, f0, f1 = 0, -1, -1
            else:
                perm, f0, f1 = unpack_(f.stem)
            stats: dict
            stats = load(f)
            record = {k: v for k, v in basic.items() if k != "path"}
            assert abs(stats.get("f0", f0) - f0) < 1e-4
            assert abs(stats.get("f1", f1) - f1) < 1e-4
            assert abs(stats.get("perm_index", perm) - perm) < 1e-4
            record.update(dict(f0=f0, f1=f1, perm_index=perm))
            for c in ("cardinal", "ordinal", "perm"):
                stats.pop(c, None)
            record.update(stats.pop("cardinal_reserved_dists", dict()))
            record.update(stats.pop("ordinal_reserved_dists", dict()))
            record.update(stats.pop("cardinal_reserved_optim", dict()))
            record.update(stats.pop("ordinal_reserved_optim", dict()))
            for name, key in (
                ("reserved", "reserved_values"),
                ("max", "maxs"),
                ("min", "mins"),
                ("ordinal_reserved", "ordinal_reserved_values"),
                ("ordinal_max", "ordinal_ordinal_maxs"),
                ("ordinal_min", "ordinal_ordinal_mins"),
            ):
                if key not in stats.keys() and key.startswith("ordinal_ordinal"):
                    key = key[len("ordinal_") :]
                record[f"{name}0"] = stats[key][0]
                record[f"{name}1"] = stats[key][1]
                record[f"{name}NonZeroMean"] = mean(stats[key][1:])
            record.update(stats)
            records.append(record)
    pd.DataFrame(records).to_csv(dst, index=False)


if __name__ == "__main__":
    typer.run(main)
