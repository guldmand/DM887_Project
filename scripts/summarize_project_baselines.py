"""Summarize project-side baseline CSVs and produce midway plots.

Reads ``results/processed/project_baselines/<prefix>_*_eval.csv``,
aggregates duplicate rows, reports coverage gaps, and writes a summary CSV
plus per-environment matplotlib plots under ``figures/midway/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CSV_DIR = REPO_ROOT / "results" / "processed" / "project_baselines"
FIGURES_MIDWAY_DIR = REPO_ROOT / "figures" / "midway"

EXPECTED_ALGORITHMS = ("ppo", "sac", "td3")
EXPECTED_ENVS = ("cartpole_swingup", "acrobot_swingup", "car_racing_continuous")
EXPECTED_SEEDS = (0, 1, 2, 3, 4)

# PPO, SAC, and TD3 are now implemented for CarRacing CNN.
EXPECTED_COMBINATIONS = {
    (a, e, s)
    for a in EXPECTED_ALGORITHMS
    for e in EXPECTED_ENVS
    for s in EXPECTED_SEEDS
}

ALGO_COLORS = {"ppo": "#4C78A8", "sac": "#F58518", "td3": "#54A24B"}


def _load_csvs(prefix: str) -> tuple[pd.DataFrame, list[Path]]:
    pattern = f"{prefix}_*_eval.csv" if prefix else "*_eval.csv"
    files = sorted(BASELINE_CSV_DIR.glob(pattern))
    if not files:
        return pd.DataFrame(), files
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["source_csv"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True), files


def _report_coverage(df: pd.DataFrame) -> None:
    print("\nCoverage (algorithm, project_env, seed):")
    if df.empty:
        print("  (no rows)")
        return
    have = set(
        df[["algorithm", "project_env", "seed"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    expected = EXPECTED_COMBINATIONS
    print(f"  expected combinations: {len(expected)}")
    print(f"  present  combinations: {len(have & expected)}")
    missing = sorted(expected - have)
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for combo in missing:
            print(f"    - algorithm={combo[0]:<4} env={combo[1]:<18} seed={combo[2]}")
    else:
        print("  all expected combinations present")
    extras = sorted({(a, e, s) for (a, e, s) in have if (a, e, s) not in expected})
    if extras:
        print(f"  Other (unexpected) combinations: {len(extras)}")


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    eval_only = df[df["eval_return"].notna()].copy()
    group_keys = ["algorithm", "project_env", "seed", "train_step", "status"]
    agg = (
        eval_only.groupby(group_keys, as_index=False)
        .agg(eval_return=("eval_return", "mean"),
             n_repeats=("eval_return", "size"))
    )
    return agg


def _plot_env(env_name: str, ax, agg: pd.DataFrame) -> bool:
    sub = agg[agg["project_env"] == env_name]
    if sub.empty:
        ax.set_title(f"{env_name}: no data")
        ax.axis("off")
        return False
    # Average across seeds for the curve, also plot std band when >=2 seeds.
    grouped = (
        sub.groupby(["algorithm", "train_step"], as_index=False)
        .agg(mean_return=("eval_return", "mean"),
             std_return=("eval_return", "std"),
             n_seeds=("seed", "nunique"))
    )
    for algo in EXPECTED_ALGORITHMS:
        a = grouped[grouped["algorithm"] == algo].sort_values("train_step")
        if a.empty:
            continue
        color = ALGO_COLORS.get(algo)
        ax.plot(a["train_step"], a["mean_return"],
                marker="o", linewidth=1.8, color=color, label=algo.upper())
        if (a["n_seeds"] > 1).any():
            std = a["std_return"].fillna(0.0)
            ax.fill_between(
                a["train_step"], a["mean_return"] - std, a["mean_return"] + std,
                color=color, alpha=0.15, linewidth=0,
            )
    ax.set_title(env_name)
    ax.set_xlabel("training steps before evaluation")
    ax.set_ylabel("undiscounted eval episode return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    return True


def _save_plots(agg: pd.DataFrame, prefix: str) -> list[Path]:
    FIGURES_MIDWAY_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for env in EXPECTED_ENVS:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_env(env, ax, agg)
        out = FIGURES_MIDWAY_DIR / f"{prefix}_{env}_baselines.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        saved.append(out)

    fig, axes = plt.subplots(1, len(EXPECTED_ENVS), figsize=(5.5 * len(EXPECTED_ENVS), 4), sharey=False)
    for ax, env in zip(axes, EXPECTED_ENVS):
        _plot_env(env, ax, agg)
    fig.suptitle(f"{prefix.capitalize()} vector-env baselines (PPO/SAC/TD3, seeds 0-4)", y=1.02)
    out = FIGURES_MIDWAY_DIR / f"{prefix}_vector_env_baselines.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(out)

    return saved


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="midway",
                   help="Filename prefix to filter CSVs (e.g. 'midway' or 'debug'). Default: midway")
    p.add_argument("--all", action="store_true",
                   help="Ignore --prefix and load every *_eval.csv.")
    args = p.parse_args(argv)

    prefix = "" if args.all else args.prefix
    df, files = _load_csvs(prefix)

    print(f"CSV directory : {BASELINE_CSV_DIR}")
    print(f"Filter prefix : {prefix or '(all)'}")
    print(f"Files matched : {len(files)}")
    for f in files:
        print(f"  - {f.name}")
    print(f"Total rows    : {len(df)}")

    if df.empty:
        print("No CSVs to summarize.", file=sys.stderr)
        return 1

    _report_coverage(df)

    agg = _aggregate(df)
    print(f"\nAggregated rows (group by algorithm,env,seed,train_step,status, mean eval_return): {len(agg)}")

    out_label = args.prefix if not args.all else "all"
    summary_csv = BASELINE_CSV_DIR / f"{out_label}_vector_summary.csv"
    agg.to_csv(summary_csv, index=False)
    print(f"Wrote summary CSV: {summary_csv}")

    saved = _save_plots(agg, out_label)
    print("Wrote plots:")
    for s in saved:
        print(f"  - {s.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
