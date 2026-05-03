"""Project-side CarRacing CNN baseline runner.

Why a separate runner?
----------------------
ObjectRL's ``base_agent.py`` asserts 1-D Box observations and its
actor/critic ``arch`` callables expect ``dim_state: int``. CarRacing-v3
returns ``(96, 96, 3)`` uint8 image observations. Adding CNN support
to ObjectRL would require modifying ``external/objectrl/`` (forbidden by
project policy). This runner is therefore explicitly project-side and
uses a small SAC-CNN implementation in ``scripts/carracing_cnn.py``.

Outputs are written in the same layout as
``scripts/run_project_objectrl_baseline.py``:

- raw   : ``results/raw/project_baselines/<run_name>/``
- csv   : ``results/processed/project_baselines/<run_name>_eval.csv``
- log   : ``results/logs/project_baselines/<run_name>.stdout.log`` (only if redirected)

Each CSV row has columns:
  ``algorithm, project_env, seed, train_step, eval_episode, eval_return,
   wall_time_seconds, status, observation_mode, model_type``.

Defaults are conservative (dry-run unless ``--run`` is passed; batch
mode requires ``--allow-batch-run``; per-run wall time can be capped
with ``--time-limit-minutes``).
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_BASELINE_DIR = REPO_ROOT / "results" / "raw" / "project_baselines"
PROCESSED_BASELINE_DIR = REPO_ROOT / "results" / "processed" / "project_baselines"
LOGS_BASELINE_DIR = REPO_ROOT / "results" / "logs" / "project_baselines"
for _p in (RAW_BASELINE_DIR, PROCESSED_BASELINE_DIR, LOGS_BASELINE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

PROJECT_ENV = "car_racing_continuous"
SUPPORTED_ALGORITHMS = ("sac", "td3", "ppo")
MODEL_TYPE = {"sac": "cnn_sac", "td3": "cnn_td3", "ppo": "cnn_ppo"}

RUN_CONFIG = {
    "debug":  {"seeds": [0],             "max_steps": 1_000,  "eval_episodes": 1, "eval_frequency": 500,  "warmup_steps": 100},
    "midway": {"seeds": [0, 1, 2, 3, 4], "max_steps": 10_000, "eval_episodes": 3, "eval_frequency": 1_000, "warmup_steps": 500},
    "final":  {"seeds": [0, 1, 2, 3, 4], "max_steps": 200_000, "eval_episodes": 5, "eval_frequency": 10_000, "warmup_steps": 1_000},
}


@dataclass
class RunSpec:
    algorithm: str
    seed: int
    max_steps: int
    warmup_steps: int
    eval_frequency: int
    eval_episodes: int
    run_mode: str

    @property
    def run_name(self) -> str:
        return f"{self.run_mode}_{self.algorithm}_{PROJECT_ENV}_seed{self.seed}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TimeLimitTripped(Exception):
    pass


@contextmanager
def _time_limit(minutes: float | None):
    if minutes is None or minutes <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _TimeLimitTripped(f"time limit of {minutes} min reached")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(max(1, minutes * 60)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def _write_eval_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm", "project_env", "seed", "train_step", "eval_episode",
        "eval_return", "wall_time_seconds", "status",
        "observation_mode", "model_type",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_one(spec: RunSpec, *, dry_run: bool, device: str,
            time_limit_minutes: float | None, verbose: bool = False) -> dict[str, Any]:
    run_dir = RAW_BASELINE_DIR / spec.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    csv_path = PROCESSED_BASELINE_DIR / f"{spec.run_name}_eval.csv"

    base_status: dict[str, Any] = {
        **asdict(spec),
        "run_name": spec.run_name,
        "project_env": PROJECT_ENV,
        "device": device,
        "observation_mode": "cnn_chw_float32_in_0_1_shape_3x96x96",
        "model_type": MODEL_TYPE.get(spec.algorithm, "cnn"),
        "time_limit_minutes": time_limit_minutes,
    }

    if spec.algorithm not in SUPPORTED_ALGORITHMS:
        rec = {**base_status, "status": "not_implemented",
               "message": (f"CarRacing CNN baseline implements {SUPPORTED_ALGORITHMS}; "
                           f"'{spec.algorithm}' image-CNN variant is out of scope for the midway report."),
               "wall_time_seconds": 0, "n_eval_rows": 0}
        _write_status(status_path, rec)
        return rec

    if dry_run:
        rec = {**base_status, "status": "dry_run", "wall_time_seconds": 0, "n_eval_rows": 0}
        _write_status(status_path, rec)
        return rec

    # Real training
    try:
        import torch  # local import so dry-runs don't require torch
        from carracing_cnn import train_sac_cnn, train_td3_cnn, train_ppo_cnn
    except Exception as exc:  # noqa: BLE001
        rec = {**base_status, "status": "import_failed",
               "message": f"{type(exc).__name__}: {exc}",
               "wall_time_seconds": 0, "n_eval_rows": 0}
        _write_status(status_path, rec)
        return rec

    if device == "cuda" and not torch.cuda.is_available():
        rec = {**base_status, "status": "cuda_unavailable",
               "message": "CUDA requested but torch.cuda.is_available() is False.",
               "wall_time_seconds": 0, "n_eval_rows": 0}
        _write_status(status_path, rec)
        return rec

    train_fn = {"sac": train_sac_cnn, "td3": train_td3_cnn, "ppo": train_ppo_cnn}[spec.algorithm]

    torch_device = torch.device(device)
    eval_rows: list[dict[str, Any]] = []
    start = time.time()

    def _on_eval(train_step: int, returns: list[float]) -> None:
        wall = round(time.time() - start, 3)
        for i, r in enumerate(returns):
            eval_rows.append({
                "algorithm": spec.algorithm, "project_env": PROJECT_ENV,
                "seed": spec.seed, "train_step": train_step,
                "eval_episode": i, "eval_return": float(r),
                "wall_time_seconds": wall, "status": "completed",
                "observation_mode": base_status["observation_mode"],
                "model_type": MODEL_TYPE.get(spec.algorithm, "cnn"),
            })
        if verbose:
            print(f"    eval@{train_step}: returns={returns}, wall={wall}s")

    summary: dict[str, Any] = {}
    status_label = "completed"
    message = ""
    try:
        with _time_limit(time_limit_minutes):
            summary = train_fn(
                seed=spec.seed,
                max_steps=spec.max_steps,
                warmup_steps=spec.warmup_steps,
                eval_frequency=spec.eval_frequency,
                eval_episodes=spec.eval_episodes,
                device=torch_device,
                buffer_capacity=min(10_000, max(spec.max_steps, 1_000)),
                learn_frequency=1,
                on_eval=_on_eval,
            )
    except _TimeLimitTripped as exc:
        status_label = "time_limit"
        message = str(exc)
    except NotImplementedError as exc:
        status_label = "not_implemented"
        message = str(exc)
    except Exception as exc:  # noqa: BLE001
        status_label = "failed"
        message = f"{type(exc).__name__}: {exc}"
        traceback_str = traceback.format_exc()
        (run_dir / "traceback.txt").write_text(traceback_str, encoding="utf-8")

    wall_time = round(time.time() - start, 3)
    # Tag any partial eval rows with the final status if not already completed
    if status_label != "completed":
        for row in eval_rows:
            row["status"] = status_label

    if eval_rows:
        _write_eval_csv(csv_path, eval_rows)

    rec = {
        **base_status,
        "status": status_label,
        "message": message,
        "wall_time_seconds": wall_time,
        "n_eval_rows": len(eval_rows),
        "summary": summary,
        "csv_path": str(csv_path) if eval_rows else "",
    }
    _write_status(status_path, rec)
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=list(SUPPORTED_ALGORITHMS) + ["all"],
                   default="sac", help="PPO, SAC, and TD3 image-CNN baselines are all implemented.")
    p.add_argument("--seed", type=int, default=None,
                   help="If omitted: seed 0 (no --mode) or the seed list from --mode preset.")
    p.add_argument("--mode", choices=list(RUN_CONFIG), default=None,
                   help="Preset for max_steps/warmup/eval/seeds.")
    p.add_argument("--max-steps", type=int, default=1_000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-frequency", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--time-limit-minutes", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--allow-batch-run", action="store_true",
                   help="Required to run more than one real experiment in a single invocation.")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="Default. Print specs without launching training.")
    g.add_argument("--run", dest="dry_run", action="store_false",
                   help="Actually launch real training.")
    return p


def _specs_from_args(args: argparse.Namespace) -> list[RunSpec]:
    if args.algorithm == "all":
        algos = list(SUPPORTED_ALGORITHMS)
    else:
        algos = [args.algorithm]

    if args.mode is not None:
        cfg = RUN_CONFIG[args.mode]
        seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
        return [RunSpec(algorithm=a, seed=s,
                        max_steps=cfg["max_steps"],
                        warmup_steps=cfg["warmup_steps"],
                        eval_frequency=cfg["eval_frequency"],
                        eval_episodes=cfg["eval_episodes"],
                        run_mode=args.mode)
                for a in algos for s in seeds]

    seed = args.seed if args.seed is not None else 0
    return [RunSpec(algorithm=a, seed=seed,
                    max_steps=args.max_steps,
                    warmup_steps=args.warmup_steps,
                    eval_frequency=args.eval_frequency,
                    eval_episodes=args.eval_episodes,
                    run_mode="debug")
            for a in algos]


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    args = _build_parser().parse_args(argv)
    specs = _specs_from_args(args)

    print(f"CarRacing CNN runner | mode={'dry-run' if args.dry_run else 'REAL TRAINING'} | runs={len(specs)}")
    if not args.dry_run and len(specs) > 1 and not args.allow_batch_run:
        print(f"Refusing to run {len(specs)} real experiments without --allow-batch-run.",
              file=sys.stderr)
        return 2
    if not args.dry_run and len(specs) > 1:
        print(f"!!! Batch real training enabled: {len(specs)} runs will be executed.")

    results: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        try:
            rec = run_one(spec, dry_run=args.dry_run, device=args.device,
                          time_limit_minutes=args.time_limit_minutes, verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001
            rec = {**asdict(spec), "run_name": spec.run_name,
                   "project_env": PROJECT_ENV, "status": "failed_outside_run_one",
                   "message": f"{type(exc).__name__}: {exc}",
                   "wall_time_seconds": 0, "n_eval_rows": 0}
        results.append(rec)
        print(f"  [{idx}/{len(specs)}] [{rec['status']}] {rec['run_name']} "
              f"wall={rec.get('wall_time_seconds', 0)}s eval_rows={rec.get('n_eval_rows', 0)}")
        if rec.get("message"):
            print(f"      {rec['message']}")

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("\nBatch summary:")
    print(f"  total runs : {len(results)}")
    print(f"  status breakdown: {by_status}")
    print(f"  raw output dir : {RAW_BASELINE_DIR}")
    print(f"  csv output dir : {PROCESSED_BASELINE_DIR}")

    ok = {"dry_run", "completed", "not_implemented", "time_limit"}
    return 0 if all(r["status"] in ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
