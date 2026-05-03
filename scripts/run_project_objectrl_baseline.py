"""Project-side ObjectRL baseline runner.

Trains PPO/SAC/TD3 (from ObjectRL) on the official DM887 project environments
(from ``scripts/project_envs.py``) without modifying ``external/objectrl`` or
``external/Gymnasium``.

Strategy
--------
ObjectRL's ``utils/make_env.py`` whitelists environment names and so cannot be
used through its CLI for the project envs (verified). The clean project-side
bridge is to monkey-patch the ``make_env`` symbol that
``objectrl.experiments.base_experiment`` already imported, so that
``ControlExperiment(config)`` constructs ``self.env`` and ``self.eval_env``
from ``scripts.project_envs.make_project_env(...)``. Nothing in
``external/objectrl`` is modified; we only re-bind a name in an
already-imported module.

The training/evaluation loop, replay buffer, agent updates and logging are all
ObjectRL's. PPO/SAC/TD3 are NOT reimplemented here.

Defaults are conservative: dry-run, CPU only, max 1k steps. Real training
requires explicit ``--run``.

Outputs
-------
- ``results/raw/project_baselines/<run_name>/``: status JSON + ObjectRL log dir.
- ``results/processed/project_baselines/<run_name>_eval.csv``: parsed eval returns.
- ``results/logs/project_baselines/<run_name>.stdout.log``: textual run log.

CarRacing
---------
``car_racing_continuous`` returns ``(96, 96, 3)`` uint8 image observations.
ObjectRL's default MLP actor/critic require 1-D ``Box`` observations, so this
env is rejected here with a clear message instead of silently failing inside
ObjectRL's network setup. CNN-policy support is left for the final project.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_envs import (  # noqa: E402
    PROJECT_ENV_NAMES,
    make_project_env,
)
from register_project_envs import register_project_envs  # noqa: E402

ALGORITHMS = ("ppo", "sac", "td3")

RUN_CONFIG: dict[str, dict[str, Any]] = {
    "debug": {"seeds": [0], "max_steps": 1_000, "eval_episodes": 1, "eval_frequency": 500, "warmup_steps": 100},
    "midway": {"seeds": [0, 1, 2, 3, 4], "max_steps": 20_000, "eval_episodes": 3, "eval_frequency": 5_000, "warmup_steps": 1_000},
    "final": {"seeds": [0, 1, 2, 3, 4], "max_steps": 500_000, "eval_episodes": 5, "eval_frequency": 20_000, "warmup_steps": 10_000},
}

RAW_BASELINE_DIR = REPO_ROOT / "results" / "raw" / "project_baselines"
PROCESSED_BASELINE_DIR = REPO_ROOT / "results" / "processed" / "project_baselines"
LOGS_BASELINE_DIR = REPO_ROOT / "results" / "logs" / "project_baselines"

for _p in (RAW_BASELINE_DIR, PROCESSED_BASELINE_DIR, LOGS_BASELINE_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Spec / matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    algorithm: str
    project_env: str
    seed: int
    max_steps: int
    eval_episodes: int
    eval_frequency: int
    warmup_steps: int
    learn_frequency: int
    run_mode: str

    @property
    def run_name(self) -> str:
        return f"{self.run_mode}_{self.algorithm}_{self.project_env}_seed{self.seed}"


# PPO is on-policy and ObjectRL asserts learn_frequency > 1 when
# normalize_advantages is enabled. Use a small rollout length for debug runs.
PPO_DEFAULT_LEARN_FREQUENCY = 256
DEFAULT_LEARN_FREQUENCY = 1


def _default_learn_frequency(algorithm: str) -> int:
    return PPO_DEFAULT_LEARN_FREQUENCY if algorithm == "ppo" else DEFAULT_LEARN_FREQUENCY


def build_matrix(run_mode: str) -> list[RunSpec]:
    cfg = RUN_CONFIG[run_mode]
    return [
        RunSpec(
            algorithm=algo,
            project_env=env,
            seed=seed,
            max_steps=cfg["max_steps"],
            eval_episodes=cfg["eval_episodes"],
            eval_frequency=cfg["eval_frequency"],
            warmup_steps=0 if algo == "ppo" else cfg["warmup_steps"],
            learn_frequency=_default_learn_frequency(algo),
            run_mode=run_mode,
        )
        for env in PROJECT_ENV_NAMES
        for algo in ALGORITHMS
        for seed in cfg["seeds"]
    ]


# ---------------------------------------------------------------------------
# ObjectRL inspection / monkey-patch bridge
# ---------------------------------------------------------------------------


def inspect_objectrl() -> dict[str, Any]:
    info: dict[str, Any] = {"importable": False}
    try:
        from objectrl.config.config import MainConfig  # noqa: F401
        from objectrl.experiments.control_experiment import ControlExperiment  # noqa: F401
        from objectrl.models.get_model import get_model  # noqa: F401

        info.update(
            {
                "importable": True,
                "MainConfig": "objectrl.config.config.MainConfig",
                "ControlExperiment": "objectrl.experiments.control_experiment.ControlExperiment",
                "get_model": "objectrl.models.get_model.get_model",
            }
        )
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _wrap_for_objectrl(env, seed: int):
    """Apply the same lightweight wrappers ObjectRL's make_env would apply."""
    import gymnasium as gym
    import numpy as np
    from gymnasium.wrappers import RescaleAction

    if not isinstance(env.action_space, gym.spaces.Discrete):
        env = RescaleAction(env, np.float32(-1.0), np.float32(1.0))

    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def _make_project_env_for_objectrl(
    env_name: str,
    seed: int,
    env_config,  # noqa: ARG001
    eval_env: bool = False,
    num_envs: int = 1,
):
    """Drop-in replacement for ``objectrl.utils.make_env.make_env``."""
    if num_envs > 1:
        raise NotImplementedError(
            "Vectorised eval (parallelize_eval=True) is not implemented in the "
            "project-side runner. Set training.parallelize_eval=False."
        )
    if env_name not in PROJECT_ENV_NAMES:
        raise ValueError(
            f"_make_project_env_for_objectrl received unsupported env name "
            f"{env_name!r}. Expected one of {PROJECT_ENV_NAMES}."
        )
    effective_seed = seed + (100 if eval_env else 0)
    env = make_project_env(env_name, seed=effective_seed)
    return _wrap_for_objectrl(env, effective_seed)


def _patch_objectrl_make_env() -> None:
    """Re-bind ``make_env`` inside ``base_experiment`` to the project factory.

    Idempotent. Does not touch any file under ``external/objectrl``.
    """
    import objectrl.experiments.base_experiment as be

    if getattr(be, "_dm887_patched", False):
        return
    be.make_env = _make_project_env_for_objectrl  # type: ignore[attr-defined]
    be._dm887_patched = True  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Experiment construction
# ---------------------------------------------------------------------------


def _build_main_config(spec: RunSpec, device: str, result_path: Path, verbose: bool):
    from objectrl.config.config import MainConfig

    cfg_dict = {
        "verbose": verbose,
        "progress": False,
        "env": {"name": spec.project_env},
        "training": {
            "max_steps": spec.max_steps,
            "warmup_steps": spec.warmup_steps,
            "learn_frequency": spec.learn_frequency,
            "eval_frequency": spec.eval_frequency,
            "eval_episodes": spec.eval_episodes,
            "parallelize_eval": False,
        },
        "system": {
            "seed": spec.seed,
            "device": device,
            "storing_device": device,
        },
        "logging": {
            "result_path": str(result_path),
            "save_frequency": spec.eval_frequency,
        },
        "model": {"name": spec.algorithm},
    }
    return MainConfig.from_config(cfg_dict)


# ---------------------------------------------------------------------------
# Time-limit guard (POSIX-only, best effort)
# ---------------------------------------------------------------------------


class _TimeLimitTripped(Exception):
    pass


class _TimeLimit:
    def __init__(self, minutes: float | None) -> None:
        self.deadline = None if not minutes else time.time() + minutes * 60.0
        self._previous = None

    def __enter__(self):
        if self.deadline is None:
            return self
        remaining = max(1, int(self.deadline - time.time()))

        def _handler(signum, frame):  # noqa: ANN001
            raise _TimeLimitTripped(f"Time limit ({remaining}s) reached.")

        self._previous = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(remaining)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.deadline is None:
            return False
        signal.alarm(0)
        if self._previous is not None:
            signal.signal(signal.SIGALRM, self._previous)
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


CARRACING_BLOCK_MESSAGE = (
    "car_racing_continuous returns (96,96,3) uint8 image observations. "
    "ObjectRL's default MLP actor/critic require 1-D Box observations. "
    "Add a CNN policy or feature/flatten wrapper before training this env."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _parse_eval_results(run_log_dir: Path) -> list[dict[str, Any]]:
    """Read ObjectRL's ``eval_results.npy`` (a dict step -> reward tensor)."""
    import numpy as np

    eval_npy = run_log_dir / "eval_results.npy"
    if not eval_npy.exists():
        return []
    try:
        data = np.load(eval_npy, allow_pickle=True).item()
    except Exception:  # noqa: BLE001
        return []
    rows: list[dict[str, Any]] = []
    if not isinstance(data, dict):
        return rows
    for step, rewards in data.items():
        if hasattr(rewards, "detach"):
            arr = rewards.detach().cpu().numpy()
        else:
            arr = np.asarray(rewards)
        arr = np.asarray(arr, dtype=float).ravel()
        for ep_idx, value in enumerate(arr):
            rows.append({"train_step": int(step), "eval_episode": int(ep_idx), "eval_return": float(value)})
    return rows


def _write_eval_csv(rows: list[dict[str, Any]], spec: RunSpec, csv_path: Path, status: str, wall_time: float) -> None:
    import csv

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm", "project_env", "seed", "train_step",
                "eval_episode", "eval_return", "wall_time_seconds", "status",
            ],
        )
        writer.writeheader()
        if not rows:
            writer.writerow({
                "algorithm": spec.algorithm, "project_env": spec.project_env, "seed": spec.seed,
                "train_step": "", "eval_episode": "", "eval_return": "",
                "wall_time_seconds": round(wall_time, 3), "status": status,
            })
            return
        for row in rows:
            writer.writerow({
                "algorithm": spec.algorithm, "project_env": spec.project_env, "seed": spec.seed,
                **row,
                "wall_time_seconds": round(wall_time, 3), "status": status,
            })


def run_one(
    spec: RunSpec,
    *,
    dry_run: bool,
    device: str = "cpu",
    time_limit_minutes: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    register_project_envs()  # harmless; keeps DM887/* IDs available too.

    run_dir = RAW_BASELINE_DIR / spec.run_name
    status_path = run_dir / "status.json"
    log_path = LOGS_BASELINE_DIR / f"{spec.run_name}.stdout.log"
    eval_csv_path = PROCESSED_BASELINE_DIR / f"{spec.run_name}_eval.csv"

    record: dict[str, Any] = {
        "run_name": spec.run_name,
        "algorithm": spec.algorithm,
        "project_env": spec.project_env,
        "seed": spec.seed,
        "run_mode": spec.run_mode,
        "max_steps": spec.max_steps,
        "warmup_steps": spec.warmup_steps,
        "learn_frequency": spec.learn_frequency,
        "eval_frequency": spec.eval_frequency,
        "eval_episodes": spec.eval_episodes,
        "device": device,
        "dry_run": dry_run,
        "objectrl_log_dir": str(run_dir),
        "status_file": str(status_path),
        "subprocess_log_file": str(log_path),
        "eval_csv": str(eval_csv_path),
        "created_at": _utc_now(),
        "status": "not_started",
    }

    if spec.project_env == "car_racing_continuous":
        record["status"] = "blocked_image_obs"
        record["message"] = CARRACING_BLOCK_MESSAGE
        _write_status(status_path, record)
        _write_eval_csv([], spec, eval_csv_path, record["status"], 0.0)
        return record

    if spec.algorithm == "ppo":
        if spec.warmup_steps != 0:
            record["status"] = "invalid_config"
            record["message"] = (
                "PPO is on-policy and does not use a warmup buffer; pass "
                "--warmup-steps 0."
            )
            _write_status(status_path, record)
            _write_eval_csv([], spec, eval_csv_path, record["status"], 0.0)
            return record
        if spec.learn_frequency <= 1:
            record["status"] = "invalid_config"
            record["message"] = (
                "PPO requires training.learn_frequency > 1 when "
                "normalize_advantages is enabled (ObjectRL assertion). "
                f"Got {spec.learn_frequency}; pass e.g. --learn-frequency 256."
            )
            _write_status(status_path, record)
            _write_eval_csv([], spec, eval_csv_path, record["status"], 0.0)
            return record

    if dry_run:
        record["status"] = "dry_run"
        _write_status(status_path, record)
        _write_eval_csv([], spec, eval_csv_path, record["status"], 0.0)
        return record

    inspection = inspect_objectrl()
    if not inspection.get("importable"):
        record["status"] = "objectrl_import_failed"
        record["message"] = inspection.get("error")
        _write_status(status_path, record)
        _write_eval_csv([], spec, eval_csv_path, record["status"], 0.0)
        return record

    _patch_objectrl_make_env()

    from objectrl.experiments.control_experiment import ControlExperiment

    config = _build_main_config(spec, device=device, result_path=run_dir, verbose=verbose)

    record["started_at"] = _utc_now()
    start = time.time()
    try:
        with _TimeLimit(time_limit_minutes), log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"[{_utc_now()}] starting {spec.run_name}\n")
            logf.flush()
            exp = ControlExperiment(config)
            logf.write(f"[{_utc_now()}] experiment constructed; entering train()\n")
            logf.flush()
            exp.train()
            logf.write(f"[{_utc_now()}] train() returned cleanly\n")
            record["status"] = "completed"
    except _TimeLimitTripped as exc:
        record["status"] = "time_limit"
        record["message"] = str(exc)
    except NotImplementedError as exc:
        record["status"] = "not_implemented"
        record["message"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    except Exception as exc:  # noqa: BLE001
        record["status"] = "failed"
        record["message"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        wall = time.time() - start
        record["wall_time_seconds"] = round(wall, 3)
        record["finished_at"] = _utc_now()

    eval_rows: list[dict[str, Any]] = []
    for npy in run_dir.rglob("eval_results.npy"):
        eval_rows.extend(_parse_eval_results(npy.parent))

    _write_eval_csv(eval_rows, spec, eval_csv_path, record["status"], record.get("wall_time_seconds", 0.0))
    record["n_eval_rows"] = len(eval_rows)
    _write_status(status_path, record)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algorithm", choices=list(ALGORITHMS) + ["all"], default="sac")
    p.add_argument("--project-env", choices=list(PROJECT_ENV_NAMES) + ["all"], default="cartpole_swingup")
    p.add_argument("--seed", type=int, default=None,
                   help="If omitted: defaults to seed 0 in debug mode, or the seed list of the chosen --mode preset.")
    p.add_argument("--mode", choices=list(RUN_CONFIG), default=None,
                   help="If set, applies the named preset for --max-steps/--warmup-steps/--eval-* and the default seed list.")
    p.add_argument("--max-steps", type=int, default=1_000)
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="Forced to 0 for PPO (on-policy).")
    p.add_argument("--learn-frequency", type=int, default=None,
                   help=("ObjectRL training.learn_frequency. Default: 1 for off-policy "
                         f"(SAC/TD3) and {PPO_DEFAULT_LEARN_FREQUENCY} for PPO."))
    p.add_argument("--eval-frequency", type=int, default=500)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--time-limit-minutes", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--allow-batch-run", action="store_true",
                   help="Required to run more than one real experiment in a single invocation.")

    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="(default) only describe the run, do not train.")
    g.add_argument("--run", dest="dry_run", action="store_false",
                   help="Actually perform real training. Required to leave dry-run mode.")
    p.set_defaults(dry_run=True)
    return p


def _specs_from_args(args: argparse.Namespace) -> list[RunSpec]:
    if args.mode is not None:
        cfg = RUN_CONFIG[args.mode]
        seeds = [args.seed] if args.seed is not None else list(cfg["seeds"])
        algos = list(ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
        envs = list(PROJECT_ENV_NAMES) if args.project_env == "all" else [args.project_env]
        specs: list[RunSpec] = []
        for env in envs:
            for algo in algos:
                learn_freq = (
                    args.learn_frequency if args.learn_frequency is not None
                    else _default_learn_frequency(algo)
                )
                warmup = 0 if algo == "ppo" else cfg["warmup_steps"]
                for seed in seeds:
                    specs.append(RunSpec(
                        algorithm=algo,
                        project_env=env,
                        seed=seed,
                        max_steps=cfg["max_steps"],
                        warmup_steps=warmup,
                        learn_frequency=learn_freq,
                        eval_frequency=cfg["eval_frequency"],
                        eval_episodes=cfg["eval_episodes"],
                        run_mode=args.mode,
                    ))
        return specs

    seed = args.seed if args.seed is not None else 0
    algos = list(ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
    envs = list(PROJECT_ENV_NAMES) if args.project_env == "all" else [args.project_env]

    def _spec(a: str, e: str) -> RunSpec:
        learn_freq = args.learn_frequency if args.learn_frequency is not None else _default_learn_frequency(a)
        warmup = 0 if a == "ppo" else args.warmup_steps
        return RunSpec(
            algorithm=a,
            project_env=e,
            seed=seed,
            max_steps=args.max_steps,
            warmup_steps=warmup,
            learn_frequency=learn_freq,
            eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes,
            run_mode="debug",
        )

    return [_spec(a, e) for e in envs for a in algos]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    specs = _specs_from_args(args)

    print(f"Mode: {'dry-run' if args.dry_run else 'REAL TRAINING'}; runs: {len(specs)}")
    if not args.dry_run and len(specs) > 1 and not args.allow_batch_run:
        print(
            f"Refusing to run {len(specs)} real experiments without --allow-batch-run. "
            "Add --allow-batch-run to opt in, or narrow --algorithm/--project-env/--seed, "
            "or use --dry-run for a matrix preview.",
            file=sys.stderr,
        )
        return 2
    if not args.dry_run and len(specs) > 1 and args.allow_batch_run:
        print(f"!!! Batch real training enabled: {len(specs)} runs will be executed.")

    register_project_envs()

    results: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        try:
            rec = run_one(
                spec,
                dry_run=args.dry_run,
                device=args.device,
                time_limit_minutes=args.time_limit_minutes,
                verbose=args.verbose,
            )
        except Exception as exc:  # noqa: BLE001
            rec = {
                "run_name": spec.run_name,
                "algorithm": spec.algorithm,
                "project_env": spec.project_env,
                "seed": spec.seed,
                "status": "failed_outside_run_one",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "wall_time_seconds": 0,
                "n_eval_rows": 0,
            }
            try:
                _write_status(
                    RAW_BASELINE_DIR / spec.run_name / "status.json", rec,
                )
            except Exception:  # noqa: BLE001
                pass
        results.append(rec)
        print(
            f"  [{idx}/{len(specs)}] [{rec['status']}] {rec['run_name']} "
            f"wall={rec.get('wall_time_seconds', 0)}s eval_rows={rec.get('n_eval_rows', 0)}"
        )
        if rec.get("message"):
            print(f"      {rec['message']}")

    # Batch summary
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("\nBatch summary:")
    print(f"  total runs : {len(results)}")
    print(f"  completed  : {by_status.get('completed', 0)}")
    print(f"  failed     : {sum(v for k, v in by_status.items() if k.startswith('failed') or k in {'time_limit', 'objectrl_import_failed', 'invalid_config', 'not_implemented'})}")
    print(f"  blocked    : {by_status.get('blocked_image_obs', 0)}")
    print(f"  dry_run    : {by_status.get('dry_run', 0)}")
    print(f"  status breakdown: {by_status}")
    print(f"  raw output dir : {RAW_BASELINE_DIR}")
    print(f"  csv output dir : {PROCESSED_BASELINE_DIR}")
    print(f"  log output dir : {LOGS_BASELINE_DIR}")

    ok = {"dry_run", "completed", "blocked_image_obs"}
    return 0 if all(r["status"] in ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
