"""Project-side Gymnasium registration for the official DM887 environments.

This registers custom Gymnasium IDs that delegate to
``scripts.project_envs.make_project_env``:

- ``DM887/CarRacingContinuous-v0``
- ``DM887/CartpoleSwingup-v0``
- ``DM887/AcrobotSwingup-v0``

Importing this module is sufficient to register the IDs; calling
``register_project_envs()`` is idempotent.

Note on ObjectRL CLI compatibility: ObjectRL's ``utils/make_env.py`` rejects
any environment name not present in its three internal whitelists, so these
custom Gymnasium IDs CANNOT be passed to ObjectRL's CLI as ``--env.name``
without modifying ``external/objectrl``. They are still useful for
``gymnasium.make(...)`` and for the project-side ObjectRL runner in
``scripts/run_project_objectrl_baseline.py``.
"""

from __future__ import annotations

from typing import Any

# Make sibling module importable when this file is run as a script as well.
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from project_envs import make_project_env  # noqa: E402

PROJECT_ENV_ID_MAP: dict[str, str] = {
    "DM887/CarRacingContinuous-v0": "car_racing_continuous",
    "DM887/CartpoleSwingup-v0": "cartpole_swingup",
    "DM887/AcrobotSwingup-v0": "acrobot_swingup",
}

_REGISTERED = False


def _entry_point_factory(short_name: str):
    def _entry_point(**kwargs: Any):
        seed = kwargs.pop("seed", None)
        # Any additional kwargs are ignored; make_project_env is intentionally
        # narrow and the per-env adapters do not accept extra arguments.
        return make_project_env(short_name, seed=seed)

    return _entry_point


def register_project_envs() -> list[str]:
    """Register the DM887 project environment IDs with Gymnasium.

    Safe to call multiple times. Returns the list of IDs successfully
    registered (or already present) on this call.
    """
    global _REGISTERED
    import gymnasium as gym
    from gymnasium.envs.registration import register

    registered: list[str] = []
    for env_id, short_name in PROJECT_ENV_ID_MAP.items():
        if env_id in gym.registry:
            registered.append(env_id)
            continue
        register(
            id=env_id,
            entry_point=_entry_point_factory(short_name),
        )
        registered.append(env_id)
    _REGISTERED = True
    return registered


def smoke_test_registered_envs(seed: int = 0) -> list[dict[str, Any]]:
    """Create, reset, and step each registered DM887/* env via gym.make."""
    import gymnasium as gym
    import numpy as np

    register_project_envs()
    results: list[dict[str, Any]] = []
    for env_id in PROJECT_ENV_ID_MAP:
        entry: dict[str, Any] = {"env_id": env_id, "status": "ok"}
        env = None
        try:
            env = gym.make(env_id)
            obs, _info = env.reset(seed=seed)
            action = env.action_space.sample()
            step_out = env.step(action)
            entry.update(
                {
                    "obs_shape": tuple(np.asarray(obs).shape),
                    "obs_after_step_shape": tuple(np.asarray(step_out[0]).shape),
                    "action_space": repr(env.action_space),
                    "observation_space": repr(env.observation_space),
                    "reward": float(step_out[1]),
                    "terminated": bool(step_out[2]),
                    "truncated": bool(step_out[3]),
                }
            )
            print(
                f"[OK] {env_id}: obs={entry['obs_shape']} "
                f"action_space={entry['action_space']} reward={entry['reward']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {env_id}: {entry['error']}")
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        results.append(entry)
    return results


# Eager registration on import keeps usage simple: ``import register_project_envs``.
register_project_envs()


if __name__ == "__main__":
    smoke_test_registered_envs()
