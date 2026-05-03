"""Project-side environment adapters for DM887 GRPO for Control.

Provides ``make_project_env`` which returns a Gymnasium-style environment for
the three official project environments listed in ``DM887_Project.pdf``:

- ``car_racing_continuous`` -> ``gymnasium.make("CarRacing-v3", continuous=True)``
- ``cartpole_swingup``      -> ``dm_control.suite.load("cartpole", "swingup")``
- ``acrobot_swingup``       -> ``dm_control.suite.load("acrobot", "swingup")``

The DM Control environments are wrapped with a small Gymnasium-compatible
adapter that flattens the observation dict into a single 1-D vector and
exposes ``reset(seed=...)`` / ``step(action) -> (obs, reward, terminated,
truncated, info)``.

This module deliberately does not import anything from ``external/objectrl``;
the goal is a project-side bridge that keeps the third-party checkout
unmodified.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

PROJECT_ENV_NAMES = (
    "car_racing_continuous",
    "cartpole_swingup",
    "acrobot_swingup",
)


# ---------------------------------------------------------------------------
# DM Control -> Gymnasium adapter
# ---------------------------------------------------------------------------


def _flatten_dmc_obs(obs: "OrderedDict[str, np.ndarray] | dict[str, np.ndarray]") -> np.ndarray:
    parts = []
    for value in obs.values():
        arr = np.asarray(value, dtype=np.float32).ravel()
        parts.append(arr)
    if not parts:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(parts, axis=0)


try:  # ``gymnasium`` may be unavailable when this module is imported in a
    # tool-only context; defer the actual base class binding to import time.
    import gymnasium as _gym
    _BaseEnv = _gym.Env
except Exception:  # pragma: no cover
    class _BaseEnv:  # type: ignore[no-redef]
        pass


class DMCGymAdapter(_BaseEnv):
    """Minimal Gymnasium-compatible wrapper around a ``dm_control`` Environment.

    - ``reset(seed=None)`` returns ``(obs, info)``.
    - ``step(action)`` returns ``(obs, reward, terminated, truncated, info)``.
    - Observations are flattened into a single ``float32`` vector.
    - ``action_space`` / ``observation_space`` are Gymnasium ``Box`` spaces.
    """

    metadata = {"render_modes": []}

    def __init__(self, domain_name: str, task_name: str, seed: int | None = None) -> None:
        from dm_control import suite  # imported lazily so import errors are explicit
        from gymnasium import spaces

        super().__init__()

        self.domain_name = domain_name
        self.task_name = task_name
        self._seed = 0 if seed is None else int(seed)
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs={"random": self._seed},
        )

        action_spec = self._env.action_spec()
        self.action_space = spaces.Box(
            low=np.asarray(action_spec.minimum, dtype=np.float32),
            high=np.asarray(action_spec.maximum, dtype=np.float32),
            shape=tuple(action_spec.shape),
            dtype=np.float32,
        )

        sample_obs = _flatten_dmc_obs(self._env.reset().observation)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        from dm_control import suite

        if seed is not None and int(seed) != self._seed:
            # dm_control seeds at construction; rebuild for a new seed.
            self._seed = int(seed)
            self._env = suite.load(
                domain_name=self.domain_name,
                task_name=self.task_name,
                task_kwargs={"random": self._seed},
            )
        time_step = self._env.reset()
        obs = _flatten_dmc_obs(time_step.observation)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        time_step = self._env.step(action)
        obs = _flatten_dmc_obs(time_step.observation)
        reward = float(time_step.reward) if time_step.reward is not None else 0.0
        done = bool(time_step.last())
        # dm_control uses discount==0 to signal a true terminal state.
        # discount==1 at the last step indicates a time-limit truncation.
        truncated = done and float(time_step.discount) == 1.0
        terminated = done and not truncated
        info: dict[str, Any] = {}
        if truncated:
            info["TimeLimit.truncated"] = True
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_project_env(name: str, seed: int | None = None):
    """Create one of the official DM887 project environments by short name."""
    if name == "car_racing_continuous":
        try:
            import gymnasium as gym
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "gymnasium is required for 'car_racing_continuous'. "
                "Install it with `pip install gymnasium[box2d]`."
            ) from exc
        try:
            env = gym.make("CarRacing-v3", continuous=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create 'CarRacing-v3'. This usually means the "
                "Gymnasium Box2D extra is not installed. Install it with "
                "`pip install 'gymnasium[box2d]'` (Box2D wheels require "
                "swig and a C++ toolchain)."
            ) from exc
        if seed is not None:
            env.reset(seed=int(seed))
            env.action_space.seed(int(seed))
        return env

    if name == "cartpole_swingup":
        return DMCGymAdapter("cartpole", "swingup", seed=seed)

    if name == "acrobot_swingup":
        return DMCGymAdapter("acrobot", "swingup", seed=seed)

    raise ValueError(
        f"Unknown project environment '{name}'. "
        f"Expected one of {PROJECT_ENV_NAMES}."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def smoke_test_project_envs(seed: int = 0) -> list[dict[str, Any]]:
    """Create, reset, and step each official project environment once.

    Returns a list of result dicts (one per environment) describing the
    outcome. Failures are captured as ``status='error'`` rather than raised so
    that callers can report on partially-working setups.
    """
    results: list[dict[str, Any]] = []
    for name in PROJECT_ENV_NAMES:
        entry: dict[str, Any] = {"name": name, "status": "ok"}
        env = None
        try:
            env = make_project_env(name, seed=seed)
            obs, _info = env.reset(seed=seed)
            action = env.action_space.sample()
            step_out = env.step(action)
            obs_after = step_out[0]
            entry.update(
                {
                    "observation_space": repr(env.observation_space),
                    "action_space": repr(env.action_space),
                    "obs_shape": tuple(np.asarray(obs).shape),
                    "obs_after_step_shape": tuple(np.asarray(obs_after).shape),
                    "reward": float(step_out[1]),
                    "terminated": bool(step_out[2]),
                    "truncated": bool(step_out[3]),
                }
            )
            print(
                f"[OK] {name}: obs={entry['obs_shape']} "
                f"action_space={entry['action_space']} reward={entry['reward']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {name}: {entry['error']}")
        finally:
            if env is not None and hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass
        results.append(entry)
    return results


if __name__ == "__main__":
    smoke_test_project_envs()
