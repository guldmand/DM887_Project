Midway baseline protocol notes

- Runtime mode: debug
- Dry run: True
- Algorithms: PPO, SAC, TD3
- Seeds requested in this mode: [0]
- Max training steps: 1000
- Evaluation episodes: 1
- Evaluation frequency: 500 training steps
- Metric: undiscounted evaluation episode return.
- Implementation source: ObjectRL PPO, SAC, and TD3; no baseline algorithm is reimplemented here.
- Local device policy: ObjectRL CLI runs on CPU with `--system.device=cpu --system.storing_device=cpu`. PyTorch reports MPS available, but ObjectRL CLI rejects `mps`.
- ObjectRL CPU smoke test on `dmc-cheetah-run` works (Section 3b.2; opt-in).
- Official project-environment adapter (`scripts/project_envs.py`) works for all three required envs (`car_racing_continuous`, `cartpole_swingup`, `acrobot_swingup`); `external/objectrl/` and `external/Gymnasium/` are unmodified.
- Project-side runner `scripts/run_project_objectrl_baseline.py` drives ObjectRL's `ControlExperiment` against the project envs via a runtime monkey-patch of `objectrl.experiments.base_experiment.make_env` (Section 3d).
- Small real **debug** training has completed for PPO, SAC, and TD3 on `cartpole_swingup` and `acrobot_swingup` (seed 0, max 1000 steps, eval_episodes=1, CPU). CSVs in `results/processed/project_baselines/`. These are pipeline-verification runs, not midway or final performance claims.
- CarRacing real training is intentionally **deferred**: `(96, 96, 3)` uint8 image observations require a CNN policy or feature/flatten preprocessing before ObjectRL's default MLP actor/critic apply.
- PPO integration constraint: ObjectRL asserts `training.learn_frequency > 1` when `normalize_advantages=True` and uses no warmup. The runner enforces `--warmup-steps 0` and defaults `learn_frequency=256` for PPO automatically.

Environment verification status:
- Car Racing continuous: ObjectRL name `CarRacing-v3`; not listed in local ObjectRL make_env.py mapping; verify continuous=True and image-observation preprocessing for ObjectRL.
- cartpole-swingup-v0: ObjectRL name `dmc-cartpole-swingup`; not listed in local ObjectRL make_env.py mapping; verify DMC name and ObjectRL whitelist/wrapper support.
- acrobot-swingup-v0: ObjectRL name `dmc-acrobot-swingup`; not listed in local ObjectRL make_env.py mapping; verify DMC name and ObjectRL whitelist/wrapper support.

Run status counts:
- PPO / acrobot-swingup-v0 / dry_run: 1
- PPO / Car Racing continuous / dry_run: 1
- PPO / cartpole-swingup-v0 / dry_run: 1
- SAC / acrobot-swingup-v0 / dry_run: 1
- SAC / Car Racing continuous / dry_run: 1
- SAC / cartpole-swingup-v0 / dry_run: 1
- TD3 / acrobot-swingup-v0 / dry_run: 1
- TD3 / Car Racing continuous / dry_run: 1
- TD3 / cartpole-swingup-v0 / dry_run: 1

Figure caption draft:
Interim baseline learning curves for PPO, SAC, and TD3. The x-axis shows the number of online training steps before evaluation, and the y-axis shows undiscounted evaluation episode return. Curves average over available seeds; shaded regions indicate one standard deviation when multiple seeds are available. Dry-run or incomplete runs should be labelled as pipeline validation rather than final empirical evidence.