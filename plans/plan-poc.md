# Plan: DM887 GRPO Midway PoC Notebook

## 0. Objective

Create a reproducible Jupyter notebook:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

The notebook should implement the midway baseline workflow for the DM887 GRPO for Control project.

The PoC must focus on:

- ObjectRL-based PPO baseline,
- ObjectRL-based SAC baseline,
- ObjectRL-based TD3 baseline,
- continuous control environments,
- reproducible seeds,
- evaluation returns,
- learning curve plots,
- exported results for the midway LaTeX report.

This is not a toy RL notebook. It is a project-relevant baseline experiment controller.

---

## 1. Required baseline matrix

### Algorithms

```python
ALGORITHMS = ["ppo", "sac", "td3"]
```

### Environments

Use the project environments, with the instructor clarification that Car Racing should be continuous-action:

```python
ENVIRONMENTS = [
    "car_racing_continuous",
    "cartpole_swingup",
    "acrobot_swingup",
]
```

The notebook must include a mapping from these project-level names to the actual ObjectRL/Gymnasium/DMC environment identifiers.

Example placeholder:

```python
OBJECTRL_ENV_NAMES = {
    "car_racing_continuous": "TODO_FIND_OBJECTRL_OR_GYM_NAME",
    "cartpole_swingup": "TODO_FIND_OBJECTRL_OR_DMC_NAME",
    "acrobot_swingup": "TODO_FIND_OBJECTRL_OR_DMC_NAME",
}
```

### Seeds

```python
SEEDS = [0, 1, 2, 3, 4]
```

If the full matrix cannot finish before the midway deadline, the notebook should support a reduced debug mode:

```python
DEBUG_SEEDS = [0]
DEBUG_MAX_STEPS = 1_000
MIDWAY_MAX_STEPS = 20_000
FINAL_MAX_STEPS = 500_000
```

---

## 2. Notebook design principle

The notebook should be an **experiment controller**, not a full reimplementation of PPO/SAC/TD3.

It should:

1. define the experiment matrix,
2. build ObjectRL commands/configurations,
3. run baseline experiments,
4. collect logs,
5. process evaluation returns,
6. plot learning curves,
7. export CSV/Parquet results,
8. export figures for the LaTeX report.

ObjectRL should provide the baseline implementations.

---

## 3. Folder assumptions

The notebook assumes this repository layout:

```text
repo/
├── objectrl/
├── notebooks/
│   └── DM887_Project_GRPO_Midway_PoC.ipynb
├── src/
├── configs/
├── results/
│   ├── raw/
│   ├── processed/
│   └── logs/
└── figures/
```

Path setup:

```python
from pathlib import Path

REPO_ROOT = Path.cwd().resolve()
OBJECTRL_DIR = REPO_ROOT / "objectrl"
RESULTS_DIR = REPO_ROOT / "results"
RAW_RESULTS_DIR = RESULTS_DIR / "raw"
PROCESSED_RESULTS_DIR = RESULTS_DIR / "processed"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = REPO_ROOT / "figures"
```

The notebook should create missing directories automatically.

---

## 4. Notebook sections

The notebook should contain the following sections.

### 1. Project context

Markdown cell explaining:

- project goal,
- midway goal,
- use of ObjectRL,
- continuous Car Racing clarification,
- algorithms and environments.

### 2. Imports and paths

Code cell:

```python
from pathlib import Path
import subprocess
import itertools
import json
import time
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

No seaborn dependency.

### 3. Experiment configuration

Define:

```python
ALGORITHMS = ["ppo", "sac", "td3"]
PROJECT_ENVIRONMENTS = ["car_racing_continuous", "cartpole_swingup", "acrobot_swingup"]
SEEDS = [0, 1, 2, 3, 4]
```

Also define runtime modes:

```python
RUN_MODE = "debug"  # debug | midway | final
```

```python
RUN_CONFIG = {
    "debug": {"seeds": [0], "max_steps": 1_000, "eval_episodes": 1},
    "midway": {"seeds": [0, 1, 2, 3, 4], "max_steps": 20_000, "eval_episodes": 3},
    "final": {"seeds": [0, 1, 2, 3, 4], "max_steps": 500_000, "eval_episodes": 5},
}
```

### 4. Inspect ObjectRL configs

Include cells that help discover correct ObjectRL environment and seed names:

```python
def grep_objectrl(pattern: str, max_lines: int = 100):
    cmd = ["grep", "-R", pattern, "-n", str(OBJECTRL_DIR)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stdout.splitlines()[:max_lines]
    return "\n".join(lines)
```

Then run:

```python
print(grep_objectrl("env.name"))
print(grep_objectrl("seed"))
print(grep_objectrl("car\|cartpole\|acrobot\|dmc\|box2d"))
```

The final notebook should replace placeholders with discovered names.

### 5. ObjectRL command builder

Implement:

```python
def build_objectrl_command(algorithm, project_env_name, seed, max_steps, eval_episodes):
    objectrl_env_name = OBJECTRL_ENV_NAMES[project_env_name]
    run_name = f"{algorithm}_{project_env_name}_seed{seed}"

    cmd = [
        "python",
        "objectrl/main.py",
        "--model.name", algorithm,
        "--env.name", objectrl_env_name,
        "--training.max_steps", str(max_steps),
        "--training.eval_episodes", str(eval_episodes),
        "--seed", str(seed),
    ]

    return cmd, run_name
```

Important: if ObjectRL uses different CLI keys, update this function after inspecting ObjectRL configs.

### 6. Run single experiment

Implement:

```python
def run_experiment(algorithm, project_env_name, seed, max_steps, eval_episodes, dry_run=False):
    cmd, run_name = build_objectrl_command(
        algorithm=algorithm,
        project_env_name=project_env_name,
        seed=seed,
        max_steps=max_steps,
        eval_episodes=eval_episodes,
    )

    run_dir = RAW_RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = LOGS_DIR / f"{run_name}.log"
    status_file = run_dir / "status.json"

    status = {
        "algorithm": algorithm,
        "environment": project_env_name,
        "seed": seed,
        "max_steps": max_steps,
        "eval_episodes": eval_episodes,
        "command": " ".join(cmd),
        "run_name": run_name,
        "log_file": str(log_file),
        "status": "not_started",
    }

    if dry_run:
        status["status"] = "dry_run"
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        return status

    start = time.time()
    with open(log_file, "w") as f:
        process = subprocess.run(
            cmd,
            cwd=OBJECTRL_DIR,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    end = time.time()

    status["return_code"] = process.returncode
    status["duration_seconds"] = end - start
    status["status"] = "completed" if process.returncode == 0 else "failed"

    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)

    return status
```

### 7. Run order

Run easier/less expensive configurations before expensive ones.

Recommended order:

```python
RUN_ORDER = [
    ("sac", "cartpole_swingup"),
    ("td3", "cartpole_swingup"),
    ("ppo", "cartpole_swingup"),
    ("sac", "acrobot_swingup"),
    ("td3", "acrobot_swingup"),
    ("ppo", "acrobot_swingup"),
    ("sac", "car_racing_continuous"),
    ("td3", "car_racing_continuous"),
    ("ppo", "car_racing_continuous"),
]
```

Rationale:

- SAC and TD3 are natural continuous-control baselines.
- PPO is included but isolated because it may be hyperparameter-sensitive.
- Car Racing may be slower and should not block earlier DMC results.

### 8. Full baseline execution

Implement:

```python
cfg = RUN_CONFIG[RUN_MODE]
all_statuses = []

for algorithm, env_name in RUN_ORDER:
    for seed in cfg["seeds"]:
        status = run_experiment(
            algorithm=algorithm,
            project_env_name=env_name,
            seed=seed,
            max_steps=cfg["max_steps"],
            eval_episodes=cfg["eval_episodes"],
            dry_run=False,
        )
        all_statuses.append(status)

status_df = pd.DataFrame(all_statuses)
status_df.to_csv(PROCESSED_RESULTS_DIR / f"status_{RUN_MODE}.csv", index=False)
status_df
```

### 9. Extract evaluation returns

Because ObjectRL logging format may vary, implement a flexible parser.

Parser requirements:

- read each `.log` file,
- look for lines containing evaluation return,
- extract training step and return,
- save a tidy dataframe:

```text
algorithm,environment,seed,step,evaluation_return
```

Skeleton:

```python
def parse_log_file(log_file):
    rows = []
    text = Path(log_file).read_text(errors="ignore")

    # TODO: adapt regex to actual ObjectRL logs
    patterns = [
        r"step[:=]\s*(\d+).*eval.*return[:=]\s*(-?\d+(?:\.\d+)?)",
        r"eval.*step[:=]\s*(\d+).*return[:=]\s*(-?\d+(?:\.\d+)?)",
    ]

    for line in text.splitlines():
        for pattern in patterns:
            match = re.search(pattern, line, flags=re.IGNORECASE)
            if match:
                step = int(match.group(1))
                ret = float(match.group(2))
                rows.append({"step": step, "evaluation_return": ret})
                break
    return rows
```

If ObjectRL writes CSV/TensorBoard/W&B logs instead, replace parser with direct file reader.

### 10. Aggregate results

Implement:

```python
results_df = ...
summary_df = (
    results_df
    .groupby(["algorithm", "environment", "step"], as_index=False)
    .agg(
        mean_return=("evaluation_return", "mean"),
        std_return=("evaluation_return", "std"),
        n_seeds=("seed", "nunique"),
    )
)

results_df.to_csv(PROCESSED_RESULTS_DIR / f"evaluation_returns_{RUN_MODE}.csv", index=False)
summary_df.to_csv(PROCESSED_RESULTS_DIR / f"evaluation_summary_{RUN_MODE}.csv", index=False)
```

### 11. Plot learning curves

Required plot style:

- x-axis: training steps before evaluation,
- y-axis: undiscounted evaluation episode return,
- one subplot per environment,
- one line per algorithm,
- show mean across seeds,
- optionally show shaded ±1 standard deviation if enough seeds exist.

Use matplotlib.

```python
def plot_learning_curves(summary_df, output_path):
    environments = summary_df["environment"].unique()
    fig, axes = plt.subplots(1, len(environments), figsize=(5 * len(environments), 4), sharey=False)

    if len(environments) == 1:
        axes = [axes]

    for ax, env_name in zip(axes, environments):
        env_df = summary_df[summary_df["environment"] == env_name]
        for algorithm in sorted(env_df["algorithm"].unique()):
            alg_df = env_df[env_df["algorithm"] == algorithm].sort_values("step")
            ax.plot(alg_df["step"], alg_df["mean_return"], label=algorithm.upper())

            if "std_return" in alg_df and alg_df["std_return"].notna().any():
                lower = alg_df["mean_return"] - alg_df["std_return"].fillna(0)
                upper = alg_df["mean_return"] + alg_df["std_return"].fillna(0)
                ax.fill_between(alg_df["step"], lower, upper, alpha=0.2)

        ax.set_title(env_name.replace("_", " "))
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Evaluation episode return")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig
```

Export:

```python
plot_learning_curves(summary_df, FIGURES_DIR / "midway_baselines.pdf")
plot_learning_curves(summary_df, FIGURES_DIR / "midway_baselines.png")
```

### 12. Report notes cell

At the end of the notebook, generate Markdown text for the report:

```python
completed = status_df[status_df["status"] == "completed"]
failed = status_df[status_df["status"] == "failed"]

print("Completed runs:")
print(completed[["algorithm", "environment", "seed"]].to_string(index=False))

print("Failed runs:")
print(failed[["algorithm", "environment", "seed", "log_file"]].to_string(index=False))
```

---

## 5. Prompt for GitHub Copilot CLI

Use this prompt from the repo root:

```text
You are helping me build a DM887 Reinforcement Learning midway PoC repository. Read project.md and plan-poc.md. Create a Jupyter notebook notebooks/DM887_Project_GRPO_Midway_PoC.ipynb that runs ObjectRL PPO, SAC, and TD3 baselines for the project environments. The notebook must define the experiment matrix, inspect ObjectRL configs to find environment names and seed/config arguments, build commands, run experiments, collect logs, parse evaluation returns, save CSV summaries, and export learning-curve figures to figures/. Do not implement PPO/SAC/TD3 from scratch. Use ObjectRL. Make the notebook robust: include debug/midway/final modes, dry_run support, and clear TODOs where ObjectRL config names must be verified. Also create any helper Python files in src/ if useful.
```

---

## 6. Prompt for Claude Code CLI

```text
Read project.md and plan-poc.md. I need a project-relevant PoC, not a toy notebook. Generate a complete Jupyter notebook for the DM887 GRPO midway report. It should orchestrate ObjectRL baseline runs for PPO, SAC, and TD3 on continuous Car Racing, cartpole-swingup-v0, and acrobot-swingup-v0. The notebook should be designed for today’s deadline: debug mode first, midway mode second, final mode later. Include robust filesystem paths, command generation, status JSON files, log capture, evaluation-return extraction, CSV export, and matplotlib learning curves. Avoid reimplementing RL algorithms. Preserve a clean structure and include markdown explanations suitable for transferring into a LaTeX report.
```

---

## 7. Prompt for ChatGPT Codex

```text
Use project.md and plan-poc.md as the specification. Build the notebook notebooks/DM887_Project_GRPO_Midway_PoC.ipynb and optional src helpers for a DM887 RL project. The notebook must use ObjectRL’s existing implementations of PPO, SAC, and TD3. It must support a baseline matrix over three environments and five seeds, with debug and midway run modes. It should inspect ObjectRL to discover config names, build CLI commands, run them with subprocess, log results, parse evaluation returns, aggregate by algorithm/environment/step, and export figures/midway_baselines.pdf and figures/midway_baselines.png. Do not use seaborn. Make it ready for VS Code/Jupyter.
```

---

## 8. Manual merge strategy for the three AI-generated candidates

After generating three candidate notebooks:

1. Keep the cleanest folder/path setup.
2. Keep the most robust ObjectRL config discovery code.
3. Keep the safest subprocess runner.
4. Keep the best parser but verify it against actual ObjectRL logs.
5. Keep one plotting implementation.
6. Remove duplicate/redundant cells.
7. Ensure all paths are relative to repo root.
8. Run the notebook from top to bottom in debug mode.
9. Commit the working version.

Suggested commit:

```bash
git add .
git commit -m "Add midway ObjectRL baseline PoC notebook"
```

---

## 9. Minimum acceptable output today

By the midway deadline, the notebook should ideally produce:

```text
results/processed/status_midway.csv
results/processed/evaluation_returns_midway.csv
results/processed/evaluation_summary_midway.csv
figures/midway_baselines.pdf
figures/midway_baselines.png
```

If not all runs finish, still export:

```text
results/processed/status_debug.csv
figures/midway_baselines_partial.pdf
```

and clearly label the report as interim/partial.

---

## 10. Common failure modes and recovery

### ObjectRL command-line key mismatch

Action:

- inspect config files,
- grep for key names,
- update `build_objectrl_command`.

### PPO runs but does not improve

Action:

- do not block the report,
- record hyperparameters,
- mark as pilot result,
- mention that PPO is sensitive and full tuning belongs to final project.

### Car Racing too slow

Action:

- run shorter training budget,
- record as pilot,
- prioritize DMC environments for completed plots.

### No parser matches logs

Action:

- inspect actual log file,
- adapt regex,
- if TensorBoard logs are produced, read event files or export manually.

### Environment missing in ObjectRL

Action:

- use ObjectRL algorithms and create Gymnasium/DMC-compatible environment wrappers,
- document wrapper design in methodology.

---

## 11. Final notebook quality checklist

- [ ] It runs top-to-bottom in debug mode.
- [ ] It does not hard-code absolute user-specific paths unless clearly marked.
- [ ] It logs failed runs instead of crashing the whole matrix.
- [ ] It saves machine-readable outputs.
- [ ] It exports report-ready figures.
- [ ] It documents all seeds.
- [ ] It includes PPO, SAC, and TD3.
- [ ] It includes all three target environments.
- [ ] It clearly distinguishes debug, midway, and final run budgets.
