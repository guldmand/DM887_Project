# Agent Instructions

This repository is for the DM887 Reinforcement Learning project:

> **GRPO for Control**

The project investigates whether a variant of **Group Relative Policy Optimization (GRPO)** can be adapted to continuous-control reinforcement learning tasks and compared against PPO, SAC, and TD3 baselines.

These instructions are intended for AI coding agents working in this repository.

---

## 1. Read these files first

Before making changes, read the following files in this order:

1. `README.md`
2. `docs/project.md`
3. `docs/project-structure.md`
4. `plans/plan-poc.md`
5. `plans/plan-midway-rapport-latex.md`
6. `docs/references/plan-references.md`
7. `docs/scientific-writing/`

After reading them, summarize:

1. the project goal,
2. the midway/interim deliverables,
3. the repository structure,
4. the intended PoC notebook workflow,
5. any assumptions that must be verified before running experiments.

Do not modify files before producing this summary unless explicitly instructed.

---

## 2. Repository layout

Use the actual repository layout.

Important paths:

```python
REPO_ROOT = Path.cwd().resolve()
OBJECTRL_DIR = REPO_ROOT / "external" / "objectrl"
GYMNASIUM_DIR = REPO_ROOT / "external" / "Gymnasium"
```

Do **not** assume ObjectRL is located directly under the repository root.

Correct:

```text
external/objectrl/
external/Gymnasium/
```

Incorrect:

```text
objectrl/
gym/
```

---

## 3. Current task: create candidate PoC notebooks

The current task is to create candidate notebooks for the midway PoC.

Do **not** overwrite the final notebook unless explicitly instructed.

Candidate notebook names:

```text
notebooks/DM887_Project_GRPO_Midway_PoC_CopilotCLI.ipynb
notebooks/DM887_Project_GRPO_Midway_PoC_ClaudeCode.ipynb
notebooks/DM887_Project_GRPO_Midway_PoC_Codex.ipynb
```

The final manually merged notebook will be:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

If you are a specific tool, write only to your assigned candidate notebook.

---

## 4. PoC notebook goal

The notebook must be a project-relevant baseline experiment controller.

It should support PPO, SAC, and TD3 baseline experiments using ObjectRL where possible.

It should not be a toy RL notebook.

The notebook should:

1. inspect the local `external/objectrl/` repository,
2. identify available ObjectRL model names/configs for PPO, SAC, and TD3,
3. identify how ObjectRL defines environments and seeds,
4. define the project baseline matrix,
5. support debug, midway, and final run modes,
6. build ObjectRL commands/configurations,
7. run experiments via `subprocess`,
8. save logs and run status files under `results/`,
9. parse evaluation returns if possible,
10. export CSV summaries,
11. generate learning-curve figures under `figures/midway/`,
12. include clear TODOs where ObjectRL names or CLI keys must be manually verified.

---

## 5. Required baseline matrix

Algorithms:

```python
ALGORITHMS = ["ppo", "sac", "td3"]
```

Project environments:

```python
PROJECT_ENVIRONMENTS = [
    "car_racing_continuous",
    "cartpole_swingup",
    "acrobot_swingup",
]
```

Seeds:

```python
SEEDS = [0, 1, 2, 3, 4]
```

Runtime modes:

```python
RUN_CONFIG = {
    "debug": {"seeds": [0], "max_steps": 1_000, "eval_episodes": 1},
    "midway": {"seeds": [0, 1, 2, 3, 4], "max_steps": 20_000, "eval_episodes": 3},
    "final": {"seeds": [0, 1, 2, 3, 4], "max_steps": 500_000, "eval_episodes": 5},
}
```

The notebook may start in `debug` mode.

---

## 6. Important project constraints

Do:

- Use ObjectRL for PPO, SAC, and TD3 where possible.
- Use Gymnasium-style environment assumptions where needed.
- Use reproducible seeds.
- Save machine-readable outputs.
- Save report-ready figures.
- Use `matplotlib` for plots.
- Keep all paths relative to the repository root.
- Log failures instead of crashing the entire experiment matrix.
- Clearly separate debug, midway, and final run budgets.

Do not:

- Implement PPO, SAC, or TD3 from scratch.
- Modify files inside `external/objectrl/` or `external/Gymnasium/` unless explicitly instructed.
- Hard-code user-specific absolute paths.
- Use seaborn.
- Overwrite the final notebook unless explicitly instructed.
- Commit raw logs, large models, videos, or generated checkpoints.

---

## 7. Results and figures

Use these output locations:

```text
results/raw/
results/processed/
results/logs/
figures/midway/
report/midway/figures/
```

Minimum useful generated outputs:

```text
results/processed/status_debug.csv
results/processed/status_midway.csv
results/processed/evaluation_returns_midway.csv
results/processed/evaluation_summary_midway.csv
figures/midway/midway_baselines.pdf
figures/midway/midway_baselines.png
```

If full runs are not completed, export partial results and label them clearly.

---

## 8. Report-facing notebook quality

The notebook should include markdown cells explaining:

1. project context,
2. why ObjectRL is used,
3. algorithms included,
4. environments included,
5. evaluation metric,
6. seeds and run modes,
7. where results are saved,
8. what remains for the final project.

The notebook should produce information that can be transferred directly into the midway LaTeX report.

---

## 9. Midway report connection

The midway report focuses on:

1. related work,
2. formal MDP notation,
3. PPO/SAC/TD3 baseline protocol and results/status.

The notebook should support the experiments section of the report.

The report source lives in:

```text
report/midway/
```

The final report source lives in:

```text
report/final/
```

---

## 10. Reference and paper handling

Do not commit copyrighted paper PDFs.

If paper PDFs are present locally under `papers/`, treat them as local reading material only.

For version-controlled references, prefer:

```text
docs/references/plan-references.md
report/midway/references.bib
report/final/references.bib
papers/bib/
```

If adding citations, use BibTeX entries in `references.bib`.

---

## 11. External repositories

External repositories are ignored by Git and should be treated as dependencies.

ObjectRL:

```text
external/objectrl/
```

Gymnasium:

```text
external/Gymnasium/
```

Do not rewrite external code. Prefer wrappers, scripts, and config files in this repository.

---

## 12. Manual merge strategy

After the three candidate notebooks are generated:

1. keep the cleanest path setup,
2. keep the most robust ObjectRL config discovery,
3. keep the safest subprocess runner,
4. keep the parser that matches actual logs,
5. keep one plotting implementation,
6. remove duplicate cells,
7. run the merged notebook in debug mode,
8. save the final notebook as:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

---

## 13. Commit discipline

Prefer small commits with clear messages.

Suggested commits:

```bash
git add docs/agent-instructions.md plans/plan-poc.md
git commit -m "Add agent instructions for PoC candidates"
```

```bash
git add notebooks/DM887_Project_GRPO_Midway_PoC_*.ipynb
git commit -m "Add candidate midway PoC notebooks"
```

```bash
git add notebooks/DM887_Project_GRPO_Midway_PoC.ipynb scripts configs
git commit -m "Add merged midway ObjectRL baseline PoC"
```

---

## 14. Final reminder

The immediate goal is a useful midway PoC and report support.

Prioritize:

1. reproducibility,
2. clear experiment structure,
3. honest status reporting,
4. report-ready outputs,
5. preserving enough structure for the final project.
