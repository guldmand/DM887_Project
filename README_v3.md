# DM887 GRPO for Control

This repository contains my working material for the DM887 Reinforcement Learning term project:

> **GRPO for Control**

The project investigates whether a variant of **Group Relative Policy Optimization (GRPO)** can be adapted to solve control tasks and compared against standard reinforcement learning baselines: **PPO**, **SAC**, and **TD3**.

The project assignment requires evaluation on:

- `CarRacing-v3` from Gymnasium/Farama Box2D,
- `cartpole-swingup-v0`,
- `acrobot-swingup-v0`,

using five seeds and learning curves based on undiscounted evaluation episode return.

---

## Current repository status

This repository is currently in the midway/interim phase.

The current implementation supports baseline experiments for:

| Environment | PPO | SAC | TD3 | Notes |
|---|---:|---:|---:|---|
| `cartpole_swingup` | yes | yes | yes | Vector observation environment, run through the ObjectRL baseline workflow. |
| `acrobot_swingup` | yes | yes | yes | Vector observation environment, run through the ObjectRL baseline workflow. |
| `car_racing_continuous` | yes | yes | yes | Image observation environment, run through a project-side PyTorch CNN baseline workflow. |

The vector-environment midway baselines have been completed locally for PPO/SAC/TD3 with seeds `0..4`.

The CarRacing CNN baseline batch is designed to run on Google Colab with CUDA and writes persistent outputs to Google Drive. This is because CarRacing uses image observations and is much slower on local CPU.

---

## Repository structure

```text
DM887_Project/
│
├── README.md
├── AGENTS.md
├── DM887_Project.pdf
│
├── configs/
│
├── docs/
│   ├── project.md
│   ├── project-structure.md
│   ├── scientific-writing/
│   └── references/
│
├── external/
│   └── objectrl/
│
├── figures/
│   └── midway/
│
├── notebooks/
│   ├── DM887_Project_GoogleColab.ipynb
│   ├── DM887_Project_GRPO_Midway_PoC.ipynb
│   ├── DM887_Project_GRPO_Midway_PoC_ClaudeCode.ipynb
│   ├── DM887_Project_GRPO_Midway_PoC_Codex.ipynb
│   └── DM887_Project_GRPO_Midway_PoC_CopilotCLI.ipynb
│
│
├── plans/
│   ├── plan-poc.md
│   └── plan-midway-rapport-latex.md
│
├── prompts/
│   ├── common-agent-context.md
│   ├── prompt-claude-code.md
│   ├── prompt-codex.md
│   ├── prompt-copilot-cli.md
│   └── prompt-merge-final-notebook.md
│
├── report/
│   ├── DM887_Report.tex
│   ├── DM887_Report.pdf
│   ├── checklist.tex
│   ├── neurips_2026.sty
│   ├── references.bib
│   ├── sections/
│   ├── figures/
│   └── builds/
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── logs/
│
├── scripts/
│   ├── carracing_cnn.py
│   ├── project_envs.py
│   ├── register_project_envs.py
│   ├── run_carracing_cnn_baseline.py
│   ├── run_midway_vector_baselines.sh
│   ├── run_project_objectrl_baseline.py
│   └── summarize_project_baselines.py
```

---

## Main scripts

### `scripts/project_envs.py`

Defines the project-specific environment registration and wrappers used by the baseline runners.

### `scripts/register_project_envs.py`

Registers the project environments before running experiments.

### `scripts/run_project_objectrl_baseline.py`

Runs PPO, SAC, and TD3 baselines for the vector-control environments:

- `cartpole_swingup`
- `acrobot_swingup`

This is the ObjectRL-based baseline path.

### `scripts/carracing_cnn.py`

Project-side PyTorch implementation for CarRacing image observations.

It contains:

- CarRacing preprocessing wrapper,
- CNN feature extractor,
- PPO-CNN,
- SAC-CNN,
- TD3-CNN,
- replay / rollout buffers,
- training loops.

This exists because ObjectRL's default baseline path expects 1-D vector observations, while CarRacing returns image observations.

### `scripts/run_carracing_cnn_baseline.py`

Runs the CarRacing CNN baselines for PPO, SAC, and TD3.

Typical Colab command:

```bash
python -u scripts/run_carracing_cnn_baseline.py \
  --mode midway \
  --algorithm all \
  --device cuda \
  --time-limit-minutes 60 \
  --allow-batch-run \
  --run
```

### `scripts/summarize_project_baselines.py`

Aggregates baseline CSV files and writes summary plots.

Example:

```bash
python scripts/summarize_project_baselines.py --prefix midway
```

---

## Notebooks

The notebook directory currently contains several AI-tool generated or assisted candidate notebooks.

```text
notebooks/
├── DM887_Project_GoogleColab.ipynb
├── DM887_Project_GRPO_Midway_PoC.ipynb
├── DM887_Project_GRPO_Midway_PoC_ClaudeCode.ipynb
├── DM887_Project_GRPO_Midway_PoC_Codex.ipynb
└── DM887_Project_GRPO_Midway_PoC_CopilotCLI.ipynb
```

Current practical status:

- `DM887_Project_GRPO_Midway_PoC_Codex.ipynb` is the main local working notebook for the midway PoC.
- `DM887_Project_GoogleColab.ipynb` is used for running the CarRacing CNN baselines on Colab GPU.
- `DM887_Project_GRPO_Midway_PoC_ClaudeCode.ipynb` and `DM887_Project_GRPO_Midway_PoC_CopilotCLI.ipynb` are candidate notebooks from other AI-tool workflows.
- Note: From the original condidate notebooks, i decided to continue building upon the `DM887_Project_GRPO_Midway_PoC_Codex.ipynb`.
- A final cleaned notebook should be produced later by merging the best, non-duplicated, and up-to-date material from the candidate notebooks.

The final merged notebook should be created only after the CarRacing Colab results have been copied back into the local repository and the figures have been regenerated.

---

## Report structure

The report is now stored directly under `report/`, not under separate `report/midway/` and `report/final/` folders.

Current layout:

```text
report/
├── DM887_Report.tex
├── DM887_Report.pdf
├── checklist.tex
├── neurips_2026.sty
├── references.bib
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_theory.tex
│   ├── 05_experiments.tex
│   ├── 06_conclusion.tex
│   └── A_proofs.tex
├── figures/
└── builds/
```

The report uses the NeurIPS-style template required by the assignment.

---

## Experiment outputs

Experiment outputs are stored under:

```text
results/
├── raw/project_baselines/
├── processed/project_baselines/
└── logs/
```

Processed CSV files follow this naming pattern:

```text
<prefix>_<algorithm>_<environment>_seed<seed>_eval.csv
```

Examples:

```text
midway_ppo_cartpole_swingup_seed0_eval.csv
midway_sac_acrobot_swingup_seed4_eval.csv
midway_td3_car_racing_continuous_seed2_eval.csv
```

The main summary command is:

```bash
python scripts/summarize_project_baselines.py --prefix midway
```

This produces:

```text
results/processed/project_baselines/midway_vector_summary.csv
figures/midway/midway_cartpole_swingup_baselines.png
figures/midway/midway_acrobot_swingup_baselines.png
figures/midway/midway_car_racing_continuous_baselines.png
figures/midway/midway_vector_env_baselines.png
```

---

## Google Colab output workflow

CarRacing CNN runs are GPU-heavy and are intended to run on Colab with CUDA.

The Colab workflow writes persistent outputs to:

```text
/content/drive/MyDrive/DM887_Project_Colab_Outputs/
├── project_baselines/
├── logs/
├── raw/
└── status/
```

After Colab finishes, copy the generated `midway_*_car_racing_continuous_seed*_eval.csv` files from Google Drive into the local repository:

```text
results/processed/project_baselines/
```

Then rerun:

```bash
python scripts/summarize_project_baselines.py --prefix midway
```

The expected complete midway baseline matrix is:

```text
Algorithms:   ppo, sac, td3
Environments: cartpole_swingup, acrobot_swingup, car_racing_continuous
Seeds:        0, 1, 2, 3, 4
Total runs:   45
```

---

## Current experiment status

### Completed locally

The following vector-control baseline matrix has been completed locally:

```text
Algorithms:   ppo, sac, td3
Environments: cartpole_swingup, acrobot_swingup
Seeds:        0, 1, 2, 3, 4
Runs:         30
```

### Running / to be imported from Colab

The following CarRacing CNN baseline matrix is intended to be run on Colab GPU:

```text
Algorithms:   ppo, sac, td3
Environment:  car_racing_continuous
Seeds:        0, 1, 2, 3, 4
Runs:         15
```

Once these 15 CSV files are copied back from Google Drive, the complete baseline matrix will contain 45 runs.

---

## External dependencies

The project uses ObjectRL for vector-environment PPO/SAC/TD3 baseline runs where possible.

External repositories are placed under:

```text
external/
```

For example:

```text
external/objectrl/
```

Do not rewrite external code. Project-specific adapters, wrappers, runners, and analysis code should live in this repository's own `scripts/`, `configs/`, and `notebooks/` directories.

---

## Final notebook merge plan

The current notebooks contain useful material but also duplicated or outdated cells from earlier AI-tool iterations.

Before final submission, create a clean merged notebook that:

1. keeps the best setup/path handling,
2. keeps the verified baseline runners,
3. removes outdated notes about CarRacing being deferred,
4. keeps the final PPO/SAC/TD3 baseline result loading,
5. includes the final generated figures,
6. clearly states which results were run locally and which were run on Colab,
7. prepares the path for the final GRPO implementation.

A reasonable target filename is:

```text
notebooks/DM887_Project_GRPO_Midway_PoC_Final.ipynb
```

The existing candidate notebooks should be kept as provenance until the final notebook has been verified.

---

## Current summary

At the current stage, the repository contains:

- a NeurIPS-style LaTeX report scaffold,
- related work and MDP/report planning documents,
- ObjectRL-based vector-control baseline workflow,
- project-side PyTorch CNN baseline workflow for CarRacing,
- PPO/SAC/TD3 baseline support for all three required environments,
- completed vector-control baseline CSV files,
- Colab workflow for completing the CarRacing baseline CSV files.

Remaining work:

- finish or import all CarRacing Colab outputs,
- regenerate final baseline summary CSV and figures,
- clean and merge the notebooks,
- update the report experiments section with the completed figures,
- implement and evaluate the GRPO-control variant for the final project,
- complete the theory/proofs section for the final project.


---

## Project Status

The midway focus is:

- literature and related work,
- MDP notation,
- baseline experiment setup,
- PPO/SAC/TD3 baseline runs,
- LaTeX report preparation.

The final phase will extend this work with:

- the proposed GRPO-control variant,
- theoretical analysis,
- full experimental comparison,
- final report writing.
