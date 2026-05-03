# DM887 GRPO for Control

This repository contains work for the DM887 Reinforcement Learning term project:

> **GRPO for Control**

The project investigates whether a variant of **Group Relative Policy Optimization (GRPO)** can be adapted to solve continuous-control reinforcement learning tasks and compared against standard baselines such as **PPO**, **SAC**, and **TD3**.

The repository supports both the midway/interim report and the final project report.

---

## Project description

The official project description is included as:

```text
DM887_Project.pdf
```

The project goal is to design, analyze, and evaluate a GRPO-based reinforcement learning algorithm for control tasks.

The required evaluation compares the proposed GRPO variant against:

- PPO
- SAC
- TD3

on the required control environments.

---

## Repository structure

```text
DM887_GRPO_Project/
│
├── README.md
├── .gitignore
├── DM887_Project.pdf
│
├── docs/
│   ├── project.md
│   ├── project-structure.md
│   ├── scientific-writing/
│   └── references/
│
├── plans/
│   ├── plan-poc.md
│   └── plan-midway-rapport-latex.md
│
├── notebooks/
│   ├── DM887_Project_GRPO_Midway_PoC.ipynb
│   └── DM887_Project_GRPO.ipynb
│
├── report/
│   ├── midway/
│   └── final/
│
├── figures/
│   ├── midway/
│   └── final/
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── logs/
│
├── configs/
│   ├── objectrl/
│   └── experiments/
│
├── scripts/
│   ├── run_baselines.py
│   ├── collect_results.py
│   └── plot_results.py
│
├── external/
└── papers/
```

---

## Key documents

- [`docs/project.md`](docs/project.md)  
  Project overview and interpretation of the assignment.

- [`docs/project-structure.md`](docs/project-structure.md)  
  Description of the repository structure.

- [`plans/plan-poc.md`](plans/plan-poc.md)  
  Plan for the midway PoC notebook.

- [`plans/plan-midway-rapport-latex.md`](plans/plan-midway-rapport-latex.md)  
  Plan for the midway LaTeX report.

- [`docs/references/plan-references.md`](docs/references/plan-references.md)  
  Notes on reference management and BibTeX usage.

- [`docs/scientific-writing/`](docs/scientific-writing/)  
  Scientific writing notes used while preparing the report.

---

## Notebooks

The midway PoC notebook is located at:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

The final project notebook will be located at:

```text
notebooks/DM887_Project_GRPO.ipynb
```

The midway notebook is intended to establish the baseline experiment workflow for PPO, SAC, and TD3.

---

## Reports

The midway report source is located in:

```text
report/midway/
```

The final report source will be located in:

```text
report/final/
```

Both reports are intended to use LaTeX and BibTeX-style reference management.

---

## External dependencies

The project may use ObjectRL for baseline implementations of PPO, SAC, and TD3.

External repositories are not committed directly to this repository. They should be placed under:

```text
external/
```

For example:

```text
external/objectrl/
```

---

## Experiment outputs

Generated experiment outputs are kept out of version control unless they are small, processed, and directly needed for the report.

Typical output locations:

```text
results/
figures/
```

Raw logs and large generated files should not be committed.

---

## Current status

This repository is currently structured for the midway/interim phase of the project.

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
