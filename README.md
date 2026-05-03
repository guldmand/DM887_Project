# DM887 GRPO for Control

This repository contains my working material for the DM887 Reinforcement Learning term project:

> **GRPO for Control**

The project investigates whether a variant of **Group Relative Policy Optimization (GRPO)** can be adapted to solve robotic/control tasks and compared against standard reinforcement learning baselines such as **PPO**, **SAC**, and **TD3**.

This repository is structured to support both:

1. the **midway/interim report**, and  
2. the **final project report and implementation**.

The midway work focuses on:

- related work,
- MDP notation,
- baseline protocol,
- PPO/SAC/TD3 baseline experiments,
- report structure,
- reference management,
- scientific writing practice.

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
│   │
│   ├── scientific-writing/
│   │   ├── plan-scientific-writing.md
│   │   ├── plan-scientific-writing-abstract.md
│   │   ├── plan-scientific-writing-introduction.md
│   │   ├── plan-scientific-writing-methodology.md
│   │   └── plan-scientific-writing-results-discussion-conclusion.md
│   │
│   └── references/
│       └── plan-references.md
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
│   │   ├── main.tex
│   │   ├── references.bib
│   │   ├── sections/
│   │   └── figures/
│   │
│   └── final/
│       ├── main.tex
│       ├── references.bib
│       ├── sections/
│       └── figures/
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
│   └── objectrl/
│
└── papers/
    ├── readme.md
    └── bib/
```

---

## Key files

### Project documentation

- [`docs/project.md`](docs/project.md)  
  Main project description, interpretation of the assignment, midway scope, and final-project direction.

- [`docs/project-structure.md`](docs/project-structure.md)  
  Explanation of the repository structure and where each type of file belongs.

### Work plans

- [`plans/plan-poc.md`](plans/plan-poc.md)  
  Plan for building the midway PoC notebook using ObjectRL, Gymnasium-style environment handling, PPO, SAC, TD3, logging, and plotting.

- [`plans/plan-midway-rapport-latex.md`](plans/plan-midway-rapport-latex.md)  
  Plan for writing the midway report in LaTeX using the required report structure.

### Scientific writing notes

- [`docs/scientific-writing/plan-scientific-writing.md`](docs/scientific-writing/plan-scientific-writing.md)  
  General scientific writing principles.

- [`docs/scientific-writing/plan-scientific-writing-abstract.md`](docs/scientific-writing/plan-scientific-writing-abstract.md)  
  Notes for writing the abstract.

- [`docs/scientific-writing/plan-scientific-writing-introduction.md`](docs/scientific-writing/plan-scientific-writing-introduction.md)  
  Notes for writing the introduction.

- [`docs/scientific-writing/plan-scientific-writing-methodology.md`](docs/scientific-writing/plan-scientific-writing-methodology.md)  
  Notes for writing methodology, experimental setup, notation, and reproducibility details.

- [`docs/scientific-writing/plan-scientific-writing-results-discussion-conclusion.md`](docs/scientific-writing/plan-scientific-writing-results-discussion-conclusion.md)  
  Notes for writing results, discussion, limitations, and conclusion.

### References

- [`docs/references/plan-references.md`](docs/references/plan-references.md)  
  Plan for using BibTeX/BibLaTeX-style references and maintaining `references.bib`.

---

## Main notebooks

### Midway PoC notebook

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

Purpose:

- run or orchestrate PPO/SAC/TD3 baseline experiments,
- build on ObjectRL where possible,
- use Gymnasium-style environment interaction,
- define seeds and evaluation protocol,
- collect logs and returns,
- generate plots for the midway report.

### Final project notebook

```text
notebooks/DM887_Project_GRPO.ipynb
```

Purpose:

- extend the midway PoC,
- include the proposed GRPO variant,
- include final PPO/SAC/TD3/GRPO comparisons,
- export final figures and tables.

---

## Report structure

### Midway report

The midway report lives in:

```text
report/midway/
```

Expected files:

```text
report/midway/
├── main.tex
├── references.bib
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_experiments.tex
│   └── 05_status_next_steps.tex
└── figures/
```

The midway report should focus on:

1. related work,
2. MDP notation,
3. PPO/SAC/TD3 baseline protocol and results/status,
4. ObjectRL usage,
5. next steps toward the final GRPO variant.

### Final report

The final report lives in:

```text
report/final/
```

The final report extends the midway report with:

- the proposed GRPO-control algorithm,
- pseudocode,
- theoretical/convergence analysis,
- full experiments,
- final discussion and conclusion.

---

## Results and figures

Experiment outputs should be separated from report source files.

Use:

```text
results/raw/
results/processed/
results/logs/
```

for raw logs, processed metrics, and execution logs.

Use:

```text
figures/midway/
figures/final/
```

for generated figures.

Figures used directly in the report can be copied into:

```text
report/midway/figures/
report/final/figures/
```

---

## External code

External repositories should be placed in:

```text
external/
```

For ObjectRL:

```text
external/objectrl/
```

The ObjectRL codebase should be treated as an external dependency. Do not rewrite it unless explicitly needed. Prefer wrappers, configuration files, and scripts in this repository.

---

## Suggested setup commands

Create the folder structure:

```bash
mkdir -p docs/scientific-writing
mkdir -p docs/references
mkdir -p plans
mkdir -p notebooks
mkdir -p report/midway/sections
mkdir -p report/midway/figures
mkdir -p report/final/sections
mkdir -p report/final/figures
mkdir -p figures/midway figures/final
mkdir -p results/raw results/processed results/logs
mkdir -p configs/objectrl configs/experiments
mkdir -p scripts
mkdir -p external
mkdir -p papers/bib
```

Move the current planning files:

```bash
mv project.md docs/project.md

mv plan-poc.md plans/plan-poc.md
mv plan-midway-rapport-latex.md plans/plan-midway-rapport-latex.md

mv plan-references.md docs/references/plan-references.md

mv plan-scientific-writing.md docs/scientific-writing/plan-scientific-writing.md
mv plan-scientific-writing-abstract.md docs/scientific-writing/plan-scientific-writing-abstract.md
mv plan-scientific-writing-introduction.md docs/scientific-writing/plan-scientific-writing-introduction.md
mv plan-scientific-writing-methodology.md docs/scientific-writing/plan-scientific-writing-methodology.md
mv plan-scientific-writing-results-discussion-conclusion.md docs/scientific-writing/plan-scientific-writing-results-discussion-conclusion.md
```

Clone ObjectRL:

```bash
cd external
git clone https://github.com/adinlab/objectrl.git
cd ..
```

---

## AI-agent workflow

This repository is designed so that AI agents can work from clear context files.

Recommended order for AI agents:

1. Read [`README.md`](README.md).
2. Read [`docs/project.md`](docs/project.md).
3. Read [`docs/project-structure.md`](docs/project-structure.md).
4. Read [`plans/plan-poc.md`](plans/plan-poc.md).
5. Read [`plans/plan-midway-rapport-latex.md`](plans/plan-midway-rapport-latex.md).
6. Use [`docs/scientific-writing/`](docs/scientific-writing/) when drafting report text.
7. Use [`docs/references/plan-references.md`](docs/references/plan-references.md) when adding citations.

---

## Immediate midway tasks

The immediate tasks are:

1. Set up the folder structure.
2. Move the `.md` files into their proper locations.
3. Clone ObjectRL into `external/objectrl/`.
4. Create the midway PoC notebook:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

5. Create the midway report LaTeX files in:

```text
report/midway/
```

6. Run whatever PPO/SAC/TD3 baseline experiments are feasible before the deadline.
7. Export figures to:

```text
figures/midway/
report/midway/figures/
```

8. Compile the midway report PDF.

---

## Final project tasks

After the midway submission:

1. Extend the PoC notebook into the final project notebook.
2. Implement the GRPO-control variant.
3. Run full PPO/SAC/TD3/GRPO comparisons.
4. Complete theory and convergence discussion.
5. Produce final plots.
6. Write and compile the final report.

---

## Short rule

Keep the root directory clean.

Use:

```text
docs/
```

as the project knowledge base.

Use:

```text
plans/
```

for concrete work plans.

Use:

```text
notebooks/
```

for executable research notebooks.

Use:

```text
report/
```

for LaTeX report source files.
