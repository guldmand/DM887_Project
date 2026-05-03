# Project Structure for DM887 GRPO Project

This document defines the recommended repository structure for the DM887 term project on **GRPO for Control**.

The goal of the structure is to keep the repository clean, AI-agent friendly, and reusable for both:

1. the **midway/interim report**, and  
2. the **final project submission**.

The repository should separate project knowledge, planning documents, notebooks, LaTeX report files, figures, experiment results, scripts, and external codebases.

---

## Design principle

Do **not** leave all `.md` files in the repository root.

The root directory should act as the **project entry point**, not as a general storage area.

The root should contain only the most important top-level files, such as:

- `README.md`
- `.gitignore`
- `DM887_Project.pdf`
- optional environment files such as `environment.yml`, `requirements.txt`, or `pyproject.toml`

All other files should be placed into clearly named folders.

---

## Recommended top-level folders

Use the following top-level folders:

| Folder | Purpose |
|---|---|
| `docs/` | Project knowledge base and reusable documentation |
| `plans/` | Concrete work plans for today and future project phases |
| `notebooks/` | Jupyter notebooks for PoC and final project work |
| `report/` | LaTeX report source files |
| `figures/` | Generated plots and report figures |
| `results/` | Raw and processed experiment outputs |
| `configs/` | Experiment and ObjectRL configuration files |
| `scripts/` | Reusable Python scripts for running, collecting, and plotting experiments |
| `external/` | External repositories such as ObjectRL |
| `papers/` | PDFs, BibTeX snippets, paper notes, and reading material |

---

## Recommended repository structure

```text
DM887_GRPO_Project/
│
├── README.md
├── .gitignore
├── DM887_Project.pdf
│
├── docs/
│   ├── project.md
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
│   │   │   ├── 01_introduction.tex
│   │   │   ├── 02_related_work.tex
│   │   │   ├── 03_methodology.tex
│   │   │   ├── 04_experiments.tex
│   │   │   └── 05_status_next_steps.tex
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

## Where to place the current Markdown files

Move the existing planning and knowledge files as follows.

### Project overview

```text
project.md
→ docs/project.md
```

### Work plans

```text
plan-poc.md
→ plans/plan-poc.md
```

```text
plan-midway-rapport-latex.md
→ plans/plan-midway-rapport-latex.md
```

### References and bibliography planning

```text
plan-references.md
→ docs/references/plan-references.md
```

### Scientific writing notes

```text
plan-scientific-writing.md
→ docs/scientific-writing/plan-scientific-writing.md
```

```text
plan-scientific-writing-abstract.md
→ docs/scientific-writing/plan-scientific-writing-abstract.md
```

```text
plan-scientific-writing-introduction.md
→ docs/scientific-writing/plan-scientific-writing-introduction.md
```

```text
plan-scientific-writing-methodology.md
→ docs/scientific-writing/plan-scientific-writing-methodology.md
```

```text
plan-scientific-writing-results-discussion-conclusion.md
→ docs/scientific-writing/plan-scientific-writing-results-discussion-conclusion.md
```

---

## Why use `docs/` instead of `knowledgebase/`?

A folder named `knowledgebase/` would not be wrong, but `docs/` is more standard in GitHub repositories.

The `docs/` folder can still function as the project knowledge base.

Recommended simple version:

```text
docs/
├── project.md
├── scientific-writing/
└── references/
```

A more explicit knowledge-base version could be used later:

```text
docs/
├── project.md
├── knowledge-base/
│   ├── scientific-writing/
│   ├── reinforcement-learning/
│   └── references/
```

For the current one-day midway workflow, keep it simple and use:

```text
docs/
plans/
notebooks/
report/
figures/
results/
configs/
scripts/
external/
papers/
```

This is clean, conventional, and understandable for AI coding agents.

---

## Commands to create the folder structure

Run these commands from the repository root:

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

---

## Commands to move the current files

Run these commands from the repository root after creating the folders:

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

---

## Intended workflow

### 1. Read project context

Start with:

```text
docs/project.md
```

This explains the DM887 GRPO control project, the midway requirements, and the final project direction.

### 2. Follow today's PoC plan

Use:

```text
plans/plan-poc.md
```

This defines what the midway PoC notebook should do.

The notebook should be created here:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

### 3. Build the midway report

Use:

```text
plans/plan-midway-rapport-latex.md
```

The midway LaTeX report should live here:

```text
report/midway/
```

### 4. Use scientific writing notes

Use:

```text
docs/scientific-writing/
```

These files summarize reusable scientific writing principles for the midway report and later thesis writing.

### 5. Manage references

Use:

```text
docs/references/plan-references.md
```

The actual BibTeX file for the midway report should be:

```text
report/midway/references.bib
```

The final report can later have its own:

```text
report/final/references.bib
```

### 6. Save experiment outputs

Use:

```text
results/raw/
results/processed/
results/logs/
figures/midway/
```

The report-specific figure folder can contain copies or symlinks:

```text
report/midway/figures/
```

---

## Notes for AI agents

AI agents should treat the repository as follows:

1. `README.md` is the entry point.
2. `docs/project.md` defines the project.
3. `plans/plan-poc.md` defines the immediate notebook task.
4. `plans/plan-midway-rapport-latex.md` defines the report task.
5. `docs/scientific-writing/` defines the writing style and report-quality rules.
6. `docs/references/plan-references.md` defines citation and BibTeX conventions.
7. `external/objectrl/` is external code and should not be rewritten unless explicitly requested.
8. `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` is the immediate PoC notebook.
9. `report/midway/` is the immediate report target.
10. `notebooks/DM887_Project_GRPO.ipynb` and `report/final/` are for the final project.

---

## Short recommendation

Use:

```text
/docs/
```

as the project knowledge base.

Use:

```text
/plans/
```

for concrete work plans.

Keep the root directory clean.
