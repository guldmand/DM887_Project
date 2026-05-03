# DM887 Term Project: GRPO for Control — Project Definition

## 0. Purpose of this repository

This repository supports the DM887 Reinforcement Learning term project on **Group Relative Policy Optimization (GRPO) for control**.

The immediate goal is to produce a **midway/interim report** before the deadline. The long-term goal is to extend the same repository into the final project submission.

This repo is intentionally structured so that it can be used with:

- GitHub Copilot CLI
- Claude Code CLI
- ChatGPT Codex
- VS Code GitHub Copilot `/plan` and `/agent`
- Jupyter notebooks
- LaTeX / BibTeX

The repo should preserve a clean separation between:

1. project description and planning,
2. code and experiment execution,
3. generated results and figures,
4. LaTeX report writing,
5. references and bibliography management,
6. scientific writing notes reusable for the master thesis.

---

## 1. Original project goal

The project asks us to design a **GRPO variant for control tasks**.

The project problem can be stated as:

> GRPO has shown strong impact in LLM reinforcement learning, but its potential for robotic/control tasks is underexplored. The project should design a GRPO variant that can solve continuous-control tasks and generalize across different robotic platforms.

The final project must:

- start from vanilla GRPO,
- design a control-suitable GRPO variant,
- evaluate against PPO, SAC, and TD3,
- show learning curves across three environments,
- analyze convergence properties mathematically,
- write the report in the official NeurIPS template.

---

## 2. Instructor clarification relevant to this repository

The instructor clarified that:

1. We do **not** need to implement SAC, TD3, and PPO from scratch.
2. We may use the ObjectRL implementations directly.
3. The continuous version of Car Racing should be used.
4. The reference to discrete-action environments in the project description should be ignored.
5. The ObjectRL TD3 implementation discrepancy from vanilla TD3 can be ignored for project purposes.

This repo therefore treats **ObjectRL as the baseline implementation source** for PPO, SAC, and TD3.

---

## 3. Midway/interim report scope

The midway report should focus on the parts explicitly connected to interim evaluation:

1. **Related work**
   - PPO
   - GAE
   - SAC
   - TD3
   - GRPO
   - GRPO limitations/gap for control tasks

2. **MDP notation and formal RL setup**
   - state space
   - action space
   - transition kernel
   - reward
   - discount factor
   - initial-state distribution
   - policy
   - trajectory
   - return
   - value function
   - action-value function
   - advantage function
   - policy-gradient objective

3. **Baseline experiment results/protocol for PPO, SAC, and TD3**
   - environments
   - seeds
   - evaluation metric
   - learning curve format
   - ObjectRL usage
   - completed runs or clearly documented pilot/interim status

The midway report should use the **full final-report structure**, but only fill the parts relevant to midway. Sections for GRPO design, theory, and final experiments may contain short “planned work” paragraphs.

---

## 4. Final project scope

The final report will extend the midway report with:

1. designed GRPO-control algorithm,
2. pseudocode,
3. component motivation,
4. convergence assumptions and mathematical statements,
5. proofs in appendix,
6. GRPO learning curves,
7. comparison against PPO, SAC, TD3,
8. discussion and limitations.

---

## 5. Proposed repository structure

```text
repo/
├── project.md
├── plan-poc.md
├── plan-midway-rapport-latex.md
├── plan-references.md
├── plan-scientific-writing.md
├── plan-scientific-writing-abstract.md
├── plan-scientific-writing-introduction.md
├── plan-scientific-writing-methodology.md
├── plan-scientific-writing-results-discussion-conclusion.md
├── README.md
├── requirements.txt
├── environment.yml
│
├── assignment/
│   └── DM887_Project.pdf
│
├── scientific-writing/
│   └── DM894_scientific_writing.pdf
│
├── objectrl/
│   └── ... cloned ObjectRL repo ...
│
├── gym/
│   └── ... optional local environment wrappers / notes ...
│
├── notebooks/
│   ├── DM887_Project_GRPO_Midway_PoC.ipynb
│   └── DM887_Project_GRPO.ipynb
│
├── src/
│   ├── __init__.py
│   ├── envs.py
│   ├── objectrl_runner.py
│   ├── results.py
│   ├── plotting.py
│   └── config.py
│
├── configs/
│   ├── midway_baselines.yaml
│   ├── ppo.yaml
│   ├── sac.yaml
│   └── td3.yaml
│
├── results/
│   ├── raw/
│   ├── processed/
│   └── logs/
│
├── figures/
│   ├── midway_baselines.pdf
│   └── midway_baselines.png
│
├── papers/
│   ├── README.md
│   └── bibtex_notes.md
│
└── report-template/
    ├── main.tex
    ├── references.bib
    ├── sections/
    │   ├── 00_abstract.tex
    │   ├── 01_introduction.tex
    │   ├── 02_related_work.tex
    │   ├── 03_methodology.tex
    │   ├── 04_theory.tex
    │   ├── 05_experiments.tex
    │   ├── 06_conclusion.tex
    │   └── 07_ai_use_statement.tex
    └── figures/
        └── midway_baselines.pdf
```

---

## 6. Suggested one-day execution strategy

### Phase 1 — repo setup

Create repository structure and copy planning files.

```bash
mkdir -p repo/{assignment,scientific-writing,notebooks,src,configs,results/raw,results/processed,results/logs,figures,papers,report-template/sections,report-template/figures,gym}
```

Clone ObjectRL:

```bash
git clone <ObjectRL GitHub URL> objectrl
```

### Phase 2 — ask three AI tools for independent PoC designs

Use the prompts in `plan-poc.md`.

Create three candidate outputs:

```text
notebooks/candidates/copilot_candidate.ipynb
notebooks/candidates/claude_candidate.ipynb
notebooks/candidates/codex_candidate.ipynb
```

Then manually merge into:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

### Phase 3 — run baseline experiments

Target matrix:

```text
Algorithms: PPO, SAC, TD3
Environments: continuous Car Racing, cartpole-swingup-v0, acrobot-swingup-v0
Seeds: 0, 1, 2, 3, 4
Metric: undiscounted evaluation episode return
```

Time-budgeted strategy:

1. Run one environment first.
2. Run SAC and TD3 first because they are continuous-control baselines.
3. Run PPO after environment and logging are verified.
4. Expand from seed 0 to seeds 0–4 as time allows.
5. Save all logs, even failed runs.

### Phase 4 — report

Use `plan-midway-rapport-latex.md`.

The report should compile even if some experiments are incomplete. Incomplete work must be clearly labelled as interim status, not presented as final findings.

---

## 7. Definition of done for midway

The midway deliverable is acceptable for this repo when:

- [ ] The LaTeX report compiles.
- [ ] The report uses the official NeurIPS template or a compatible local copy.
- [ ] Related work section is drafted.
- [ ] MDP notation section is drafted.
- [ ] Experiment protocol is specified.
- [ ] ObjectRL baseline usage is stated.
- [ ] PPO, SAC, TD3 are all included in the planned or executed baseline matrix.
- [ ] At least attempted runs are logged.
- [ ] Any completed baseline results are plotted.
- [ ] AI use is acknowledged.
- [ ] BibTeX references compile.

---

## 8. Definition of done for final project

The final project is done when:

- [ ] GRPO-control variant is implemented.
- [ ] GRPO pseudocode is included in the report.
- [ ] Full PPO/SAC/TD3/GRPO comparison is run.
- [ ] Three-environment figure with subpanels is included.
- [ ] Five seeds are used or deviations are justified.
- [ ] Theory section contains assumptions, definitions, lemmas, theorems, and proofs.
- [ ] Appendix contains proofs.
- [ ] Final limitations and future work are discussed.

---

## 9. AI use policy for this repo

AI tools may be used for:

- planning,
- code scaffolding,
- notebook generation,
- LaTeX structure,
- BibTeX cleanup,
- explanation drafts,
- proof brainstorming.

The human author remains responsible for:

- technical correctness,
- final wording,
- source verification,
- experiment validity,
- academic integrity.

Suggested report statement:

```text
AI tools were used to assist with planning, code scaffolding, and initial drafting. All final technical decisions, implementation choices, verification, and written conclusions are the responsibility of the author.
```
