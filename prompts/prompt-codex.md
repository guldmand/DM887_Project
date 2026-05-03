# Prompt: ChatGPT Codex Candidate Notebook

Use this prompt from the repository root.

Copy/paste this entire prompt into the tool. It is self-contained and already includes the full common project context.

```text
# Common Context for DM887 GRPO Agent Prompts

Use this context for all AI coding agents working on the DM887 GRPO project.

## Official assignment source and midway scope

The authoritative task definition is the file:

```text
DM887_Project.pdf
```

You must treat `DM887_Project.pdf` as the source of truth for the project.

### Full final project task

The full project is:

> Design, analyze, and evaluate a GRPO variant for control tasks.

The final project must:

1. start from vanilla Group Relative Policy Optimization (GRPO),
2. design a GRPO variant that can solve control tasks,
3. evaluate it against PPO, SAC, and TD3,
4. use the required control environments,
5. provide theoretical/convergence-oriented analysis,
6. write the final report in the required NeurIPS-style structure.

### Required environments

Use the course project environments:

1. continuous Car Racing from Gymnasium/Farama Box2D,
2. `cartpole-swingup-v0` from DeepMind Control Suite,
3. `acrobot-swingup-v0` from DeepMind Control Suite.

Instructor clarification: use the continuous version of Car Racing and ignore the original reference to discrete-action environments.

### Evaluation protocol

The project requires:

- online training,
- regular evaluation intervals,
- learning curves,
- y-axis: undiscounted evaluation episode return,
- x-axis: number of training steps before each evaluation,
- five seeds,
- seed numbers specified in the report,
- Python and PyTorch implementation.

### Midway/interim scope

The midway/interim deliverable is **not the full final GRPO project**.

For the midway version, focus only on the parts of `DM887_Project.pdf` that are explicitly evaluated as interim/midway work:

1. **Related work**  
   The related work section is evaluated as part of the interim report.

2. **MDP notation / formal setup**  
   The methodology subsection that introduces the complete set of basic notation required to describe the key elements of the MDP is evaluated as part of the interim report.

3. **PPO/SAC/TD3 baseline results/protocol**  
   In the experiments section, 10 points are reserved for complete PPO, SAC, and TD3 results to be graded in the interim report.

The midway PoC notebook should therefore prioritize the PPO, SAC, and TD3 baseline pipeline and report-ready outputs. It should not attempt to implement the final GRPO variant unless explicitly requested.

### What the midway PoC notebook must support

The candidate midway PoC notebook must support:

- ObjectRL-based PPO, SAC, and TD3 baseline runs,
- the required environments,
- five seeds,
- debug/midway/final run modes,
- logging under `results/`,
- figure export under `figures/midway/`,
- report-facing markdown notes explaining how the notebook supports the midway report.

Do not implement the final GRPO-control algorithm in the midway candidate notebook unless explicitly instructed.

---

## Read first

Read the repository context files first:

- `DM887_Project.pdf`
- `README.md`
- `AGENTS.md`
- `docs/project.md`
- `docs/project-structure.md`
- `docs/course-materials.md`
- `docs/references/plan-references.md`
- `docs/references/reading-list.md`
- `plans/plan-poc.md`
- `plans/plan-midway-rapport-latex.md`
- `docs/scientific-writing/`

The repository may also contain local Git-ignored folders:

- `papers/`
- `slides/`

Use `papers/` and `slides/` only as supporting local material if needed. Do not copy long passages from them. Prefer the committed reading list and BibTeX references for report-facing claims.

Before making changes, summarize:

1. the full final project goal from `DM887_Project.pdf`,
2. the exact midway/interim deliverables,
3. the repository structure,
4. the intended PoC notebook workflow,
5. assumptions that must be verified before running experiments.

---

## Required repository paths

Use the actual repository layout:

```python
REPO_ROOT = Path.cwd().resolve()
OBJECTRL_DIR = REPO_ROOT / "external" / "objectrl"
GYMNASIUM_DIR = REPO_ROOT / "external" / "Gymnasium"
```

Do not assume ObjectRL is located directly under the repository root.

---

## General constraints

- Do not implement PPO, SAC, or TD3 from scratch.
- Use ObjectRL for PPO, SAC, and TD3 where possible.
- Do not modify files inside `external/objectrl/` or `external/Gymnasium/` unless explicitly instructed.
- Use Gymnasium-style environment assumptions where needed.
- Use `matplotlib`, not seaborn.
- Keep paths relative to the repository root.
- Save logs/results under `results/`.
- Save midway figures under `figures/midway/`.
- The notebook must be a project-relevant baseline experiment controller, not a toy RL notebook.
- Include clear TODOs where ObjectRL config names, environment names, or CLI keys must be manually verified.


---

# Tool-specific task

Create this candidate notebook:

notebooks/DM887_Project_GRPO_Midway_PoC_Codex.ipynb

The notebook must be a project-relevant ObjectRL baseline controller, not a toy RL notebook.

The baseline matrix must include:

Algorithms:
- PPO
- SAC
- TD3

Project environments:
- continuous Car Racing from Gymnasium/Farama Box2D
- cartpole-swingup-v0 from DeepMind Control Suite
- acrobot-swingup-v0 from DeepMind Control Suite

Seeds:
- 0, 1, 2, 3, 4

Evaluation:
- online training
- regular evaluation intervals
- y-axis: undiscounted evaluation episode return
- x-axis: number of training steps before each evaluation
- export learning curves to figures/midway/

Use runtime modes:
- debug: one seed and short run budget
- midway: five seeds and smaller training budget
- final: five seeds and larger training budget

Required notebook features:
1. markdown explanation of the official assignment and midway scope,
2. robust path handling,
3. ObjectRL config inspection,
4. model name discovery for PPO, SAC, and TD3,
5. environment/config argument discovery,
6. experiment matrix definition,
7. command generation,
8. dry-run mode,
9. subprocess execution,
10. log saving,
11. run status JSON/CSV export,
12. evaluation-return parsing if possible,
13. aggregation by algorithm/environment/seed/step,
14. learning-curve export to figures/midway/,
15. report-facing notes for the midway report,
16. clear TODOs where ObjectRL environment names or CLI keys must be verified.

Do not implement PPO/SAC/TD3 from scratch. Use ObjectRL. Do not rewrite ObjectRL. Do not modify external repositories. Do not use seaborn. Do not overwrite notebooks/DM887_Project_GRPO_Midway_PoC.ipynb.

Make it ready for VS Code/Jupyter.
```
