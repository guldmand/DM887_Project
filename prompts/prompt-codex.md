# Prompt: ChatGPT Codex Candidate Notebook

Use this prompt from the repository root.

```text
Read the repository context files first:

- README.md
- AGENTS.md
- docs/project.md
- docs/project-structure.md
- docs/course-materials.md
- docs/references/plan-references.md
- docs/references/reading-list.md
- plans/plan-poc.md
- plans/plan-midway-rapport-latex.md
- docs/scientific-writing/

The repository may also contain local Git-ignored folders:

- papers/
- slides/

Use papers/ and slides/ only as supporting local material if needed. Do not copy long passages from them. Prefer the committed reading-list and BibTeX references for report-facing claims.

Before editing, summarize:
1. the project goal,
2. the midway deliverables,
3. the repository structure,
4. the intended PoC workflow,
5. assumptions that must be verified before running experiments.

Then create this candidate notebook:

notebooks/DM887_Project_GRPO_Midway_PoC_Codex.ipynb

Use the actual repository layout:

OBJECTRL_DIR = REPO_ROOT / "external" / "objectrl"
GYMNASIUM_DIR = REPO_ROOT / "external" / "Gymnasium"

Build a DM887 midway baseline PoC notebook that uses ObjectRL implementations of PPO, SAC, and TD3. The notebook must support the three project environments and five seeds, with debug/midway/final run modes.

Required features:
1. ObjectRL config discovery,
2. model name discovery for PPO, SAC, and TD3,
3. environment/config argument discovery,
4. command generation,
5. dry-run mode,
6. subprocess execution,
7. log saving,
8. run status JSON/CSV export,
9. evaluation-return parsing,
10. aggregation by algorithm/environment/seed/step,
11. learning-curve export to figures/midway/,
12. report-facing markdown notes.

The baseline matrix must include:
- PPO, SAC, TD3
- continuous Car Racing, cartpole-swingup-v0, acrobot-swingup-v0
- seeds 0, 1, 2, 3, 4

Do not use seaborn. Do not implement PPO/SAC/TD3 from scratch. Do not modify external repositories. Do not overwrite notebooks/DM887_Project_GRPO_Midway_PoC.ipynb. Make it ready for VS Code/Jupyter.
```
