# Common Context for DM887 GRPO Agent Prompts

Use this context for all AI coding agents working on the DM887 GRPO project.

## Read first

Read the repository context files first:

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

## Required repository paths

Use the actual repository layout:

```python
REPO_ROOT = Path.cwd().resolve()
OBJECTRL_DIR = REPO_ROOT / "external" / "objectrl"
GYMNASIUM_DIR = REPO_ROOT / "external" / "Gymnasium"
```

Do not assume ObjectRL is located directly under the repository root.

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
