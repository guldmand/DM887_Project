# Prompt: Claude Code CLI Candidate Notebook

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

Do not modify files until you have summarized:
1. the DM887 GRPO project goal,
2. the midway report requirements,
3. the repository structure,
4. the intended PoC workflow,
5. assumptions that must be verified before running experiments.

Then create this candidate notebook:

notebooks/DM887_Project_GRPO_Midway_PoC_ClaudeCode.ipynb

This must be a project-relevant ObjectRL baseline controller, not a toy RL notebook. It should orchestrate PPO, SAC, and TD3 baseline runs for the project environments. Use the local external repositories:

OBJECTRL_DIR = REPO_ROOT / "external" / "objectrl"
GYMNASIUM_DIR = REPO_ROOT / "external" / "Gymnasium"

The notebook should include:
1. markdown explanations suitable for the midway report,
2. robust path handling,
3. ObjectRL config inspection,
4. environment/model/seed argument discovery,
5. experiment matrix definition,
6. debug/midway/final modes,
7. dry-run support,
8. subprocess execution,
9. status JSON files,
10. log capture,
11. evaluation-return extraction,
12. CSV export,
13. aggregation by algorithm/environment/seed/step,
14. matplotlib learning curves exported to figures/midway/.

The baseline matrix must include PPO, SAC, and TD3 on continuous Car Racing, cartpole-swingup-v0, and acrobot-swingup-v0, with seeds 0, 1, 2, 3, 4.

Do not rewrite ObjectRL and do not implement PPO/SAC/TD3 from scratch. Do not use seaborn. Do not overwrite notebooks/DM887_Project_GRPO_Midway_PoC.ipynb.
```
