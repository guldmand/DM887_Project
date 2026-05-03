# Prompt: GitHub Copilot CLI Candidate Notebook

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
1. the project goal,
2. the midway deliverables,
3. the repository structure,
4. the intended PoC notebook workflow,
5. any assumptions that must be verified before running experiments.

After the summary, create this candidate notebook:

notebooks/DM887_Project_GRPO_Midway_PoC_CopilotCLI.ipynb

The notebook must use the actual repository structure:

OBJECTRL_DIR = REPO_ROOT / "external" / "objectrl"
GYMNASIUM_DIR = REPO_ROOT / "external" / "Gymnasium"

The notebook should build a project-relevant midway PoC for PPO, SAC, and TD3 using ObjectRL where possible. It must inspect ObjectRL configs, identify available model names and environment/config arguments, define debug/midway/final run modes, define the baseline matrix, build commands, run subprocess experiments, save logs/status files under results/, parse evaluation returns if possible, aggregate results by algorithm/environment/seed/step, and export learning-curve figures under figures/midway/.

The baseline matrix must include:

Algorithms:
- PPO
- SAC
- TD3

Project environments:
- continuous Car Racing
- cartpole-swingup-v0
- acrobot-swingup-v0

Seeds:
- 0, 1, 2, 3, 4

Use runtime modes:
- debug: one seed and short run budget
- midway: five seeds and smaller training budget
- final: five seeds and larger training budget

Do not implement PPO/SAC/TD3 from scratch. Use ObjectRL. Do not rewrite ObjectRL. Do not use seaborn. Include clear TODOs where ObjectRL environment names or CLI keys must be verified.

Make the notebook ready for VS Code/Jupyter.
```
