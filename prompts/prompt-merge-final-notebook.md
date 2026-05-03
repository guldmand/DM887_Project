# Prompt: Merge Candidate Notebooks into Final Midway PoC

Use this prompt only after the three candidate notebooks have been created.

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

Then inspect these candidate notebooks:

- notebooks/DM887_Project_GRPO_Midway_PoC_CopilotCLI.ipynb
- notebooks/DM887_Project_GRPO_Midway_PoC_ClaudeCode.ipynb
- notebooks/DM887_Project_GRPO_Midway_PoC_Codex.ipynb

Create the final merged notebook:

notebooks/DM887_Project_GRPO_Midway_PoC.ipynb

The merged notebook should keep:
1. the cleanest path setup,
2. the most robust ObjectRL config discovery,
3. the safest subprocess runner,
4. the best log/status handling,
5. the most adaptable evaluation-return parser,
6. one clean plotting implementation,
7. markdown cells useful for the midway report.

Use the actual repository layout:

OBJECTRL_DIR = REPO_ROOT / "external" / "objectrl"
GYMNASIUM_DIR = REPO_ROOT / "external" / "Gymnasium"

Do not implement PPO/SAC/TD3 from scratch. Use ObjectRL where possible. Do not modify external repositories. Use matplotlib, not seaborn. Keep all paths relative to the repository root.

The final notebook must run in debug mode from top to bottom, or clearly indicate which cells require manual ObjectRL configuration verification before execution.
```
