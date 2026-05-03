# Prompts for DM887 GRPO Project

This folder contains final prompts for generating candidate PoC notebooks using different AI coding tools.

Recommended folder location:

```text
prompts/
```

## Files

- `prompt-copilot-cli.md`
- `prompt-claude-code.md`
- `prompt-codex.md`
- `prompt-merge-final-notebook.md`
- `common-agent-context.md`

## Workflow

1. Run the Copilot CLI prompt to create:

```text
notebooks/DM887_Project_GRPO_Midway_PoC_CopilotCLI.ipynb
```

2. Run the Claude Code prompt to create:

```text
notebooks/DM887_Project_GRPO_Midway_PoC_ClaudeCode.ipynb
```

3. Run the Codex prompt to create:

```text
notebooks/DM887_Project_GRPO_Midway_PoC_Codex.ipynb
```

4. Inspect all three notebooks.

5. Use `prompt-merge-final-notebook.md` to create:

```text
notebooks/DM887_Project_GRPO_Midway_PoC.ipynb
```

## Local material

The repository may contain Git-ignored local folders:

```text
papers/
slides/
```

These are supporting material only. Agents should prioritize committed files such as:

```text
docs/references/reading-list.md
docs/course-materials.md
```
