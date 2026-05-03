# Prompts for DM887 GRPO Project

This folder contains self-contained prompts for generating candidate PoC notebooks using different AI coding tools.

Each tool-specific prompt now includes the full common project context. You do **not** need to paste `common-agent-context.md` first.

## Files

- `common-agent-context.md`  
  Shared context for documentation and future prompt editing.

- `prompt-copilot-cli.md`  
  Self-contained prompt for GitHub Copilot CLI.

- `prompt-claude-code.md`  
  Self-contained prompt for Claude Code CLI.

- `prompt-codex.md`  
  Self-contained prompt for ChatGPT Codex.

- `prompt-merge-final-notebook.md`  
  Self-contained prompt for merging the three candidate notebooks.

## Workflow

1. Paste the full contents of `prompt-copilot-cli.md` into Copilot CLI.
2. Paste the full contents of `prompt-claude-code.md` into Claude Code.
3. Paste the full contents of `prompt-codex.md` into Codex.
4. Inspect all three candidate notebooks.
5. Use `prompt-merge-final-notebook.md` to create the final merged notebook.

## Important

The prompts explicitly state that:

- `DM887_Project.pdf` is the source of truth.
- The full project is a GRPO-control project.
- The midway scope is only:
  1. related work,
  2. MDP notation/formal setup,
  3. PPO/SAC/TD3 baseline results/protocol.
- The midway notebook should not implement the final GRPO variant unless explicitly requested.
