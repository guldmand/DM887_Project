# Plan: Writing the Introduction

## 0. Purpose

The introduction should tell the reader:

1. what problem is studied,
2. why it matters,
3. what gap exists,
4. what this report contributes,
5. how the report is structured.

For the midway report, the introduction should be honest about interim scope.

---

## 1. Introduction ingredients

A strong introduction contains:

- background,
- related-work framing,
- motivation,
- statement of purpose,
- contribution summary,
- report structure.

---

## 2. Paragraph plan for DM887 midway

### Paragraph 1 — RL control context

Explain that continuous-control RL requires stable policy optimization and reliable evaluation.

### Paragraph 2 — Existing baselines

Introduce PPO, SAC, and TD3 as standard baselines representing different families:

- PPO: on-policy stochastic policy optimization,
- SAC: off-policy maximum-entropy actor-critic,
- TD3: deterministic actor-critic for continuous control.

### Paragraph 3 — Why GRPO is interesting

Explain that GRPO uses group-relative information and may reduce dependence on conventional value estimation, but control tasks create different challenges.

### Paragraph 4 — Interim scope

State clearly:

```text
This interim report does not present the final GRPO-control algorithm. Instead, it establishes the literature foundation, notation, and baseline protocol required for the final comparison.
```

### Paragraph 5 — Structure

Briefly describe the report structure.

---

## 3. Suggested contribution statement

```latex
The contribution of this interim report is threefold. First, it positions the project relative to PPO, SAC, TD3, and GRPO. Second, it defines the MDP notation, return, value functions, and advantage estimates required for the final method. Third, it establishes an ObjectRL-based baseline protocol for PPO, SAC, and TD3 on the required control environments.
```

---

## 4. Common mistakes

Avoid:

- writing a generic RL textbook introduction,
- delaying the project aim until the end,
- making unsupported claims about GRPO superiority,
- hiding the fact that this is interim work,
- making the introduction too long.

For this report, the introduction should be roughly 0.5–1 page.

---

## 5. Thesis reuse

For the master thesis, use the same introduction shape:

1. broad context: decision support for stock investment,
2. problem: risk-aware sequential decisions under uncertainty,
3. gap: black-box trading bots vs interpretable DSS,
4. aim: RL-based decision support with risk profile/strategy abstraction,
5. contribution: architecture, data pipeline, evaluation, risk-aware RL,
6. structure.
