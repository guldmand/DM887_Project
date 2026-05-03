# Course Materials for DM887 GRPO Project

This file documents the local course material used as background for the DM887 GRPO for Control project.

The actual course slide PDFs are stored locally in:

```text
slides/
```

The `slides/` folder is intentionally Git-ignored because it may contain course-restricted material.

This file is safe to commit. It documents what is available locally and how the material should be used.

---

## 1. Local slides folder

Current local slide folder:

```text
slides/
```

Known files:

```text
slides/
├── 1_Basic_Concepts.pdf
├── 2_Markov_Decision_Processes.pdf
├── 2026-03-04-distrl_slides.pdf
├── 3_Model_Based_RL.pdf
├── 4_Model_Free_RL.pdf
├── 5_Policy_Gradient_Methods.pdf
├── 6_Conservative_Policy_Iteration.pdf
├── DM887___Stochastic_Processes.pdf
├── DM887_Convergence.pdf
├── MARL-SLIDES.pdf
├── Proofs.pdf
├── RL_GuestLecture.pdf
└── walds-identity.pdf
```

---

## 2. How the slides should be used

The slides are local background material for:

1. understanding RL notation,
2. writing the MDP methodology section,
3. framing policy-gradient methods,
4. understanding conservative policy iteration,
5. preparing the later theory/convergence section.

The slides should not be copied verbatim into the report.

Use them to guide understanding and then cite primary literature where possible.

---

## 3. Most relevant slides for the midway report

| Slide file | Use for midway report |
|---|---|
| `1_Basic_Concepts.pdf` | Basic RL terminology, agent/environment loop, reward, return |
| `2_Markov_Decision_Processes.pdf` | Formal MDP notation, states, actions, transitions, rewards |
| `4_Model_Free_RL.pdf` | Model-free RL framing and baseline context |
| `5_Policy_Gradient_Methods.pdf` | PPO, policy gradients, advantage-based updates |
| `6_Conservative_Policy_Iteration.pdf` | Conservative policy improvement, policy update foundations |
| `DM887_Convergence.pdf` | Useful later for final theory/convergence section |
| `Proofs.pdf` | Useful later for appendix/proof-writing style |
| `2026-03-04-distrl_slides.pdf` | Distributional RL context if relevant to later work |
| `RL_GuestLecture.pdf` | Supplementary context, likely not central for midway |

---

## 4. Recommended use by report section

### Related work

Use the slides only as orientation.

Cite primary papers instead:

- PPO
- GAE
- SAC
- TD3
- TRPO
- conservative policy iteration
- GRPO

### Methodology

The most relevant slides are:

```text
2_Markov_Decision_Processes.pdf
5_Policy_Gradient_Methods.pdf
```

Use them to ensure consistent notation for:

- MDP,
- policy,
- trajectory,
- return,
- value function,
- action-value function,
- advantage function,
- policy objective.

### Experiments

Use the official project description, ObjectRL documentation/paper, Gymnasium, and DMC references.

### Theory

For the final report, use:

```text
6_Conservative_Policy_Iteration.pdf
DM887_Convergence.pdf
Proofs.pdf
```

The midway report should not fake a complete proof. It should only establish notation and describe the final-theory plan.

---

## 5. Agent guidance

AI agents may inspect the local `slides/` folder if present.

However:

- Do not copy slide text verbatim into the report.
- Do not commit slide PDFs.
- Do not cite slides as primary research papers unless explicitly needed.
- Prefer citing primary papers in `docs/references/reading-list.md`.
- Use slides mainly to align notation and course expectations.

---

## 6. Git policy

These folders are local and should remain Git-ignored:

```text
papers/
slides/
```

These documentation files should be committed:

```text
docs/references/reading-list.md
docs/course-materials.md
```

---

## 7. Immediate use for the midway deadline

For the current midway deadline, the slides are useful mainly for:

1. writing the MDP notation section,
2. writing the PPO/policy-gradient background,
3. framing conservative policy improvement,
4. avoiding notation mismatches with the course.

Do not let slide reading block the PoC notebook work. Use them as support, not as a separate reading project.
