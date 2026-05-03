# Plan: Scientific Writing Principles for the DM887 Midway Report and Thesis

## 0. Purpose

This file summarizes reusable scientific-writing principles for the DM887 midway report and later master thesis writing.

It is based on the uploaded lecture slides on scientific report writing and adapted to the current RL project.

---

## 1. Scientific report vs research paper vs thesis

A scientific report is a structured document that describes, analyzes, and evaluates a specific project or experiment. It may focus on practical work and experiments rather than requiring a fully original theoretical contribution.

A research paper aims to contribute to academic discourse and usually presents original research or a novel evaluation of existing research.

A master’s thesis is a larger independent research project. It demonstrates the ability to conduct independent research and contribute insights or perspectives to the field.

For DM887, the report is closest to a compact research-style project report:

```text
Problem → related work → method → experiments → analysis → discussion → conclusion
```

For the master thesis, the same structure applies but with more depth, more careful positioning, and stronger methodology.

---

## 2. High-level report structure

Use the following structure for experiment/project reports:

1. Abstract
2. Introduction
3. Related Work
4. Methodology / Materials and Methods
5. Theory, if required
6. Experiments / Results
7. Analysis / Discussion
8. Conclusion
9. References
10. Appendix, if needed

For DM887 final report, the required structure is:

1. Introduction
2. Related Work
3. Methodology
4. Theory
5. Experiments
6. Conclusion
7. References
8. Appendix with proofs

For the midway report, use the final structure but fill only the midway-relevant content fully.

---

## 3. The three C’s

Scientific writing should prioritize:

1. **Content** — what matters is included.
2. **Context** — the reader understands why it matters.
3. **Conclusion** — the message is clear and supported.

Applied to the RL project:

- Content: PPO/SAC/TD3/GRPO, MDP notation, environments, evaluation protocol.
- Context: why GRPO for control is interesting and non-trivial.
- Conclusion: what the interim report establishes and what remains.

---

## 4. Avoid filler

Do not write long textbook explanations.

Good:

```text
PPO is included as the on-policy baseline because its clipped surrogate objective is a standard stabilization mechanism for policy-gradient updates.
```

Too much filler:

```text
Reinforcement learning is a very broad and exciting field with many algorithms and many applications in robotics, games, control, optimization, and artificial intelligence. Over many years, researchers have developed many methods...
```

For a compact report, every paragraph should either:

- define something necessary,
- motivate the project,
- explain a design choice,
- report a result,
- interpret a result,
- state a limitation,
- state a next step.

---

## 5. Academic language

Use language that is:

- formal,
- objective,
- technical,
- precise,
- not pretentious.

Prefer:

```text
The algorithm improves performance iteratively by updating the policy using sampled trajectories.
```

Avoid:

```text
The algorithm gets smarter and smarter over time.
```

Avoid overclaiming:

```text
The preliminary runs suggest...
```

is safer than:

```text
The results prove...
```

unless the evidence actually proves the claim.

---

## 6. Section structure

Each section should usually follow this pattern:

1. Opening sentence: what the section does.
2. Background/definition.
3. Specific content.
4. Short summary or transition.

Example:

```text
This section defines the Markov Decision Process notation used throughout the report. Each control task is modelled as an MDP ... These definitions allow PPO, SAC, TD3, and the later GRPO variant to be described within a shared formal framework.
```

Do not create headings with no introductory text underneath.

---

## 7. Use sections and subsections carefully

Use sections and subsections for important structure.

Use subsubsections sparingly.

Instead of:

```text
3. Methodology
3.1 MDP
3.1.1 States
3.1.2 Actions
3.1.3 Rewards
```

prefer:

```text
3. Methodology
3.1 MDP notation
```

and define states, actions, and rewards inside prose.

---

## 8. Results vs discussion

Keep these separate.

### Results

Present what was observed.

```text
SAC reached a higher evaluation return than TD3 in the initial cartpole-swingup run.
```

### Discussion

Interpret what it means.

```text
This may reflect SAC's entropy-regularized exploration, although the result should not be overinterpreted before the full five-seed experiment is complete.
```

Do not interpret too aggressively in the results section.

---

## 9. Figures and tables

Figures and tables should be impactful.

Every figure should have:

- a clear caption,
- labelled axes,
- units where relevant,
- algorithm names,
- environment names,
- seed aggregation information,
- a reference in the text.

Bad:

```latex
\caption{Results.}
```

Good:

```latex
\caption{Interim baseline learning curves for PPO, SAC, and TD3. The x-axis shows the number of training steps before evaluation, and the y-axis shows undiscounted evaluation episode return. Curves are averaged across available seeds.}
```

Do not repeat all numeric values in prose if they are already in the figure. Explain the main pattern.

---

## 10. Discussion and conclusion

The discussion should answer:

- Did the experiment achieve its goal?
- What do the results imply?
- What are the limitations?
- What should be improved?

The conclusion should answer:

- What is the take-home message?
- What has been established?
- What remains?

Do not introduce new data in the conclusion.

---

## 11. Common mistakes to avoid

- No clear aim.
- Related work is just a list of papers.
- Introduction becomes a textbook chapter.
- Methods lack enough detail to reproduce.
- Results are disconnected from objectives.
- Figures lack useful captions.
- Discussion overclaims from weak evidence.
- Conclusion introduces new claims.
- References are inconsistent.
- Placeholder text remains.
- AI-generated text is not checked.

---

## 12. DM887-specific writing strategy

For the midway report:

- Be explicit that this is an interim report.
- Focus on foundations.
- Do not pretend the GRPO algorithm is finished.
- Report completed baseline runs honestly.
- If runs are partial, label them partial.
- State the exact next steps toward final submission.

For the final report:

- Convert interim status into completed baseline results.
- Add GRPO method.
- Add theory.
- Add final comparison.
- Add limitations and future work.
