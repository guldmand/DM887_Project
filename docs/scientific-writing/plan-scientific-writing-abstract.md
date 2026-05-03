# Plan: Writing the Abstract

## 0. Purpose of an abstract

An abstract is a concise summary of the report. It helps readers quickly decide whether the work is relevant and communicates the key contribution to readers who may not read the full text.

For the DM887 midway report, the abstract should summarize:

1. the final project goal,
2. the interim scope,
3. the methods/baselines,
4. the current status,
5. the next step.

---

## 1. Recommended abstract structure

Use 5 compact sentences:

1. Context/problem.
2. Final project aim.
3. Interim contribution.
4. Baseline/evaluation setup.
5. Next step/final extension.

Template:

```text
Group Relative Policy Optimization (GRPO) has recently attracted attention in reinforcement learning from human or preference feedback, but its role in continuous-control tasks remains less established. This project studies whether a GRPO-style policy optimization method can be adapted to robotic/control environments and evaluated against standard continuous-control baselines. This interim report establishes the related work, formal Markov Decision Process notation, and baseline evaluation protocol needed for the final project. The baseline protocol uses ObjectRL implementations of PPO, SAC, and TD3 on continuous Car Racing, cartpole-swingup-v0, and acrobot-swingup-v0, with evaluation based on undiscounted episode return across fixed seeds. The final report will extend this foundation with the proposed GRPO-control variant, convergence-oriented analysis, and full empirical comparison.
```

---

## 2. Abstract checklist

- [ ] 50–300 words.
- [ ] No citations.
- [ ] No unexplained details.
- [ ] No claims not supported in the report.
- [ ] Mentions interim scope.
- [ ] Mentions final direction.
- [ ] Does not read like an introduction.
- [ ] Does not read like a conclusion only.

---

## 3. Common mistakes

Avoid:

- making the abstract too long,
- copy-pasting from the introduction,
- including citations,
- claiming final results before they exist,
- writing only background and no contribution,
- writing only future work and no current work.

---

## 4. For the master thesis

The same pattern can be reused:

1. broader problem,
2. thesis aim,
3. method/system,
4. evaluation,
5. contribution and implication.

For the thesis, the abstract should be written last, after the results and conclusion are stable.
