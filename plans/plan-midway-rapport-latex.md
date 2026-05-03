# Plan: Midway Report in LaTeX / NeurIPS Template

## 0. Objective

Create a midway report for the DM887 GRPO for Control project using the official NeurIPS template or a local compatible copy.

The report should use the **full final-project structure**, but only fully develop the parts required for the midway/interim report:

1. related work,
2. MDP notation / formal setup,
3. PPO/SAC/TD3 baseline protocol and results/status.

The report should compile to PDF before the deadline.

---

## 1. Report location

Use:

```text
report-template/
├── main.tex
├── references.bib
├── sections/
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology.tex
│   ├── 04_theory.tex
│   ├── 05_experiments.tex
│   ├── 06_conclusion.tex
│   └── 07_ai_use_statement.tex
└── figures/
    └── midway_baselines.pdf
```

Copy generated figures from:

```text
figures/midway_baselines.pdf
```

to:

```text
report-template/figures/midway_baselines.pdf
```

or reference the original path if LaTeX compilation permits it.

---

## 2. Main LaTeX file skeleton

`main.tex` should look like this:

```latex
\documentclass{article}

% Use official NeurIPS style when available
\usepackage[preprint]{neurips_2025}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{algorithm}
\usepackage{algpseudocode}

\title{Interim Report: Group Relative Policy Optimization for Control}

\author{Jannik Guldmand \\
Department of Mathematics and Computer Science \\
University of Southern Denmark \\
\texttt{jagul24@student.sdu.dk}}

\begin{document}

\maketitle

\begin{abstract}
\input{sections/00_abstract}
\end{abstract}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_methodology}
\input{sections/04_theory}
\input{sections/05_experiments}
\input{sections/06_conclusion}
\input{sections/07_ai_use_statement}

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
```

Adjust the NeurIPS style filename to match the actual downloaded template.

---

## 3. Midway report structure

Recommended section structure:

```text
Abstract
1. Introduction
2. Related Work
   2.1 Proximal Policy Optimization and GAE
   2.2 Soft Actor-Critic
   2.3 Twin Delayed Deep Deterministic Policy Gradient
   2.4 Group Relative Policy Optimization
   2.5 Gap: GRPO for Continuous Control
3. Methodology
   3.1 Markov Decision Process Notation
   3.2 Policy, Return, and Value Functions
   3.3 Advantage Estimation
   3.4 Baseline Algorithms and Evaluation Protocol
   3.5 Planned GRPO-Control Extension
4. Theory
   4.1 Scope of Final Theoretical Analysis
5. Experiments
   5.1 Environments
   5.2 Baseline Implementation with ObjectRL
   5.3 Evaluation Metric and Seeds
   5.4 Interim Baseline Results
   5.5 Remaining Experimental Work
6. Conclusion and Next Steps
AI Use Statement
References
```

For the midway submission, sections 4 and the GRPO-specific parts may be short and forward-looking.

---

## 4. Writing priorities for today

### Must be strong

- Related work
- MDP notation
- Experiment protocol
- ObjectRL baseline setup
- AI use statement
- References

### Can be brief today

- Full GRPO algorithm design
- Theory/convergence analysis
- Final discussion
- Final conclusion

---

## 5. Suggested abstract draft

```latex
This interim report establishes the foundation for a project on adapting Group Relative Policy Optimization (GRPO) to continuous-control reinforcement learning tasks. The final project aims to design a GRPO variant that performs competitively with established continuous-control methods while retaining the group-relative advantage structure that motivates GRPO. The interim contribution is threefold. First, it reviews the main baseline methods used in the project: PPO with GAE, SAC, and TD3. Second, it introduces the Markov Decision Process notation, value functions, returns, and advantage estimates required to describe the later GRPO-control algorithm. Third, it defines and begins the baseline evaluation protocol using ObjectRL implementations of PPO, SAC, and TD3 on continuous Car Racing, cartpole-swingup, and acrobot-swingup environments. The final project will extend this foundation with the proposed GRPO-control algorithm, convergence analysis, and full five-seed empirical comparison.
```

Keep the abstract between 150 and 250 words.

Do not include citations in the abstract.

---

## 6. Introduction plan

The introduction should answer:

1. What is the problem?
2. Why does it matter?
3. Why is GRPO interesting?
4. Why is control different from LLM-style RL?
5. What does the interim report accomplish?
6. What remains for the final project?

Suggested structure:

```text
Paragraph 1: RL control and the need for stable policy optimization.
Paragraph 2: PPO/SAC/TD3 are strong baselines but represent different design families.
Paragraph 3: GRPO motivates a group-relative alternative to value-based advantage estimation.
Paragraph 4: Control tasks require sequential credit assignment and robust continuous-action learning.
Paragraph 5: This interim report establishes related work, notation, and baseline protocol.
```

Include a clear statement like:

```latex
This interim report focuses on the parts of the final project that must be established before the GRPO variant can be evaluated: related work, formal MDP notation, and the PPO/SAC/TD3 baseline protocol.
```

---

## 7. Related work plan

### 7.1 PPO and GAE

Discuss:

- PPO as on-policy policy-gradient method,
- clipped surrogate objective,
- practical stability,
- GAE as variance-reduced advantage estimation,
- relevance as baseline.

Possible references:

- Schulman et al., PPO
- Schulman et al., GAE

### 7.2 SAC

Discuss:

- off-policy actor-critic,
- entropy regularization,
- exploration,
- continuous control relevance.

Possible reference:

- Haarnoja et al., SAC

### 7.3 TD3

Discuss:

- deterministic policy gradient family,
- clipped double Q-learning,
- delayed policy updates,
- target policy smoothing,
- continuous control relevance.

Possible reference:

- Fujimoto et al., TD3

### 7.4 GRPO

Discuss:

- group-relative policy optimization,
- relative rewards/advantages within sampled groups,
- reduced dependence on a separate value model in LLM-style settings,
- challenge when moving to control.

Possible reference:

- DeepSeekMath / GRPO source paper, if used.

### 7.5 Gap

Key message:

```text
The central gap is that GRPO-style group-relative advantage estimation is not automatically sufficient for continuous-control tasks, where temporal credit assignment, action-space geometry, and value estimation strongly affect learning stability.
```

---

## 8. Methodology: MDP notation plan

This is a midway-critical section.

Use formal notation:

```latex
We model each task as a Markov Decision Process
\[
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0),
\]
where \(\mathcal{S}\) is the state space, \(\mathcal{A}\) is the action space, \(P(s'\mid s,a)\) is the transition kernel, \(r(s,a)\) is the reward function, \(\gamma \in [0,1)\) is the discount factor, and \(\rho_0\) is the initial-state distribution.
```

Define trajectory:

```latex
A trajectory is denoted
\[
\tau = (s_0,a_0,r_0,s_1,a_1,r_1,\ldots),
\]
where \(s_0 \sim \rho_0\), \(a_t \sim \pi_\theta(\cdot \mid s_t)\), and \(s_{t+1} \sim P(\cdot \mid s_t,a_t)\).
```

Define return:

```latex
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}.
```

Define value function:

```latex
V^{\pi}(s) = \mathbb{E}_{\pi}\left[G_t \mid s_t=s\right].
```

Define action-value:

```latex
Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[G_t \mid s_t=s, a_t=a\right].
```

Define advantage:

```latex
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s).
```

Define objective:

```latex
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right].
```

Then explain how PPO/GRPO differ primarily in how they estimate and use the advantage.

---

## 9. Methodology: baseline algorithms and protocol

Write:

```latex
The interim implementation uses ObjectRL implementations of PPO, SAC, and TD3. This follows the instructor clarification that these baselines do not need to be reimplemented from scratch. The purpose of the interim experiments is therefore not to establish novel baseline implementations, but to establish a reproducible evaluation protocol against which the later GRPO-control variant can be compared.
```

Protocol:

```text
Algorithms: PPO, SAC, TD3
Environments: continuous Car Racing, cartpole-swingup-v0, acrobot-swingup-v0
Seeds: 0, 1, 2, 3, 4
Metric: undiscounted evaluation episode return
X-axis: training steps before evaluation
Curves: mean across seeds, with variation if available
```

---

## 10. Theory section for midway

Do not try to fake a complete convergence proof today.

Write a short scope section:

```latex
The final report will contain a formal convergence-oriented analysis of the proposed GRPO-control variant. The analysis will first state assumptions on the MDP, policy class, boundedness of rewards, sampling process, and the approximation error between the group-relative advantage estimate and the true policy advantage. It will then characterize conditions under which the proposed update constitutes an approximate policy improvement step. Proofs will be placed in the appendix. Since the GRPO variant is not yet finalized at the interim stage, this report only fixes the notation required for the later theoretical analysis.
```

This is honest and useful.

---

## 11. Experiments section plan

### 11.1 Environments

Describe each environment briefly:

- continuous Car Racing: image/continuous control task from Gymnasium Farama Box2D,
- cartpole-swingup-v0: swing up and balance a pole,
- acrobot-swingup-v0: swing up an underactuated two-link system.

### 11.2 ObjectRL baseline implementation

Write:

```latex
The baseline runs use ObjectRL for PPO, SAC, and TD3. Each run is identified by algorithm, environment, seed, training budget, and evaluation interval. Logs are saved per run and processed into a tidy table with columns for algorithm, environment, seed, training step, and evaluation return.
```

### 11.3 Figure

Use:

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/midway_baselines.pdf}
    \caption{Interim baseline learning curves for PPO, SAC, and TD3. The x-axis shows the number of training steps before evaluation, and the y-axis shows undiscounted evaluation episode return. Curves are averaged across available seeds. Full five-seed runs will be completed for the final report.}
    \label{fig:midway-baselines}
\end{figure}
```

If no complete figure exists, use a table of run status instead.

### 11.4 Run status table

```latex
\begin{table}[t]
\centering
\caption{Interim baseline run matrix.}
\label{tab:baseline-status}
\begin{tabular}{llll}
\toprule
Algorithm & Environment & Seeds & Status \\
\midrule
PPO & Car Racing continuous & 0--4 & In progress \\
PPO & cartpole-swingup-v0 & 0--4 & In progress \\
PPO & acrobot-swingup-v0 & 0--4 & In progress \\
SAC & Car Racing continuous & 0--4 & In progress \\
SAC & cartpole-swingup-v0 & 0--4 & In progress \\
SAC & acrobot-swingup-v0 & 0--4 & In progress \\
TD3 & Car Racing continuous & 0--4 & In progress \\
TD3 & cartpole-swingup-v0 & 0--4 & In progress \\
TD3 & acrobot-swingup-v0 & 0--4 & In progress \\
\bottomrule
\end{tabular}
\end{table}
```

Replace “In progress” with actual status from the notebook.

---

## 12. Conclusion and next steps

Midway conclusion should not overclaim.

Suggested draft:

```latex
This interim report established the foundation for the final GRPO-control project. The related work situates the project relative to PPO, SAC, TD3, and GRPO. The methodology section fixed the MDP notation and value-function definitions needed to describe both the baselines and the later GRPO variant. The experimental section defined the ObjectRL-based baseline protocol for PPO, SAC, and TD3 on the required environments. The next steps are to complete the full baseline matrix, implement the GRPO-control variant, add the convergence-oriented analysis, and evaluate whether the proposed method can outperform PPO while remaining competitive with SAC and TD3.
```

---

## 13. AI use statement

Create `sections/07_ai_use_statement.tex`:

```latex
\section*{Use of AI Tools}
AI tools were used to assist with planning, code scaffolding, LaTeX structuring, and drafting of preliminary explanatory text. The final technical choices, verification of claims, implementation decisions, experiment interpretation, and submitted text remain the responsibility of the author.
```

---

## 14. Bibliography setup

Use BibTeX:

```latex
\bibliographystyle{plainnat}
\bibliography{references}
```

Suggested references are listed in `plan-references.md`.

Compile sequence:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

or use `latexmk`:

```bash
latexmk -pdf main.tex
```

---

## 15. Deadline checklist

Before upload:

- [ ] PDF compiles.
- [ ] Title says “Interim Report”.
- [ ] Related work is present.
- [ ] MDP notation is present.
- [ ] PPO, SAC, TD3 are all discussed.
- [ ] ObjectRL usage is stated.
- [ ] Continuous Car Racing clarification is reflected.
- [ ] Seeds are specified.
- [ ] Figure or status table is included.
- [ ] References compile.
- [ ] AI use statement is included.
- [ ] No placeholder text like “TODO” remains unless intentionally labelled as “planned final work”.
