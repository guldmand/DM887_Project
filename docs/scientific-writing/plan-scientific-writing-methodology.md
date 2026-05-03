# Plan: Writing the Methodology Section

## 0. Purpose

The methodology section should make the work understandable and reproducible.

For the DM887 midway report, methodology must especially define the MDP notation and baseline evaluation setup.

---

## 1. What methodology must include

For this project:

1. formal MDP setup,
2. trajectory and return definitions,
3. value and advantage definitions,
4. baseline algorithm roles,
5. environment/evaluation protocol,
6. planned GRPO-control extension.

---

## 2. MDP notation subsection

Start with:

```latex
This section defines the notation used to describe the baseline algorithms and the later GRPO-control variant.
```

Then define:

```latex
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0).
```

Explain each element in one sentence.

---

## 3. Policy and trajectory subsection

Define:

```latex
\pi_\theta(a \mid s)
```

for stochastic policies and optionally:

```latex
\mu_\theta(s)
```

for deterministic policies such as TD3.

Define trajectory:

```latex
\tau = (s_0,a_0,r_0,s_1,a_1,r_1,\ldots).
```

---

## 4. Return and value functions

Define:

```latex
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}
```

```latex
V^{\pi}(s) = \mathbb{E}_{\pi}[G_t \mid s_t=s]
```

```latex
Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t \mid s_t=s,a_t=a]
```

```latex
A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)
```

Explain that advantage estimation is central because PPO and GRPO use advantage-like signals to update policies.

---

## 5. Baseline protocol subsection

Explain:

- PPO, SAC, TD3 are used as baselines,
- implementations come from ObjectRL,
- experiments are run online,
- evaluation is performed at regular intervals,
- metric is undiscounted evaluation episode return,
- seeds are `{0,1,2,3,4}`.

---

## 6. Planned GRPO-control extension

Keep this short for midway:

```text
The final method will modify GRPO for continuous control by combining group-relative advantage normalization with a value-function or multi-step temporal-difference component. This is motivated by the need for temporal credit assignment in control tasks.
```

Do not include fake pseudocode unless you have decided the algorithm.

---

## 7. Common methodology mistakes

Avoid:

- only saying “we used ObjectRL” without explaining the evaluation protocol,
- writing excessive theory unrelated to the project,
- omitting seeds,
- omitting metric,
- omitting environment names,
- mixing results into methodology.

---

## 8. Reproducibility checklist

- [ ] Algorithms named.
- [ ] Environments named.
- [ ] Seeds listed.
- [ ] Evaluation metric defined.
- [ ] Training/evaluation distinction clear.
- [ ] Implementation framework stated.
- [ ] Output format described.
- [ ] Figure generation described.
