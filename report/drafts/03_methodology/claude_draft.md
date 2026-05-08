# Methodology — Inspiration Draft (claude)

This document is **inspiration material** for the Methodology section of the
DM887 midway report. It is not final report text. The final wording lives in
`report/sections/03_methodology.tex` and must be written manually.

Project: *Group Relative Policy Optimization for Control* (DM887, Spring 2026).
Stage: midway / interim report.

The local theory notebooks under `theory/` are the **primary notation
source** for this section. The Methodology section should not invent new
notation that conflicts with those notebooks.

---

## 1. Methodology section goal analysis

In this **midway** report the Methodology section has to do four jobs and it
has to do them under the constraint that GRPO has not yet been implemented:

1. **Fix the formal RL setup.** Define the MDP, trajectories, returns,
   stochastic and deterministic policies, value and action-value functions,
   and the advantage. The notation must be consistent with the local theory
   notebooks (`theory/0_1_notation_in_rl.ipynb`,
   `theory/3_markov_decision_processes.ipynb`,
   `theory/4_discounted_decision_processes.ipynb`,
   `theory/5_episodic_markov_decision_processes.ipynb`).
2. **State the experimental methodology.** Identify the algorithms (PPO, SAC,
   TD3) by their role as baselines, the environments by short names, and the
   evaluation metric (undiscounted evaluation episode return). Methodology
   should *fix* the protocol; Experiments will *report* on it.
3. **Document the two implementation paths.** Vector environments are run
   through ObjectRL via a project-side bridge; CarRacing is run through a
   project-side PyTorch CNN pipeline. The reader must understand that this
   is a deliberate consequence of ObjectRL's 1-D Box assumption and the
   no-modification policy on `external/objectrl/`.
4. **Prepare the GRPO-control extension.** Forward-reference the proposed
   variant in one short paragraph, stating that the methodology is shared
   so that the final stage's comparison is on equal footing.

What this section should explicitly **not** do at the midway stage:

- It must not contain the GRPO-control algorithm in detail.
- It must not include theorems, proofs, or convergence arguments — those
  belong in Theory and the Appendix.
- It must not duplicate the experimental matrix listing, the result
  validation phrasing, or the per-environment baseline results — those are
  already in `report/sections/05_experiments.tex`.
- It must not repeat textbook material (e.g., a full Bellman derivation)
  that the assignment does not require.

Length target: roughly **1.0–1.5 pages** in the NeurIPS template. Keep it
tight. The Theory and Experiments sections will carry the heavier weight.

---

## 2. Consistency check against existing Introduction and Experiments

The Introduction and Experiments sections are already drafted in the LaTeX
tree. Methodology must thread between them without duplication.

**Already established in `report/sections/01_introduction.tex`:**

- The project investigates GRPO for control, using PPO, SAC, TD3 as
  baselines (citations: `shao2024deepseekmath`, `schulman2017proximal`,
  `haarnoja2018sacapps`, `fujimoto2018td3`).
- The midway scope is foundation, not GRPO results.
- The matrix is 3 algos × 3 environments × 5 seeds = 45 runs and 900
  evaluation rows.
- Vector environments use ObjectRL (`baykal2025objectrl`); CarRacing uses
  project-side CNN implementations because the image observations do not
  fit ObjectRL's vector-observation workflow (`towers2024gymnasium`).
- No HPO, midway budgets validate the pipeline rather than asymptotic
  performance.

**Already established in `report/sections/05_experiments.tex`:**

- Subsections already present: Experimental Matrix, Environments and
  Implementation Paths, Evaluation Protocol, Result Validation, Baseline
  Results, Stability and Variation, Limitations, Transition to the Final
  GRPO Stage.
- Already defines a discounted state-value function inline:
  `\(V^\pi_\gamma(s) = \mathbb{E}^\pi\!\big[\sum_{t=0}^\infty \gamma^t
  r(s_t,a_t) \mid s_0 = s\big]\)`.
- Already states 20\,000-step vector-control budget, 10\,000-step CarRacing
  budget, undiscounted evaluation episode return as the metric.
- Already describes the 45-run matrix and the validation step, and labels
  `\label{sec:experimental-matrix}`, `\label{sec:implementation-paths}`,
  `\label{sec:evaluation-protocol}`, `\label{sec:result-validation}`,
  `\label{sec:baseline-results}`, `\label{sec:stability-variation}`,
  `\label{sec:experiment-limitations}`, `\label{sec:transition-final-grpo}`.

**What Methodology should briefly *repeat* (one sentence each, for a
self-contained section):**

- Algorithm names (PPO, SAC, TD3) and the fact that they serve as the
  comparison surface.
- Environment short names.
- The split between ObjectRL and the project-side CNN — but only to
  motivate the *modeling* choice (vector vs. image observations), not to
  re-state the file paths.

**What Methodology should *not* repeat:**

- The 45-run matrix counting and the 900-row validation. This belongs in
  Experiments and is already there.
- Per-environment baseline results, stability tables, or rankings.
- Concrete training-step budgets and eval cadences. The Experiments
  section already pins these down. Methodology can briefly cite the
  Experiments section's protocol via `\ref{sec:evaluation-protocol}`.
- The figure generation pipeline.

**What Methodology must clarify *before* the Experiments section can be
fully understood:**

- The MDP tuple notation that the rest of the report uses.
- The notation for stochastic vs. deterministic policies. The Experiments
  section refers to the algorithms but does not formally distinguish PPO
  / SAC (stochastic) from TD3 (deterministic).
- The training objective (discounted) versus the evaluation metric
  (undiscounted episode return). Experiments alludes to this distinction
  but does not pin it down formally.
- The advantage function $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$, which is
  needed both for PPO/SAC/TD3 framing and for the eventual GRPO discussion.

**Notation drift to fix.** The Experiments section already uses lowercase
$s_t, a_t, r$ and $\sum_{t=0}^\infty \gamma^t r(s_t,a_t)$. The local
notation notebook (`theory/0_1_notation_in_rl.ipynb`) prefers uppercase
random-variable form: $S_t, A_t, R_{t+1}$ and
$G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$. The Methodology section
should pick **one convention and stick to it for the whole report**. See
Section 4 of this draft for a recommendation.

---

## 3. Proposed subsection structure (5–6 subsections)

Suggested layout for the midway Methodology section.

### 3.1 Problem formulation as an MDP
- **Purpose.** Define the MDP tuple and the components used in the rest of
  the report.
- **Key message.** Each of the three project tasks (`cartpole_swingup`,
  `acrobot_swingup`, `car_racing_continuous`) is treated as a discrete-time
  MDP with continuous action spaces; the differences between them are
  surface (observation type, reward scale) rather than structural.
- **Connects to.** `theory/3_markov_decision_processes.ipynb`,
  `theory/0_1_notation_in_rl.ipynb`, `scripts/project_envs.py`
  (which actually constructs the environments).
- **Midway / final.** Midway. Wording is reusable in the final report.

### 3.2 Policies, trajectories, and returns
- **Purpose.** Define $\pi_\theta$ for stochastic policies (PPO, SAC),
  $\mu_\theta$ for deterministic policies (TD3), the trajectory, and both
  the discounted return $G_t$ used as the *training* signal and the
  undiscounted episodic return used as the *evaluation* metric.
- **Key message.** PPO/SAC and TD3 differ in the form of policy
  representation; both are evaluated under the same protocol. The
  evaluation metric is intentionally undiscounted episode return.
- **Connects to.** `theory/0_1_notation_in_rl.ipynb`,
  `theory/4_discounted_decision_processes.ipynb`,
  `theory/5_episodic_markov_decision_processes.ipynb`,
  `report/sections/05_experiments.tex` (training-vs-evaluation distinction).
- **Midway / final.** Midway.

### 3.3 Value functions and advantage
- **Purpose.** Introduce $V^\pi$, $Q^\pi$, and $A^\pi$ in their standard
  forms.
- **Key message.** PPO uses advantage estimates for policy updates;
  GRPO replaces the value-baseline portion of the advantage with a group-
  relative quantity, which is what motivates the project.
- **Connects to.** `theory/0_1_notation_in_rl.ipynb`,
  `theory/3_markov_decision_processes.ipynb`,
  `theory/4_discounted_decision_processes.ipynb`. Forward-references
  Theory (Section 4).
- **Midway / final.** Midway. Theory section will sharpen these
  definitions in the final report.

### 3.4 Baseline algorithms and their methodological role
- **Purpose.** Identify each baseline by family and role, *not* repeat
  algorithm details that belong in Related Work.
- **Key message.** PPO is the on-policy stochastic baseline; SAC is the
  off-policy maximum-entropy baseline; TD3 is the off-policy deterministic
  baseline. Their role here is to define the comparison surface.
- **Connects to.** `report/sections/02_related_work.tex` (forward-ref),
  Experiments (`\ref{sec:experimental-matrix}`).
- **Midway / final.** Midway. The final report adds GRPO as a fourth
  algorithm.

### 3.5 Environment and implementation pipeline
- **Purpose.** State the modeling reason for the two-track implementation
  (ObjectRL for vector envs, project-side CNN for CarRacing) without
  duplicating the file-path-level detail of Experiments.
- **Key message.** ObjectRL's actor/critic architectures expect 1-D
  vector observations, which is a natural fit for the two DM Control
  swing-up tasks. CarRacing returns $(96,96,3)$ uint8 image observations
  and therefore requires a CNN-based policy; the project ships PPO, SAC,
  and TD3 with a shared CNN trunk for that environment. No code under
  `external/objectrl/` is modified in either path.
- **Connects to.** `scripts/project_envs.py`,
  `scripts/run_project_objectrl_baseline.py`,
  `scripts/run_carracing_cnn_baseline.py`,
  `scripts/carracing_cnn.py`, and the Experiments subsection
  `\ref{sec:implementation-paths}`.
- **Midway / final.** Midway. The final report can mention any GRPO-
  specific implementation choice here.

### 3.6 Planned GRPO-control extension
- **Purpose.** Forward-reference the final-stage method without
  pretending it is implemented.
- **Key message.** The methodology is fixed so that, in the final stage,
  GRPO can be added as a fourth algorithm under the same MDP setup, the
  same evaluation protocol, and the same seed structure. The detailed
  algorithm and theoretical analysis live in the Theory section
  (forward-ref).
- **Connects to.** `report/sections/04_theory.tex`,
  `\ref{sec:transition-final-grpo}`.
- **Midway / final.** Midway. Becomes a real subsection in the final
  report.

(Optional, can be folded into 3.4 if length is tight.)

---

## 4. Formal notation proposal

The following notation is proposed for use throughout the report. It is
chosen to **match `theory/0_1_notation_in_rl.ipynb` whenever possible**,
and to be compatible with the Sutton–Barto convention used in the local
theory notebooks. Where the existing Experiments section already uses a
lowercase realisation form, the Methodology section can introduce both
forms briefly and then commit to one for the rest of the report.

### MDP tuple
For an infinite-horizon discounted MDP (the natural choice for the
training objective):
$$
\mathcal{M} \;=\; (\mathcal{S},\,\mathcal{A},\,P,\,r,\,\gamma,\,\rho_0).
$$
- $\mathcal{S}$: state space (continuous, possibly high-dimensional).
- $\mathcal{A}$: action space (continuous in this project; CarRacing has
  three continuous channels, the swing-up tasks have one).
- $P(s' \mid s,a)$: transition kernel.
- $r(s,a)$: bounded reward function. Use $R_{\max}$ as its sup norm.
- $\gamma \in [0,1)$: discount factor used in the *training* objective
  (the actual numeric value follows the chosen ObjectRL/algorithm
  defaults; defer the value to Experiments).
- $\rho_0$: initial-state distribution.

For the episodic statement (used to define the evaluation metric), see the
`theory/5_episodic_markov_decision_processes.ipynb` framing.

### Random variables vs. realisations

The notation notebook uses uppercase random variables $S_t, A_t, R_{t+1}$.
The Experiments section already in the report uses lowercase $s_t, a_t,
r$. Recommendation: **use lowercase $s_t, a_t, r_t$ throughout the
report**, and write conditional expectations as
$\mathbb{E}^\pi[\,\cdot \mid s_t = s, a_t = a\,]$. This matches Experiments
without contradicting the notation notebook (which presents both forms).
Mention the choice once in Section 3.1 of Methodology.

### Trajectory
$$
\tau \;=\; (s_0, a_0, r_0, s_1, a_1, r_1, \ldots),
\qquad s_0 \sim \rho_0,\; a_t \sim \pi(\cdot \mid s_t),\;
s_{t+1} \sim P(\cdot \mid s_t, a_t).
$$

### Policies
- Stochastic policy: $\pi_\theta(a \mid s)$, used by PPO and SAC.
- Deterministic policy: $\mu_\theta(s)$, used by TD3.
- When a single symbol is convenient, use $\pi_\theta$ as the umbrella
  notation and specialise to $\mu_\theta$ only where needed.

### Returns
- Discounted return (training-side):
  $$
  G_t \;=\; \sum_{k=0}^{\infty} \gamma^k \, r_{t+k}.
  $$
- Episodic undiscounted evaluation return:
  $$
  J^{\mathrm{eval}}(\pi)
  \;=\;
  \mathbb{E}^{\pi}\!\left[\,\sum_{t=0}^{T-1} r_t \,\right],
  $$
  where $T$ is the (random) episode termination time. This is the metric
  reported in the experiments. The training objective and the evaluation
  metric are deliberately distinct.

### Value, action-value, advantage
$$
V^\pi(s) \;=\; \mathbb{E}^\pi[\,G_t \mid s_t = s\,],
\qquad
Q^\pi(s,a) \;=\; \mathbb{E}^\pi[\,G_t \mid s_t = s,\, a_t = a\,],
\qquad
A^\pi(s,a) \;=\; Q^\pi(s,a) - V^\pi(s).
$$

### Empirical evaluation across seeds
For algorithm $\alpha$ on environment $e$ at training step $k$, with seed
set $\mathcal{Z}$ (here $\mathcal{Z} = \{0,1,2,3,4\}$) and $M$ evaluation
episodes per snapshot (here $M = 3$), denote the per-seed mean evaluation
return by $\hat{J}_{\alpha,e,k,z}$ and the cross-seed mean by
$$
\bar{J}_{\alpha,e,k}
\;=\;
\frac{1}{|\mathcal{Z}|} \sum_{z \in \mathcal{Z}} \hat{J}_{\alpha,e,k,z}.
$$
This makes the curve-aggregation step in
`scripts/summarize_project_baselines.py` formal. (Optional: include this
only if Experiments would otherwise have nowhere to anchor the
aggregation.)

### Run identifiers
Use $(\alpha, e, z)$ for an algorithm–environment–seed run, so that the
midway matrix is the cross-product
$\mathcal{A}_{\mathrm{algo}} \times \mathcal{E}_{\mathrm{env}} \times
\mathcal{Z}$ with $|\mathcal{A}_{\mathrm{algo}}| = 3$,
$|\mathcal{E}_{\mathrm{env}}| = 3$, $|\mathcal{Z}| = 5$.

---

## 5. Safe claims

These claims are safe to make in the Methodology section.

**Setup**

- Each project task is modelled as an infinite-horizon discounted MDP for
  training and evaluated by undiscounted episodic return.
- The action spaces are continuous for all three environments.
- The observation spaces differ: `cartpole_swingup` and `acrobot_swingup`
  are vector observations (DM Control swing-up tasks flattened into 1-D
  float32 vectors via `DMCGymAdapter` in `scripts/project_envs.py`);
  `car_racing_continuous` is `(96, 96, 3)` uint8 images from
  `gymnasium.make("CarRacing-v3", continuous=True)`.
- PPO and SAC are implemented as stochastic-policy methods; TD3 is
  implemented as a deterministic-policy method.

**Implementation paths**

- The vector-environment baselines reuse PPO, SAC, and TD3 from ObjectRL
  via `scripts/run_project_objectrl_baseline.py`, which monkey-patches
  the `make_env` symbol that
  `objectrl.experiments.base_experiment` has already imported. No code
  under `external/objectrl/` is modified.
- The CarRacing baselines use a project-side PyTorch implementation of
  PPO, SAC, and TD3 with a shared CNN trunk in
  `scripts/carracing_cnn.py`, driven by
  `scripts/run_carracing_cnn_baseline.py`. This path exists because
  `objectrl.agents.base_agent` asserts 1-D `Box` observations.
- The CarRacing CNN runs were executed on Google Colab with CUDA and
  copied back into the local repository so that all baselines share the
  same result-pipeline layout.

**Protocol (one sentence each, the rest belongs in Experiments)**

- The evaluation metric is the undiscounted evaluation episode return.
- Five seeds (0..4) are used per algorithm–environment combination.
- Training-step budgets, evaluation cadence, warm-up, and learn-frequency
  are pinned down in the Experiments section
  (`\ref{sec:evaluation-protocol}`).

**Cautious wording**

- "At the midway stage, the methodology fixes the notation, the
  evaluation protocol, and the implementation pipeline."
- "The current implementation validates the pipeline rather than final
  algorithmic superiority."
- "The GRPO-control variant is planned for the final stage."
- "The baseline protocol establishes the comparison surface against
  which the final GRPO method will be evaluated."

---

## 6. Claims to avoid

Do **not** write any of the following in the Methodology section:

- "We propose the GRPO-control algorithm and prove its convergence."
- "Algorithm $X$ is methodologically superior."
- "We perform a hyperparameter sweep over $\eta, \tau, \epsilon$."
- "The midway baselines are converged."
- "PPO/SAC/TD3 were re-implemented from scratch."
- "CarRacing was deferred to the final report."
- "PPO-CNN was not implemented" or "TD3-CNN was not implemented."
- "We modify ObjectRL to support image observations."
- "Our notation differs from the standard convention" (the local theory
  notebooks *are* the standard for this report; align with them instead).
- Any statement that introduces a theorem or proof in this section. Theory
  belongs in `report/sections/04_theory.tex`.

---

## 7. Implementation facts to extract

These facts are directly visible in the repository.

**Environment construction (`scripts/project_envs.py`)**

- `PROJECT_ENV_NAMES = ("car_racing_continuous", "cartpole_swingup",
  "acrobot_swingup")`.
- `cartpole_swingup` and `acrobot_swingup` are loaded via
  `dm_control.suite.load(domain_name, "swingup")` and wrapped by
  `DMCGymAdapter`, which flattens the observation `OrderedDict` into a
  1-D float32 vector and exposes Gymnasium-style `reset(seed=...)` and
  `step(action) -> (obs, reward, terminated, truncated, info)`.
- `car_racing_continuous` is `gymnasium.make("CarRacing-v3",
  continuous=True)`.

**ObjectRL bridge (`scripts/run_project_objectrl_baseline.py`)**

- `_patch_objectrl_make_env()` rebinds
  `objectrl.experiments.base_experiment.make_env` to
  `_make_project_env_for_objectrl`, which uses
  `make_project_env(...)` from `scripts/project_envs.py` and applies
  ObjectRL-style wrapping (`RescaleAction` to $[-1, 1]$, seeded reset).
- The patch is idempotent and does not modify any file under
  `external/objectrl/`.
- Algorithms are PPO, SAC, TD3 from ObjectRL.
- Eval seed offset: `effective_seed = seed + (100 if eval_env else 0)`.

**CarRacing CNN (`scripts/carracing_cnn.py`,
`scripts/run_carracing_cnn_baseline.py`)**

- `CarRacingCNNWrapper`: HWC uint8 → CHW float32 in $[0,1]$. Output
  observation space:
  `Box(0.0, 1.0, (3, 96, 96), dtype=float32)`.
- A small Atari-style 3-conv CNN feature extractor is shared across the
  actor and critic.
- Implementations of PPO-CNN, SAC-CNN, and TD3-CNN exist in the project
  side; this is the runner used for the CarRacing midway baselines.
- Runner-side budgets for the midway mode are pinned in
  `RUN_CONFIG["midway"]` (CHECK SCRIPT for the exact numbers; the
  Experiments section already contains them).

**Aggregation (`scripts/summarize_project_baselines.py`)**

- Expected combinations:
  $\{(\alpha, e, z) : \alpha \in \{\mathrm{ppo}, \mathrm{sac},
  \mathrm{td3}\},\, e \in \{\mathrm{cartpole\_swingup},
  \mathrm{acrobot\_swingup}, \mathrm{car\_racing\_continuous}\},\,
  z \in \{0,1,2,3,4\}\}$, total 45.
- Aggregation groups by `(algorithm, project_env, seed, train_step,
  status)` and reports the mean `eval_return` and `n_repeats`.
- Plots draw mean curves and shade a $\pm$std band when at least two
  seeds contribute (CHECK NOTEBOOK for the exact phrasing if the
  notebook overrides the script's labels).

**Reproducibility surface that Methodology can mention**

- A single CSV per run is written under
  `results/processed/project_baselines/`.
- The aggregated summary CSV is
  `results/processed/project_baselines/midway_vector_summary.csv`.
- The report-facing notebook
  `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` validates these
  artefacts (CHECK NOTEBOOK for the exact validation cells).

---

## 8. Danish draft

> **Bemærk.** Akademisk sprog, men læsbart. Brug dette som udgangspunkt og
> redigér selv. Citationer er sat med `\citep{...}`. Notation følger
> `theory/0_1_notation_in_rl.ipynb`.

**3 Metodologi**

**3.1 Problemformulering som MDP.**
Hver af projektets tre opgaver — `cartpole_swingup`, `acrobot_swingup` og
`car_racing_continuous` — modelleres som en uendelig-horisont diskonteret
Markov-beslutningsproces (MDP)
$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0),
$$
hvor $\mathcal{S}$ er tilstandsrummet, $\mathcal{A}$ er handlingsrummet
($\mathcal{A} \subseteq \mathbb{R}^d$ for et miljøafhængigt $d$),
$P(s' \mid s, a)$ er overgangskernen, $r(s, a)$ er en begrænset
belønningsfunktion, $\gamma \in [0, 1)$ er diskonteringsfaktoren brugt i
træningsformuleringen, og $\rho_0$ er fordelingen over starttilstande. De
to swing-up-opgaver bruger 1-D vektor-observationer, mens
`car_racing_continuous` bruger billed-observationer; den strukturelle
MDP-formulering er den samme. Trajektorier noteres
$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ med
$s_0 \sim \rho_0$, $a_t \sim \pi(\cdot \mid s_t)$ og
$s_{t+1} \sim P(\cdot \mid s_t, a_t)$.

**3.2 Politikker, returns og evaluerings-metrik.**
PPO og SAC repræsenterer politikken som en stokastisk afbildning
$\pi_\theta(a \mid s)$. TD3 anvender en deterministisk politik
$\mu_\theta(s)$. Den diskonterede return brugt i træningsformuleringen er
$$
G_t = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k},
$$
og den tilsvarende træningsmålsætning er
$J(\theta) = \mathbb{E}^{\pi_\theta}[G_0]$.
Evaluerings-metrikken er bevidst forskellig: den udiskonterede
episode-return
$$
J^{\mathrm{eval}}(\pi) = \mathbb{E}^{\pi}\!\left[\sum_{t=0}^{T-1} r_t\right],
$$
hvor $T$ er den (stokastiske) episode-termineringstid. Det er denne
udiskonterede episode-return, der rapporteres i Eksperiments-afsnittet
(jf. \ref{sec:evaluation-protocol}).

**3.3 Værdifunktioner og fordel.**
Standardværdifunktionerne under en politik $\pi$ er
$$
V^\pi(s) = \mathbb{E}^\pi[\,G_t \mid s_t = s\,],
\qquad
Q^\pi(s, a) = \mathbb{E}^\pi[\,G_t \mid s_t = s,\, a_t = a\,],
$$
og fordelsfunktionen
$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s).
$$
PPO og SAC opdaterer politikken ved hjælp af estimater af $A^\pi$ og
$Q^\pi$; TD3 baserer sin politikopdatering på en lært
handlingsværdifunktion. Det er disse størrelser, som projektets endelige
GRPO-control-variant vil modificere ved at erstatte den lærte
værdi-baseline med en gruppe-relativ størrelse.

**3.4 Baseline-algoritmer og deres metodologiske rolle.**
PPO \citep{schulman2017proximal} indgår som on-policy-stokastisk baseline,
SAC \citep{haarnoja2018sacapps} som off-policy maksimal-entropi baseline,
og TD3 \citep{fujimoto2018td3} som off-policy deterministisk baseline.
Algoritmerne er ikke det metodiske bidrag i denne midtvejsrapport; deres
rolle er at definere sammenligningsfladen for den endelige GRPO-control-
variant. Algoritmernes detaljer beskrives i Related Work
(\ref{sec:related}).

**3.5 Miljø- og implementations-pipeline.**
Projektets to vektor-baserede miljøer flades til 1-D float32-vektorer via
`DMCGymAdapter` i `scripts/project_envs.py` og afvikles gennem ObjectRL
\citep{baykal2025objectrl} ved hjælp af et projekt-egent bridge-modul
(`scripts/run_project_objectrl_baseline.py`), der monkey-patcher
`make_env`-symbolet i `objectrl.experiments.base_experiment` uden at
ændre kode under `external/objectrl/`. CarRacing returnerer
$(96, 96, 3)$ uint8-billeder \citep{towers2024gymnasium}, hvilket ikke er
foreneligt med ObjectRL's antagelse om 1-D `Box`-observationer. Projektet
implementerer derfor PPO, SAC og TD3 med en delt CNN-feature-extraktor
i `scripts/carracing_cnn.py`, drevet af
`scripts/run_carracing_cnn_baseline.py`. CarRacing-kørslerne blev udført
på Google Colab med CUDA og kopieret tilbage til det lokale repository,
sådan at alle baseline-resultater bruger den samme outputstruktur
(\ref{sec:implementation-paths}).

**3.6 Planlagt GRPO-control-udvidelse.**
Den endelige GRPO-control-variant er ikke implementeret på dette stadium.
Den ovenstående metodologi er bevidst valgt, så GRPO i den endelige fase
kan tilføjes som en fjerde algoritme under den samme MDP-formulering, det
samme evaluerings-metrik og den samme seed-struktur. De konkrete
algoritmiske og teoretiske detaljer udvikles i Theory-afsnittet
(\ref{sec:theory}).

---

## 9. English academic draft

> **Note.** Concise, report-facing. Citation placeholders use
> `\citep{...}`. Section references use `\ref{...}` placeholders that
> point to labels already declared in `report/sections/05_experiments.tex`.

**3 Methodology**

**3.1 Problem formulation as an MDP.**
Each of the three project tasks — `cartpole_swingup`, `acrobot_swingup`,
and `car_racing_continuous` — is modelled as an infinite-horizon
discounted Markov decision process (MDP)
$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0),
$$
where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space
(continuous, with environment-specific dimension), $P(s' \mid s, a)$ is
the transition kernel, $r(s, a)$ is a bounded reward function,
$\gamma \in [0, 1)$ is the discount factor used in the *training*
formulation, and $\rho_0$ is the initial-state distribution. The two
swing-up tasks expose 1-D vector observations, while
`car_racing_continuous` exposes image observations; the structural MDP
formulation is the same. A trajectory is denoted
$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ with
$s_0 \sim \rho_0$, $a_t \sim \pi(\cdot \mid s_t)$, and
$s_{t+1} \sim P(\cdot \mid s_t, a_t)$.

**3.2 Policies, returns, and evaluation metric.**
PPO and SAC represent the policy as a stochastic mapping
$\pi_\theta(a \mid s)$. TD3 uses a deterministic policy $\mu_\theta(s)$.
The discounted return used in the training formulation is
$$
G_t = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k},
$$
and the corresponding training objective is
$J(\theta) = \mathbb{E}^{\pi_\theta}[G_0]$. The evaluation metric is
deliberately distinct: the undiscounted episodic return
$$
J^{\mathrm{eval}}(\pi) = \mathbb{E}^{\pi}\!\left[\sum_{t=0}^{T-1} r_t\right],
$$
where $T$ is the (random) episode termination time. This undiscounted
episodic return is what is reported in the Experiments section (see
\ref{sec:evaluation-protocol}).

**3.3 Value functions and advantage.**
The state-value and action-value functions under a policy $\pi$ are
$$
V^\pi(s) = \mathbb{E}^\pi[\,G_t \mid s_t = s\,],
\qquad
Q^\pi(s, a) = \mathbb{E}^\pi[\,G_t \mid s_t = s,\, a_t = a\,],
$$
and the advantage function is
$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s).
$$
PPO and SAC use estimates of $A^\pi$ and $Q^\pi$ for policy updates; TD3
updates its policy through a learned action-value function. These
quantities are also where the eventual GRPO-control variant intervenes:
GRPO replaces the value-baseline component of the advantage with a
group-relative quantity computed from sampled trajectories.

**3.4 Baseline algorithms and their methodological role.**
PPO \citep{schulman2017proximal} serves as the on-policy stochastic
baseline, SAC \citep{haarnoja2018sacapps} as the off-policy
maximum-entropy baseline, and TD3 \citep{fujimoto2018td3} as the
off-policy deterministic baseline. Together they span the on-/off-policy
and stochastic/deterministic axes that any new policy-optimization
method should be measured against. In this midway report the algorithms
are *not* the methodological contribution; their role is to define the
comparison surface against which the final GRPO-control variant will be
evaluated. Algorithmic details are discussed in Related Work
(\ref{sec:related}).

**3.5 Environment and implementation pipeline.**
The two vector-control environments are flattened into 1-D float32
vectors by `DMCGymAdapter` in `scripts/project_envs.py` and run through
ObjectRL \citep{baykal2025objectrl} via a project-side bridge in
`scripts/run_project_objectrl_baseline.py` that monkey-patches the
`make_env` symbol already imported by
`objectrl.experiments.base_experiment`; no code under
`external/objectrl/` is modified. CarRacing returns $(96, 96, 3)$ uint8
image observations \citep{towers2024gymnasium}, which is incompatible
with ObjectRL's assumption of 1-D `Box` observations enforced by an
assertion in `objectrl.agents.base_agent`. The project therefore ships
PPO, SAC, and TD3 with a shared CNN feature extractor in
`scripts/carracing_cnn.py`, driven by
`scripts/run_carracing_cnn_baseline.py`. The CarRacing runs were
executed on Google Colab with CUDA and copied back into the local
repository so that all baseline outputs share the same result pipeline
(\ref{sec:implementation-paths}).

**3.6 Planned GRPO-control extension.**
The GRPO-control variant is not implemented at the midway stage. The
methodology above is fixed precisely so that, in the final stage, GRPO
can be added as a fourth algorithm under the same MDP formulation, the
same evaluation metric, and the same seed structure. The algorithmic
form and the convergence-oriented analysis of GRPO-control are deferred
to the Theory section (\ref{sec:theory}).

---

## 10. Suggested equations

The Methodology section should include a small, focused set of
equations. The list below lists each candidate, what it is for, where it
belongs, and a one-sentence reason. Put nothing in Methodology that
properly belongs in Theory.

| # | Purpose | LaTeX form | Plain-language explanation | Place |
|---|---|---|---|---|
| 1 | MDP definition | `\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0)` | Names the components used in every later equation. | Methodology (3.1) |
| 2 | Trajectory | `\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)` with $s_0 \sim \rho_0$, $a_t \sim \pi(\cdot \mid s_t)$, $s_{t+1} \sim P(\cdot \mid s_t, a_t)$ | Defines the random object that returns and value functions are taken with respect to. | Methodology (3.1) |
| 3 | Discounted return (training-side) | `G_t = \sum_{k=0}^{\infty} \gamma^k\, r_{t+k}` | The signal optimised by PPO/SAC/TD3 in their respective training formulations. | Methodology (3.2) |
| 4 | Episodic undiscounted evaluation return | `J^{\mathrm{eval}}(\pi) = \mathbb{E}^{\pi}\!\big[\sum_{t=0}^{T-1} r_t\big]` | The metric reported in figures and summary tables; deliberately distinct from the training objective. | Methodology (3.2) |
| 5 | State-value | `V^\pi(s) = \mathbb{E}^\pi[\, G_t \mid s_t = s\,]` | Standard value-function definition; needed for advantage. | Methodology (3.3) |
| 6 | Action-value | `Q^\pi(s,a) = \mathbb{E}^\pi[\, G_t \mid s_t = s,\, a_t = a\,]` | Used directly by SAC/TD3 critics. | Methodology (3.3) |
| 7 | Advantage | `A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)` | Connects PPO's policy update to the eventual GRPO group-relative replacement. | Methodology (3.3) |
| 8 | Cross-seed mean evaluation return (optional) | `\bar{J}_{\alpha,e,k} = \tfrac{1}{|\mathcal{Z}|}\sum_{z \in \mathcal{Z}} \hat{J}_{\alpha,e,k,z}` | Formalises the curve-aggregation done by `summarize_project_baselines.py`. | Methodology (3.5) — only if the Experiments section has nowhere else to anchor it; otherwise omit. |
| (X) | Bellman expectation equations | `V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s,a)[r(s,a) + \gamma V^\pi(s')]` | Useful for Theory but not required at midway in Methodology. | **Theory (Section 4), not here** |
| (X) | PPO clipped surrogate / SAC entropy / TD3 target smoothing | (algorithm-specific) | Belongs in Related Work or Theory, not in Methodology. | **Not here** |
| (X) | GRPO group-relative update | (project-specific, not yet implemented) | Methodology should not include the algorithm at the midway stage. | **Final report, in Theory** |

The first seven equations are the recommended core. Equation 8 is optional
and should be added only if the Experiments section needs the formal
anchor.

---

## 11. Suggested citation placeholders

Citations to insert in the Methodology section (keys already present in
`report/references.bib`):

| Where in Methodology | Cite | Existing key |
|---|---|---|
| First mention of PPO (Subsection 3.4) | PPO paper | `\citep{schulman2017proximal}` |
| Optional, when discussing advantage estimation | GAE paper | `\citep{schulman2015gae}` |
| First mention of SAC (Subsection 3.4) | SAC paper | `\citep{haarnoja2018sacapps}` |
| First mention of TD3 (Subsection 3.4) | TD3 paper | `\citep{fujimoto2018td3}` |
| First mention of ObjectRL (Subsection 3.5) | ObjectRL paper | `\citep{baykal2025objectrl}` |
| First mention of Gymnasium / CarRacing-v3 (Subsection 3.5) | Gymnasium paper | `\citep{towers2024gymnasium}` |
| Optional, when describing `cartpole_swingup` / `acrobot_swingup` as DM Control suite tasks | DM Control paper | `\citep{tunyasuvunakool2020dmcontrol}` |
| Forward-reference to GRPO in Subsection 3.6 (only if needed in addition to the Introduction) | GRPO / DeepSeekMath | `\citep{shao2024deepseekmath}` |

Notes:

- The Introduction already cites `shao2024deepseekmath`,
  `schulman2017proximal`, `haarnoja2018sacapps`, `fujimoto2018td3`,
  `baykal2025objectrl`, and `towers2024gymnasium`. Re-citing these on
  first use in Methodology is fine and helps a reader who skips around;
  do not cite them on every later mention.
- DDPG (`lillicrap2015continuous`), TRPO (`schulman2015trpo`), DPG
  (`silver2014deterministic`), and Kakade–Langford
  (`kakade2002approximately`) are in the bib but belong in Related Work
  or Theory.
- For an MDP/RL textbook reference, the bib does **not** currently
  contain a Sutton–Barto entry. Use only the local theory notebooks as
  the in-report justification for notation; if you want a textbook
  citation, add a new BibTeX entry first (decision left to the author).

---

## 12. Final author notes

**To decide manually**

- Whether to use uppercase ($S_t, A_t, R_{t+1}$) or lowercase ($s_t, a_t,
  r_t$) for the trajectory random variables. The local notation notebook
  uses the uppercase convention; the existing `05_experiments.tex`
  already uses the lowercase convention. Pick **one** and apply it to the
  whole report. The English/Danish drafts here use lowercase to stay
  consistent with the already-written Experiments section.
- Whether to keep Subsection 3.6 ("Planned GRPO-control extension") as a
  separate one-paragraph subsection, or fold it into a closing sentence
  of Subsection 3.4. The standalone subsection is more honest about
  midway scope; the folded version saves a header.
- Whether to include the optional Equation 8 (cross-seed mean
  aggregation) in Methodology, in Experiments, or omit it entirely.
- Whether the optional `tunyasuvunakool2020dmcontrol` citation is
  appropriate when describing the swing-up tasks. The drafts here do
  not include it; add it only if Methodology actually leans on the
  DM Control framing.
- The exact value of $\gamma$: defer to ObjectRL/algorithm defaults and
  point to the Experiments section instead of pinning a number in
  Methodology, unless the assignment requires it.

**To check against the assignment PDF (`DM887_Project.pdf`)**

- Whether the assignment requires the MDP to be presented in a specific
  tuple form (e.g., $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ vs.
  including $\rho_0$).
- Whether the assignment asks for a discrete-time vs. continuous-time
  formulation (the project tasks are discrete-time).
- Required environment names. Match the PDF's spelling
  (e.g., `CarRacing-v3`, `cartpole-swingup-v0`, `acrobot-swingup-v0` vs.
  the project short names used in the scripts).
- Required notation for evaluation episode return (the PDF should be the
  authority).
- Whether the PDF requires a separate "Algorithms" section rather than
  folding the algorithm naming into Methodology 3.4.

**To align later with Related Work, Experiments, and Theory**

- The algorithm short-names used here must match Related Work and the
  Experiments section.
- The MDP tuple in Methodology must match the tuple used in Theory; if
  Theory adopts the Sutton–Barto convention with $R_{t+1}$, change
  Methodology to match (or vice versa).
- The advantage definition $A^\pi = Q^\pi - V^\pi$ must match the form
  Theory uses for the GRPO group-relative discussion in the final
  report.
- The figure of evaluation aggregation is described informally here and
  formally in `summarize_project_baselines.py`. Make sure the
  Methodology equation (Equation 8 above) matches the script's
  computation if you include it.
- When the Theory section gets its first draft, sweep Methodology again
  and remove anything that has migrated upward.

**To postpone until the final report**

- The full GRPO-control algorithmic specification.
- Any convergence theorem or proof.
- Any algorithm-specific pseudocode.
- Any hyperparameter table beyond what is already pinned in the
  Experiments section.
- An ablation methodology (CNN trunk, batch size, etc.).

**Theory-notebook standardization items the author should resolve before
final LaTeX**

- Reward subscript: $R_{t+1}$ vs. $r_t$. The notation notebook
  (`theory/0_1_notation_in_rl.ipynb`) uses $R_{t+1}$; the existing
  Experiments section uses $r_t$ in $\sum_{t=0}^\infty \gamma^t
  r(s_t, a_t)$. Decide once.
- MDP tuple ordering. The notation notebook lists
  $(\mathcal{S}, \mathcal{A}, P, r, T, r_T)$ for finite-horizon and
  discusses the discounted variant separately. Pick one canonical
  tuple for the report and reuse it everywhere.
- Episodic-return notation. `theory/5_episodic_markov_decision_processes.ipynb`
  uses an SSP/episodic framing with hitting time $\tau$. The
  Methodology drafts here use $T$ for the episode termination index;
  rename to $\tau$ if you want exact notebook parity (but be aware
  that $\tau$ is also the trajectory symbol in the same drafts —
  prefer $T$ to avoid clashing).
- Optimal-policy notation. The notebook uses $V^*, Q^*, \pi^*$. The
  midway report does not need to use these; introduce them only when
  Theory does.

**Reminders**

- This file is **inspiration only**. Do not paste the Danish or English
  drafts into `report/sections/03_methodology.tex` verbatim; rewrite in
  your own voice.
- Re-check every concrete reference (file paths, symbol names) against
  the current state of `scripts/`, `theory/`, and the notebook before
  printing it.
- Keep the Methodology section honest about what is *not yet* in the
  project. The midway report wins by being precise about its
  foundation, not by overstating its results.
