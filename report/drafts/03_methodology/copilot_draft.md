# Methodology draft inspiration for the DM887 midway report

This document is inspiration material for `report/sections/03_methodology.tex`. It is not final report text and should be manually refined before being copied into the LaTeX report.

The main sources for this draft are:

- `report/sections/01_introduction.tex`
- `report/sections/05_experiments.tex`
- `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`
- `scripts/project_envs.py`
- `scripts/run_project_objectrl_baseline.py`
- `scripts/run_carracing_cnn_baseline.py`
- `scripts/carracing_cnn.py`
- `scripts/summarize_project_baselines.py`
- the local theory notebooks under `theory/`

## 1. Methodology section goal analysis

The Methodology section should connect the project framing in the Introduction to the concrete baseline results in the Experiments section. It should define the formal reinforcement-learning problem and document the experimental protocol in enough detail that the baseline matrix is understandable and reproducible.

For this midway report, the section should do four things:

1. **Fix notation.** Define the MDP, policies, trajectories, returns, value functions, action-value functions, and advantages in a notation consistent with the local theory notebooks and usable later for GRPO.
2. **Clarify the modelling view.** Explain that the three benchmark environments are treated as episodic control tasks, even though training algorithms may internally optimize discounted objectives.
3. **Document the baseline methodology.** Explain the role of PPO, SAC, and TD3 as baselines, the environment setup, seeds, training budgets, evaluation metric, and output files.
4. **Prepare the final GRPO extension.** State that the GRPO-control algorithm is planned for the final stage; the midway methodology establishes the notation and comparison surface, not the final method itself.

The section should avoid becoming a proof section. Bellman operators, contraction arguments, convergence assumptions, and detailed GRPO proofs belong in the Theory section or final report, not in the midway Methodology section.

## 2. Consistency check against existing Introduction and Experiments

### Connection to `report/sections/01_introduction.tex`

The Introduction already states that:

- the project studies GRPO for control;
- PPO, SAC, and TD3 are baselines rather than the contribution;
- the midway contribution is the experimental/methodological foundation;
- vector environments use ObjectRL;
- CarRacing uses project-side CNN implementations;
- no hyperparameter optimization was performed.

Methodology should **repeat briefly**:

- the project is at the midway stage;
- PPO/SAC/TD3 define the baseline comparison surface;
- the final GRPO-control variant is not yet implemented;
- the report needs a formal MDP setup before experiments can be interpreted.

Methodology should **not repeat in full**:

- the broad motivation for GRPO from language-model RL;
- the full contribution narrative from the Introduction;
- the final report structure.

Methodology must **clarify before Experiments**:

- what an MDP is in this report;
- what is meant by a policy, trajectory, return, value, Q-value, and advantage;
- why both stochastic and deterministic policies are relevant;
- what "undiscounted evaluation episode return" means;
- why the two implementation paths are methodologically necessary.

### Connection to `report/sections/05_experiments.tex`

The Experiments section already includes a substantial amount of protocol detail: experiment matrix, implementation paths, evaluation protocol, validation, results, stability, limitations, and transition to final GRPO.

Methodology should **repeat briefly**:

- algorithms: PPO, SAC, TD3;
- environments: `cartpole_swingup`, `acrobot_swingup`, `car_racing_continuous`;
- seeds: 0--4;
- vector environments use ObjectRL, CarRacing uses project-side CNN agents;
- evaluation metric is undiscounted evaluation episode return.

Methodology should **not repeat in full**:

- result validation numbers beyond the minimum needed for protocol context;
- final-return rankings;
- progress and stability tables;
- detailed result interpretation;
- figure discussion.

Methodology must **clarify before Experiments**:

- the distinction between training return/objectives and reported evaluation return;
- the common output schema that lets all runs be aggregated;
- that the baseline matrix is designed to be reused for final GRPO comparison;
- why differences in implementation path do not mean CarRacing was excluded.

## 3. Proposed subsection structure

### 3.1 Problem formulation as an MDP

- **Purpose:** Define the common formal model for all control tasks.
- **Key message:** Each environment is modelled as an MDP with states/observations, actions, stochastic transitions, rewards, discounting, and an initial-state distribution.
- **Connect to:** `theory/3_markov_decision_processes.ipynb`, `theory/4_discounted_decision_processes.ipynb`, `theory/5_episodic_markov_decision_processes.ipynb`, `plans/plan-midway-rapport-latex.md`.
- **Midway or final:** Belongs in the midway report; final report may add GRPO-specific assumptions.

### 3.2 Policies, trajectories, and returns

- **Purpose:** Define how agents interact with the MDP and how returns are measured.
- **Key message:** A policy induces a distribution over trajectories. Training algorithms may use discounted return, while reported evaluation uses undiscounted episodic return.
- **Connect to:** `theory/0_1_notation_in_rl.ipynb`, `theory/3_markov_decision_processes.ipynb`, `theory/4_discounted_decision_processes.ipynb`, `theory/5_episodic_markov_decision_processes.ipynb`.
- **Midway or final:** Belongs in the midway report.

### 3.3 Value functions and advantages

- **Purpose:** Define `V`, `Q`, and `A` because PPO and the later GRPO extension rely on advantage-like signals.
- **Key message:** `V^\pi` measures expected return from a state, `Q^\pi` from a state-action pair, and `A^\pi` the relative value of an action compared with the state baseline.
- **Connect to:** `theory/0_1_notation_in_rl.ipynb`, `theory/3_markov_decision_processes.ipynb`, `theory/4_discounted_decision_processes.ipynb`, PPO/GAE references.
- **Midway or final:** Belongs in the midway report; detailed GAE/GRPO estimators can wait for final method/theory unless already discussed in Related Work.

### 3.4 Baseline algorithms and comparison role

- **Purpose:** Explain why PPO, SAC, and TD3 appear in Methodology without turning this into Related Work.
- **Key message:** The baselines represent on-policy stochastic policy optimization, off-policy stochastic actor-critic learning, and deterministic actor-critic learning. They are used to define the comparison surface, not as new contributions.
- **Connect to:** Introduction; `report/references.bib`; `scripts/run_project_objectrl_baseline.py`; `scripts/run_carracing_cnn_baseline.py`.
- **Midway or final:** Belongs in the midway report. Detailed algorithm derivations should stay in Related Work/Theory.

### 3.5 Environment and observation setup

- **Purpose:** Explain the two implementation paths.
- **Key message:** Vector-control environments are adapted to Gymnasium-style 1-D observations and run through ObjectRL; CarRacing is an RGB image-control task and therefore uses project-side CNN agents.
- **Connect to:** `scripts/project_envs.py`, `scripts/register_project_envs.py`, `scripts/carracing_cnn.py`, final notebook Section 6.
- **Midway or final:** Belongs in the midway report because it is essential for reproducibility.

### 3.6 Training and evaluation protocol

- **Purpose:** Specify seeds, budgets, evaluation intervals, evaluation episodes, metric, and result format.
- **Key message:** The baseline protocol fixes a reproducible matrix that the final GRPO-control method can reuse.
- **Connect to:** final notebook Sections 3, 5, 7; `scripts/run_project_objectrl_baseline.py`; `scripts/run_carracing_cnn_baseline.py`; `scripts/summarize_project_baselines.py`; `results/processed/project_baselines/midway_vector_summary.csv`.
- **Midway or final:** Belongs in the midway report. Final report should update budgets and add GRPO.

### 3.7 Planned GRPO-control extension

- **Purpose:** Bridge Methodology to the final project without pretending the method is implemented.
- **Key message:** The later GRPO-control variant will be expressed in the same MDP notation and evaluated using the same baseline protocol.
- **Connect to:** Introduction, Theory section, assignment PDF, GRPO references.
- **Midway or final:** Include only a short paragraph now. Algorithm pseudocode and component motivation should wait for the final report.

## 4. Formal notation proposal

The local theory notebooks use closely related but not identical notation:

- `theory/3_markov_decision_processes.ipynb` defines finite-horizon MDPs as `(S, A, P, r, T, r_T)`.
- `theory/4_discounted_decision_processes.ipynb` defines discounted MDPs as `(S, A, P, r, gamma)`.
- `theory/5_episodic_markov_decision_processes.ipynb` defines episodic/SSP MDPs as `<S, S_G, A, p, r>`.
- `theory/2_markov_chains.ipynb` uses `p_0` for the initial distribution.
- `plans/plan-midway-rapport-latex.md` recommends a compact deep-RL tuple `(S, A, P, r, gamma, rho_0)`.

**Recommendation:** Use the compact report notation

```latex
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0)
```

and mention in prose that the benchmark environments are episodic and terminate through the environment API. This keeps the Methodology concise and standard for deep RL, while still aligning conceptually with the local theory notebooks. If you want to stay closer to the notebooks, replace `\rho_0` with `p_0`; do not use both in the final LaTeX.

### Proposed notation table

| Concept | Recommended report notation | Notes |
|---|---|---|
| Deterministic decision process | `s_{t+1} = f(s_t,a_t)` | Useful only as one-sentence background; no need for a subsection. |
| Markov chain | `P(s' \mid s)` | Useful only to explain that an MDP adds actions to Markovian dynamics. |
| MDP | `\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0)` | Main formal object for the report. |
| State / observation space | `\mathcal{S}` | Strictly, image/vector observations are observations; for this report they can be treated as states/observations to avoid POMDP complexity. |
| Action space | `\mathcal{A}` or `\mathcal{A}(s)` | Use `\mathcal{A}` unless state-dependent actions matter. |
| Transition kernel | `P(s' \mid s,a)` | Matches theory notebooks. |
| Reward function | `r(s,a)` or `r(s,a,s')` | Use `r(s,a)` in the main text; mention the environment returns realized rewards `r_t`. |
| Discount factor | `\gamma \in [0,1)` | Use for training/objective notation. |
| Initial-state distribution | `\rho_0` | Standard deep-RL notation; local notebooks use `p_0`. Pick one in final LaTeX. |
| Stochastic policy | `\pi_\theta(a \mid s)` | PPO and SAC use stochastic policies. |
| Deterministic policy | `\mu_\theta(s)` | Useful for TD3. |
| Trajectory | `\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots)` | Matches notation notebook's reward indexing. |
| Discounted return | `G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}` | Matches notation notebook. |
| Episodic evaluation return | `R_{\mathrm{eval}}(\tau)=\sum_{t=0}^{T_\tau-1} r_{t+1}` | Undiscounted return used in experiments. |
| State-value function | `V^\pi(s)=\mathbb{E}_\pi[G_t \mid S_t=s]` | Use discounted training return. |
| Action-value function | `Q^\pi(s,a)=\mathbb{E}_\pi[G_t \mid S_t=s,A_t=a]` | Needed for SAC/TD3 and advantage. |
| Advantage function | `A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)` | Needed for PPO and later GRPO framing. |
| Evaluation mean | `\bar{R}_{a,e}(n)=\frac{1}{|\mathcal{I}|}\sum_{i\in\mathcal{I}} R^{(i)}_{a,e}(n)` | Optional; use if describing mean curves formally. |
| Seed/run index | `i \in \{0,\ldots,4\}` | Useful for empirical mean notation. |

## 5. Safe claims

Safe claims for the Methodology section:

- Each benchmark task is treated as an MDP/control environment.
- The local theory notebooks motivate the progression from deterministic decision processes to Markov chains to MDPs.
- A deterministic decision process has transitions `s_{t+1}=f(s_t,a_t)`.
- A Markov chain has stochastic transitions depending only on the current state.
- An MDP combines actions with Markovian stochastic transitions.
- Discounted return is a standard training/objective quantity in RL.
- The reported experimental metric is undiscounted evaluation episode return.
- PPO and SAC use stochastic policy representations; TD3 uses a deterministic actor with exploration noise during training.
- PPO, SAC, and TD3 are baseline algorithms, not the proposed method.
- The midway implementation validates a PPO/SAC/TD3 baseline pipeline.
- Vector-control environments are `cartpole_swingup` and `acrobot_swingup`.
- The vector-control runner uses ObjectRL and a project-side bridge without modifying `external/objectrl`.
- `DMCGymAdapter` flattens DM Control observation dictionaries into `float32` vectors and exposes a Gymnasium-style API.
- `car_racing_continuous` is based on `CarRacing-v3` with `continuous=True`.
- CarRacing image observations are converted by `CarRacingCNNWrapper` from HWC `uint8` to CHW `float32` in `[0,1]` with shape `(3,96,96)`.
- CarRacing uses project-side PPO-CNN, SAC-CNN, and TD3-CNN training loops.
- Seeds are `0,1,2,3,4`.
- Vector-control midway runs use 20,000 steps, evaluation every 5,000 steps, and 3 evaluation episodes.
- CarRacing midway runs use 10,000 steps, evaluation every 1,000 steps, and 3 evaluation episodes.
- The result matrix is complete at the midway stage: 45 runs and 900 evaluation rows.
- No hyperparameter optimization was performed.
- The final GRPO-control method is planned for the next stage.

## 6. Claims to avoid

Avoid these claims:

- Do not claim the final GRPO-control algorithm is implemented.
- Do not include fake GRPO pseudocode.
- Do not claim that the methodology proves convergence.
- Do not claim that PPO/SAC/TD3 superiority is proven generally.
- Do not claim all algorithms converged.
- Do not claim state-of-the-art performance.
- Do not describe the final notebook as a dry run.
- Do not say CarRacing is deferred.
- Do not say PPO-CNN is missing.
- Do not imply ObjectRL directly supports the CarRacing image-observation setup used here.
- Do not imply vector and CarRacing results are directly comparable in wall-clock time.
- Do not introduce heavy Bellman contraction proofs in Methodology.
- Do not introduce undisclosed hyperparameter tuning.
- Do not present debug mode as the main methodology.
- Do not invent algorithm details beyond what scripts/notebook show.
- Do not overstate observations as true environment states if you want strict POMDP language; either say "states/observations" or define that observations are treated as states for this baseline report.

## 7. Implementation facts to extract

Concrete facts visible in the final notebook and scripts:

### Overall matrix

- Algorithms: `ppo`, `sac`, `td3`.
- Environments: `cartpole_swingup`, `acrobot_swingup`, `car_racing_continuous`.
- Seeds: `0,1,2,3,4`.
- Total runs: `3 x 3 x 5 = 45`.
- Total result rows: `900`.
- All rows are completed in the final notebook validation.

### Vector-control pipeline

- Defined in `scripts/run_project_objectrl_baseline.py`.
- Algorithms are `("ppo", "sac", "td3")`.
- Midway config: seeds `[0,1,2,3,4]`, `max_steps=20_000`, `eval_episodes=3`, `eval_frequency=5_000`, `warmup_steps=1_000`.
- PPO warmup is set to `0`; SAC/TD3 use the configured warmup.
- The project runner builds a matrix over `PROJECT_ENV_NAMES`, algorithms, and seeds.
- The runner patches ObjectRL's `base_experiment.make_env` symbol in process to use the project factory; it does not modify files in `external/objectrl`.
- ObjectRL config sets `training.parallelize_eval=False`.
- Evaluation environments use an effective seed offset of `+100`.
- ObjectRL outputs are parsed from `eval_results.npy` into CSV rows.
- ObjectRL's default MLP actor/critic path is blocked for CarRacing because it requires 1-D Box observations.

### Environment adapters

- Defined in `scripts/project_envs.py`.
- Project environment names:
  - `car_racing_continuous`
  - `cartpole_swingup`
  - `acrobot_swingup`
- `cartpole_swingup` maps to `dm_control.suite.load("cartpole", "swingup")`.
- `acrobot_swingup` maps to `dm_control.suite.load("acrobot", "swingup")`.
- `car_racing_continuous` maps to `gymnasium.make("CarRacing-v3", continuous=True)`.
- `DMCGymAdapter` flattens DM Control observation dictionaries into 1-D `float32` vectors.
- `DMCGymAdapter` exposes `reset(seed=...)` and `step(action) -> (obs, reward, terminated, truncated, info)`.
- `DMCGymAdapter` uses Gymnasium `Box` spaces for action and observation spaces.
- `register_project_envs.py` registers:
  - `DM887/CarRacingContinuous-v0`
  - `DM887/CartpoleSwingup-v0`
  - `DM887/AcrobotSwingup-v0`
- `register_project_envs.py` notes that ObjectRL's CLI rejects custom Gymnasium IDs unless `external/objectrl` is modified; the project-side runner avoids this.

### CarRacing CNN pipeline

- Defined in `scripts/carracing_cnn.py` and driven by `scripts/run_carracing_cnn_baseline.py`.
- Project environment is `car_racing_continuous`.
- Supported algorithms are `("sac", "td3", "ppo")`.
- Model types are `cnn_sac`, `cnn_td3`, `cnn_ppo`.
- Midway config: seeds `[0,1,2,3,4]`, `max_steps=10_000`, `eval_episodes=3`, `eval_frequency=1_000`, `warmup_steps=500`.
- `CarRacingCNNWrapper` converts `(96,96,3)` HWC `uint8` observations to `(3,96,96)` CHW `float32` in `[0,1]`.
- `CNNFeatureExtractor` uses an Atari-style 3-convolution stack and a linear projection to a 256-dimensional feature representation.
- SAC-CNN uses a stochastic actor and twin critics.
- TD3-CNN uses a deterministic actor, twin critics, target policy smoothing, and delayed policy updates.
- PPO-CNN uses a shared CNN actor-critic, rollout buffer, GAE-lambda advantages, clipped surrogate objective, value loss, entropy bonus, and gradient clipping.
- CarRacing CNN evaluation records rows with columns including `algorithm`, `project_env`, `seed`, `train_step`, `eval_episode`, `eval_return`, `wall_time_seconds`, `status`, `observation_mode`, and `model_type`.
- The final notebook states the full CarRacing midway matrix was executed on Google Colab with CUDA and copied back into `results/processed/project_baselines/`.

### Aggregation and figures

- `scripts/summarize_project_baselines.py` reads `results/processed/project_baselines/<prefix>_*_eval.csv`.
- Expected algorithms: `ppo`, `sac`, `td3`.
- Expected environments: `cartpole_swingup`, `acrobot_swingup`, `car_racing_continuous`.
- Expected seeds: `0,1,2,3,4`.
- It aggregates by `algorithm`, `project_env`, `seed`, `train_step`, and `status`.
- It computes mean `eval_return` and `n_repeats`.
- It plots mean return across seeds with a standard-deviation band where at least two seeds are present.
- The y-axis label is "undiscounted eval episode return".

### Facts to mark as CHECK if needed

- Exact ObjectRL internal hyperparameters for PPO/SAC/TD3: **CHECK ObjectRL configs** if you want to report them.
- Exact discount factors used by ObjectRL baselines: **CHECK ObjectRL configs** before stating values.
- Exact final GRPO algorithm components: **not implemented / final-stage work**.
- Whether to call observations "states" or "observations" in final LaTeX: **author decision**.

## 8. Danish draft

Dette afsnit beskriver den metodiske ramme for midway-rapporten. Formålet er at fastlægge den notation og evalueringsprotokol, der bruges til baseline-eksperimenterne, og som senere kan genbruges til den endelige GRPO-relaterede kontrolmetode. Ved midway-stadiet præsenteres der ikke en færdig GRPO-kontrolalgoritme. Metodologien etablerer i stedet det formelle og praktiske grundlag for sammenligningen med PPO, SAC og TD3.

Vi modellerer hver opgave som en Markov decision process
\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0).
\]
Her er \(\mathcal{S}\) tilstands- eller observationsrummet, \(\mathcal{A}\) handlingsrummet, \(P(s'\mid s,a)\) overgangskernen, \(r(s,a)\) belønningsfunktionen, \(\gamma\in[0,1)\) diskonteringsfaktoren, og \(\rho_0\) startfordelingen. Denne notation følger den samme ide som i de lokale teori-notebooks: en MDP kombinerer handlinger fra deterministiske beslutningsprocesser med Markovske, stokastiske overgange.

En stokastisk politik skrives \(\pi_\theta(a\mid s)\), mens en deterministisk politik skrives \(\mu_\theta(s)\). Denne skelnen er relevant, fordi PPO og SAC bruger stokastiske politikker, mens TD3 bygger på en deterministisk actor. En trajektorie kan skrives som
\[
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots),
\]
hvor starttilstanden trækkes fra \(\rho_0\), handlinger vælges af politikken, og næste tilstand trækkes fra overgangskernen. Det diskonterede afkast fra tid \(t\) defineres som
\[
G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
\]
I selve evalueringen rapporteres dog undiscounted episodic return, dvs. summen af observerede belønninger i en evaluerings-episode.

Værdifunktionerne bruges til at beskrive, hvad en politik forventes at opnå. Tilstandsværdien er
\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
\]
og handlingsværdien er
\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
\]
Fordelsfunktionen defineres som
\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]
Denne notation er vigtig, fordi advantage-lignende signaler er centrale for PPO og senere for den planlagte GRPO-variant.

Baseline-metoden består af PPO, SAC og TD3. De bruges ikke som nye metodiske bidrag, men som referencepunkter for den endelige GRPO-sammenligning. PPO repræsenterer en on-policy policy-gradient tilgang, SAC en off-policy maksimum-entropi actor-critic tilgang, og TD3 en deterministisk actor-critic tilgang for kontinuerte handlingsrum.

Miljøerne opdeles i to implementeringsspor. `cartpole_swingup` og `acrobot_swingup` behandles som vektorbaserede kontrolmiljøer og køres gennem ObjectRL via et projekt-side miljøbridge. DM Control observationer flades ud til én `float32`-vektor og eksponeres med en Gymnasium-lignende `reset`/`step` API. Dette gør det muligt at bruge ObjectRL uden at ændre den eksterne ObjectRL-kode.

`car_racing_continuous` kræver et andet spor, fordi CarRacing-v3 returnerer RGB-billedobservationer med form `(96,96,3)`. Den anvendte ObjectRL-baseline forventer derimod 1-dimensionelle vektorobservationer. Derfor anvender projektet egne CNN-baserede implementeringer af PPO, SAC og TD3 til CarRacing. Wrapperen konverterer observationer fra HWC `uint8` til CHW `float32` i intervallet `[0,1]`, og de CNN-baserede agenter trænes som projekt-side baselines.

Den reproducerbare midway-protokol bruger algoritmerne PPO, SAC og TD3 på miljøerne `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous` med seeds \(0,1,2,3,4\). Vektorbaserede kørsler bruger 20.000 miljøskridt, evaluering hver 5.000 skridt og 3 evaluerings-episoder. CarRacing-kørsler bruger 10.000 miljøskridt, evaluering hver 1.000 skridt og 3 evaluerings-episoder. Der blev ikke udført hyperparameteroptimering.

Denne metodologi etablerer dermed sammenligningsfladen for den afsluttende GRPO-fase. Den endelige rapport skal udvide samme notation og samme evalueringsprotokol med en konkret GRPO-kontrolvariant, pseudokode, komponentmotivation og teoretisk analyse.

## 9. English academic draft

This section defines the methodology used in the midway report. Its purpose is to fix the formal notation and experimental protocol used for the baseline experiments, and to prepare the later GRPO-control extension. At the midway stage, the report does not present a completed GRPO-control algorithm. Instead, the methodology establishes a reproducible comparison surface for PPO, SAC, and TD3.

Each task is modelled as a Markov Decision Process
\[
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0).
\]
Here, \(\mathcal{S}\) denotes the state or observation space, \(\mathcal{A}\) the action space, \(P(s'\mid s,a)\) the transition kernel, \(r(s,a)\) the reward function, \(\gamma\in[0,1)\) the discount factor, and \(\rho_0\) the initial-state distribution. This follows the conceptual progression in the local theory notebooks: deterministic decision processes specify action-dependent transitions, Markov chains introduce stochastic Markovian dynamics, and MDPs combine actions with stochastic transitions.

A stochastic policy is written as \(\pi_\theta(a\mid s)\), while a deterministic policy is written as \(\mu_\theta(s)\). This distinction is useful because PPO and SAC use stochastic policies, whereas TD3 uses a deterministic actor. A trajectory is denoted
\[
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots),
\]
where the initial state is drawn from \(\rho_0\), actions are selected by the policy, and subsequent states are sampled from the transition kernel. The discounted return from time \(t\) is
\[
G_t=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
\]
The reported experimental metric is different: evaluation curves use undiscounted episodic evaluation return, i.e. the sum of rewards observed during an evaluation episode.

The value function and action-value function under a policy \(\pi\) are defined as
\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
\qquad
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
\]
The advantage function is
\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
\]
This notation is important because advantage-like quantities are central to PPO and will also be needed to describe the planned GRPO-control variant.

The baseline methodology uses PPO, SAC, and TD3 as reference algorithms rather than as novel contributions. PPO represents on-policy stochastic policy optimization, SAC represents off-policy maximum-entropy actor-critic learning, and TD3 represents deterministic actor-critic learning for continuous action spaces \citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}. These methods define the comparison surface for the final GRPO-control evaluation.

The implementation uses two environment paths. The vector-control environments `cartpole_swingup` and `acrobot_swingup` are run through ObjectRL using a project-side environment bridge \citep{baykal2025objectrl}. The bridge constructs the project environments without modifying `external/objectrl`, flattens DeepMind Control observation dictionaries into `float32` vectors, and exposes a Gymnasium-style API. The CarRacing environment follows a separate project-side CNN path because `CarRacing-v3` returns image observations of shape `(96,96,3)`, while the ObjectRL baseline path used here expects one-dimensional vector observations. The CarRacing wrapper converts RGB `uint8` observations to channel-first `float32` tensors in `[0,1]`, and PPO-CNN, SAC-CNN, and TD3-CNN are trained through project-side PyTorch implementations.

The midway protocol evaluates PPO, SAC, and TD3 on `cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous` using seeds \(0,1,2,3,4\). Vector-control runs use 20,000 environment steps, evaluation every 5,000 steps, and 3 evaluation episodes. CarRacing runs use 10,000 environment steps, evaluation every 1,000 steps, and 3 evaluation episodes. No hyperparameter optimization was performed. The resulting CSV files store algorithm, environment, seed, training step, evaluation episode, evaluation return, status, and implementation metadata, which allows the same aggregation pipeline to produce validated summary tables and learning curves.

This methodology establishes the baseline comparison surface for the final stage. The final report should extend the same notation and protocol with the concrete GRPO-control algorithm, pseudocode, component motivation, and theory-backed comparison against PPO, SAC, and TD3.

## 10. Suggested equations

### 1. MDP definition

- **Purpose:** Establish the formal model.
- **Plain-language explanation:** Each task is represented as states/observations, actions, stochastic transitions, rewards, discounting, and an initial-state distribution.
- **LaTeX form:**
  ```latex
  \mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0).
  ```
- **One-sentence explanation:** This tuple defines the common RL/control setting used for all baseline environments.
- **Where it belongs:** Methodology.

### 2. Markov property

- **Purpose:** Connect to the theory notebooks and justify the MDP model.
- **Plain-language explanation:** The next state depends only on the current state and action, not the full history.
- **LaTeX form:**
  ```latex
  \Pr(S_{t+1}=s' \mid S_t=s, A_t=a, H_t)
  =
  P(s'\mid s,a).
  ```
- **One-sentence explanation:** This is the Markov assumption underlying the MDP formulation.
- **Where it belongs:** Methodology, but optional if space is tight.

### 3. Trajectory

- **Purpose:** Define the sampled interaction sequence.
- **Plain-language explanation:** A trajectory is the sequence of states/observations, actions, and rewards generated by a policy in an environment.
- **LaTeX form:**
  ```latex
  \tau = (s_0,a_0,r_1,s_1,a_1,r_2,\ldots).
  ```
- **One-sentence explanation:** Trajectories are the data from which training and evaluation returns are obtained.
- **Where it belongs:** Methodology.

### 4. Discounted return

- **Purpose:** Define the standard RL return used in value-function notation.
- **Plain-language explanation:** Future rewards are geometrically weighted by a discount factor.
- **LaTeX form:**
  ```latex
  G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}.
  ```
- **One-sentence explanation:** This is the return underlying the value and action-value functions.
- **Where it belongs:** Methodology.

### 5. Undiscounted episodic evaluation return

- **Purpose:** Define the reported experiment metric.
- **Plain-language explanation:** During evaluation, the report sums observed episode rewards without discounting.
- **LaTeX form:**
  ```latex
  R_{\mathrm{eval}}(\tau)
  =
  \sum_{t=0}^{T_\tau-1} R_{t+1}.
  ```
- **One-sentence explanation:** This is the y-axis quantity in the baseline learning curves.
- **Where it belongs:** Methodology and Experiments.

### 6. Value and action-value functions

- **Purpose:** Define expected returns under a policy.
- **Plain-language explanation:** `V` evaluates a state; `Q` evaluates taking a specific action first.
- **LaTeX form:**
  ```latex
  V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],
  \qquad
  Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
  ```
- **One-sentence explanation:** These functions describe the expected quality of states and state-action pairs under a policy.
- **Where it belongs:** Methodology.

### 7. Advantage function

- **Purpose:** Prepare PPO and GRPO notation.
- **Plain-language explanation:** The advantage measures how much better an action is than the policy's average action value in that state.
- **LaTeX form:**
  ```latex
  A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
  ```
- **One-sentence explanation:** Advantage-like signals motivate PPO updates and the later GRPO-control extension.
- **Where it belongs:** Methodology.

### 8. Empirical mean evaluation return across seeds

- **Purpose:** Formalize the plotted learning curves.
- **Plain-language explanation:** At each evaluation step, returns are averaged across the five seeds for a given algorithm and environment.
- **LaTeX form:**
  ```latex
  \bar{R}_{a,e}(n)
  =
  \frac{1}{|\mathcal{I}|}
  \sum_{i\in\mathcal{I}} R^{(i)}_{a,e}(n),
  \qquad
  \mathcal{I}=\{0,1,2,3,4\}.
  ```
- **One-sentence explanation:** This defines the mean curve shown for algorithm `a` on environment `e` at evaluation step `n`.
- **Where it belongs:** Methodology if space permits; otherwise Experiments.

### Equations to postpone to Theory

- Bellman optimality equations for the final GRPO analysis.
- Contraction inequalities for discounted MDPs.
- Any theorem about GRPO policy improvement or convergence.
- Full policy-gradient or PPO clipped surrogate derivations, unless required for Related Work.

## 11. Suggested citation placeholders

Use only existing BibTeX keys from `report/references.bib` unless you add new entries manually later.

- **PPO baseline / stochastic policy optimization:** `\citep{schulman2017proximal}`
- **GAE / advantage estimation:** `\citep{schulman2015gae}`
- **SAC baseline:** `\citep{haarnoja2018sacapps}`
- **TD3 baseline:** `\citep{fujimoto2018td3}`
- **Deterministic policy-gradient background:** `\citep{silver2014deterministic}` or `\citep{lillicrap2015continuous}` if useful.
- **ObjectRL implementation framework:** `\citep{baykal2025objectrl}`
- **Gymnasium / CarRacing interface:** `\citep{towers2024gymnasium}`
- **DeepMind Control environments:** `\citep{tunyasuvunakool2020dmcontrol}`
- **GRPO source:** `\citep{shao2024deepseekmath}`
- **Course/local theory notebooks:** No BibTeX key exists in `report/references.bib`; use a descriptive placeholder such as `[course material / local theory notebooks]` only in notes, or cite no external key unless the assignment requires it.

Possible compact citation sentence:

> PPO, SAC, and TD3 are used as complementary baseline families for stochastic on-policy optimization, maximum-entropy actor-critic learning, and deterministic actor-critic learning, respectively \citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}. The vector-control baseline path uses ObjectRL \citep{baykal2025objectrl}, while CarRacing-v3 follows the Gymnasium/Farama interface \citep{towers2024gymnasium}.

## 12. Final author notes

- Decide whether the final LaTeX should use `\rho_0` or `p_0` for the initial-state distribution. The theory notebooks use `p_0`; the report plan uses `\rho_0`. I recommend `\rho_0` for the report because it is common in deep-RL papers and avoids conflict with transition probabilities, but use one consistently.
- Decide whether to call `\mathcal{S}` the state space or "state/observation space". Strictly, image observations are observations, but introducing POMDP notation would be unnecessary for the midway report.
- Check the assignment PDF requirement for Methodology: it specifically asks for the complete basic notation required to describe the MDP. Make sure the final section includes the MDP tuple, policy, trajectory, return, value, Q-value, and advantage definitions.
- Align with Related Work by not re-explaining PPO/SAC/TD3 mechanisms in detail; cite them and describe their role.
- Align with Experiments by not repeating all validation and ranking numbers. Methodology should define the protocol; Experiments should report the results.
- Align with Theory by avoiding proof-heavy Bellman/convergence material in Methodology. The Theory section can later use the same notation for assumptions, lemmas, theorems, and proofs.
- Postpone final GRPO pseudocode, GRPO loss, group construction, value-function augmentation, and convergence claims until the final report.
- Before writing final LaTeX, standardize notation across sections: use either `R_t`/`r_t` consistently for realized rewards, decide whether trajectories include `r_t` or `r_{t+1}`, and keep the evaluation return notation distinct from the discounted training return.
