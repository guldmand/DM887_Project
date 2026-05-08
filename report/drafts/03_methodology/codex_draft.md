# Methodology Draft Inspiration for the DM887 GRPO Midway Report

This document is inspiration material for `report/sections/03_methodology.tex`. It is not final LaTeX text. It is written to align the Methodology section with the current midway status: a completed PPO/SAC/TD3 baseline matrix, a reproducible experimental pipeline, and a planned GRPO-control extension for the final stage.

Primary local sources inspected:

- `DM887_Project.pdf`
- `README.md`
- `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`
- `report/DM887_Report.tex`
- `report/sections/01_introduction.tex`
- `report/sections/03_methodology.tex`
- `report/sections/05_experiments.tex`
- `report/references.bib`
- `plans/plan-midway-rapport-latex.md`
- `docs/scientific-writing/plan-scientific-writing-methodology.md`
- `docs/scientific-writing/plan-scientific-writing.md`
- `docs/references/reading-list.md`
- `docs/references/plan-references.md`
- `scripts/run_project_objectrl_baseline.py`
- `scripts/run_carracing_cnn_baseline.py`
- `scripts/carracing_cnn.py`
- `scripts/project_envs.py`
- `scripts/register_project_envs.py`
- `scripts/summarize_project_baselines.py`
- `results/processed/project_baselines/midway_vector_summary.csv`
- `theory/0_1_notation_in_rl.ipynb`
- `theory/1_deterministic_decision_processes.ipynb`
- `theory/2_markov_chains.ipynb`
- `theory/3_markov_decision_processes.ipynb`
- `theory/4_discounted_decision_processes.ipynb`
- `theory/5_episodic_markov_decision_processes.ipynb`

## 1. Methodology Section Goal Analysis

The Methodology section should do four things in this specific midway report.

First, it should fix the formal reinforcement-learning notation used by the rest of the report. The Introduction already frames the project as GRPO for continuous-control style benchmark environments, while the Experiments section reports the completed baseline matrix. Methodology should therefore define the MDP, policy, trajectory, return, value, action-value, and advantage notation needed to make those later descriptions precise.

Second, it should make the experimental protocol reproducible without turning into an implementation manual. The section should explain which environments, algorithms, seed set, training budgets, evaluation cadence, evaluation metric, and result artifacts define the baseline matrix. It should be clear that the comparison surface is built from `algorithm, project_env, seed, train_step, eval_episode, eval_return`.

Third, it should justify the implementation split. The vector-control environments are run through ObjectRL using a project-side environment bridge, while CarRacing uses project-side CNN implementations because the image observation setting does not fit ObjectRL's 1-D vector-observation assumption. This is a methodology point, not just a code detail, because it affects reproducibility and comparability.

Fourth, it should prepare the final GRPO extension without presenting it as completed. At the midway stage, GRPO-control is a planned next step. Methodology can say that the baseline protocol establishes the common notation and comparison surface for the final GRPO comparison, but the GRPO-control algorithmic details, pseudocode, and final experiments should wait for the final report.

The section should not overclaim. It should use cautious wording such as "At the midway stage", "The methodology fixes the notation and evaluation protocol", "The baseline protocol establishes the comparison surface", "No hyperparameter optimization was performed", and "The GRPO-control variant is planned for the final stage".

## 2. Consistency Check Against Existing Introduction and Experiments

### Connection to `report/sections/01_introduction.tex`

The Introduction already establishes the red thread:

- the project investigates reinforcement learning for control;
- PPO, SAC, and TD3 are used as baselines;
- the midway contribution is a reproducible baseline matrix and validated pipeline;
- the final GRPO-control extension is not yet complete.

Methodology should repeat this only briefly. It should not re-explain the full motivation for GRPO or repeat the high-level project story. Instead, it should turn the Introduction's framing into formal and procedural terms: MDP formulation, policies, returns, baseline protocol, and implementation paths.

Useful brief repetition:

- This is a midway-stage methodology.
- The current experiments cover PPO, SAC, and TD3.
- The final GRPO-control method will be added later.
- The baseline protocol is used to create a comparison surface for that final stage.

Avoid repeating:

- broad motivational paragraphs about RL and control;
- detailed related-work narrative;
- detailed baseline result rankings;
- final-stage GRPO claims.

### Connection to `report/sections/05_experiments.tex`

The Experiments section already documents the matrix and interprets learning curves. Methodology should provide the definitions needed before Experiments:

- what an environment is in formal MDP terms;
- what a policy is, including stochastic and deterministic policy notation;
- what discounted training returns and undiscounted evaluation returns mean;
- how a run is indexed by algorithm, environment, seed, and evaluation step;
- why ObjectRL and project-side CNN pipelines coexist.

Useful brief repetition:

- 3 algorithms x 3 environments x 5 seeds = 45 runs;
- 900 evaluation rows validated by the final notebook;
- vector-control environments use ObjectRL;
- CarRacing uses project-side CNN implementations;
- no hyperparameter optimization was performed.

Avoid repeating:

- detailed CartPole, Acrobot, and CarRacing result rankings;
- progress tables, stability tables, and figure-level discussion;
- final numerical interpretation of learning curves;
- extended limitation discussion that belongs in Experiments or Conclusion.

What Methodology must clarify before Experiments:

- The report distinguishes the discounted RL objective from the logged undiscounted evaluation return.
- PPO and SAC are represented as stochastic-policy baselines, while TD3 is naturally represented as a deterministic-policy baseline with exploration noise during training.
- The CarRacing image pipeline is not an unfinished or deferred environment. It is implemented through project-side CNN code because the ObjectRL baseline interface is designed for 1-D vector observations.
- The current protocol validates the pipeline and baseline comparison surface; it does not establish final algorithm superiority.

## 3. Proposed Subsection Structure

### 3.1 Problem Formulation as an MDP

Purpose: Define the formal problem class used by the report.

Key message: Each benchmark task is treated as an episodic interaction with an MDP-like environment, with states or observations, continuous actions, transition dynamics, rewards, a discount factor for learning, and an initial-state distribution.

Connect to: theory notebooks `0_1`, `3`, `4`, `5`; `docs/scientific-writing/plan-scientific-writing-methodology.md`; assignment PDF requirement for interim MDP notation.

Midway or final: Belongs in the midway report.

### 3.2 Policies, Trajectories, and Returns

Purpose: Define the objects that training and evaluation optimize or report.

Key message: A trajectory records state/observation, action, reward, and next-state information; training algorithms use discounted-return objectives, while the report logs undiscounted evaluation episode return.

Connect to: theory notebooks `0_1`, `3`, `4`, `5`; notebook report-ready evaluation protocol.

Midway or final: Belongs in the midway report.

### 3.3 Value Functions and Advantages

Purpose: Give enough notation to understand PPO, SAC, TD3, and later GRPO-control comparisons.

Key message: The value function, action-value function, and advantage function provide the shared language for policy-gradient, actor-critic, and advantage-based methods. Heavy Bellman proofs should be kept in Theory.

Connect to: theory notebooks `0_1`, `3`, `4`; Related Work and Theory sections later.

Midway or final: Belongs in the midway report, but keep concise.

### 3.4 Baseline Algorithms and Their Comparison Role

Purpose: Explain why PPO, SAC, and TD3 are included.

Key message: PPO, SAC, and TD3 are not presented as optimized or definitive winners. They are standard baseline families that provide the comparison surface for the final GRPO-control stage.

Connect to: `report/sections/01_introduction.tex`, `report/sections/05_experiments.tex`, `references.bib`, reading list.

Midway or final: Belongs in the midway report.

### 3.5 Environment and Observation Setup

Purpose: Explain the three environments and the two implementation paths.

Key message: CartPole Swingup and Acrobot Swingup are vector-control tasks adapted through a project-side DM Control to Gymnasium bridge and trained with ObjectRL. CarRacing is an image-observation task using project-side CNN implementations for PPO, SAC, and TD3.

Connect to: `scripts/project_envs.py`, `scripts/register_project_envs.py`, `scripts/run_project_objectrl_baseline.py`, `scripts/run_carracing_cnn_baseline.py`, `scripts/carracing_cnn.py`, notebook setup notes.

Midway or final: Belongs in the midway report.

### 3.6 Training and Evaluation Protocol

Purpose: Define how runs are generated and compared.

Key message: The midway protocol covers 45 runs with five seeds per algorithm-environment pair. It logs deterministic evaluation episodes at fixed intervals and stores the results in reproducible CSV artifacts. No hyperparameter optimization was performed.

Connect to: final notebook, `scripts/run_project_objectrl_baseline.py`, `scripts/run_carracing_cnn_baseline.py`, `scripts/summarize_project_baselines.py`, `results/processed/project_baselines/midway_vector_summary.csv`.

Midway or final: Belongs in the midway report.

### 3.7 Planned GRPO-Control Extension

Purpose: Connect the current protocol to the final project stage.

Key message: The GRPO-control variant is planned for the final stage and should be evaluated against the same baseline comparison surface. The final report should add the actual GRPO-control objective, pseudocode, and implementation details.

Connect to: assignment PDF, Introduction, Related Work, future final Experiments.

Midway or final: Include only a short midway paragraph now. Full algorithm details should wait for the final report.

## 4. Formal Notation Proposal

The local theory notebooks use mostly standard notation but mix finite-horizon, discounted, and episodic formulations. The report should choose one compact notation and then use it consistently.

Recommended report convention:

- Use uppercase random variables for stochastic process notation when helpful: `S_t`, `A_t`, `R_{t+1}`.
- Use lowercase realizations in trajectories: `s_t`, `a_t`, `r_{t+1}`.
- Use `\tau` for a trajectory.
- Avoid using `\tau` for an episode termination time, because `theory/5_episodic_markov_decision_processes.ipynb` uses `\tau` as a hitting time while `theory/0_1_notation_in_rl.ipynb` uses `\tau` for a trajectory. Use `H`, `T_{\mathrm{ep}}`, or `T_i` for episode length instead.
- Use `\rho_0` for the initial-state distribution if included in the MDP tuple. Some theory notebooks use `p_0`; choose one notation for the report and define it once.

### Deterministic Decision Process, if useful

Use only as a short bridge, not as a main methodology object:

`s_{t+1} = f_t(s_t,a_t)` or, in the time-homogeneous case, `s_{t+1}=f(s_t,a_t)`.

Purpose: This can motivate the control setting, but it is not necessary to build the midway methodology around deterministic decision processes.

### Markov Chain, if useful

Use only as background:

`\Pr(X_{t+1}=j \mid X_t=i, X_{t-1},\ldots,X_0)=\Pr(X_{t+1}=j \mid X_t=i)=p_{ij}`.

Purpose: This can motivate the Markov property, but most of the report should move directly to MDPs.

### MDP

Recommended compact tuple for the report:

`\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0)`.

Definitions:

- `\mathcal{S}`: state or observation space. In the experiments this may be a flattened vector observation for DM Control tasks or an image tensor for CarRacing.
- `\mathcal{A}`: action space. The project tasks use continuous-action control settings.
- `P(s' \mid s,a)`: transition kernel.
- `r(s,a)=\mathbb{E}[R_{t+1}\mid S_t=s,A_t=a]`: expected one-step reward.
- `\gamma \in [0,1)`: discount factor used in the training objective.
- `\rho_0`: initial-state distribution.

Alternative local-theory tuples:

- finite horizon: `(\mathcal{S},\mathcal{A},P,r,T,r_T)`;
- discounted infinite horizon: `(\mathcal{S},\mathcal{A},P,r,\gamma)`.

Recommendation: Use the compact tuple with `\rho_0` in Methodology, and mention finite episodes separately through `H` or `T_{\mathrm{ep}}`. This keeps the notation compatible with discounted training and episodic evaluation.

### Policy

Generic policy:

`\pi(a\mid s)` or parameterized `\pi_\theta(a\mid s)`.

Stochastic policy:

`\pi_\theta(a\mid s)` gives a distribution over actions. This is natural for PPO and SAC.

Deterministic policy:

`\mu_\theta(s)` gives an action directly. This is natural for TD3, with exploration noise used during training.

### Trajectory

Use:

`\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots,s_H)`.

This follows the local `R_{t+1}` reward convention and avoids using the same symbol for episode termination time.

### Discounted Return

For learning-objective notation:

`G_t^\gamma=\sum_{k=0}^{H-t-1}\gamma^k R_{t+k+1}` for finite episodes, or `G_t^\gamma=\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}` in the discounted infinite-horizon notation.

### Episodic Evaluation Return

For logged evaluation:

`G^{\mathrm{eval}}(\tau)=\sum_{t=0}^{H-1}R_{t+1}`.

This is the undiscounted episode return used in the notebook and figure summaries.

### Value, Action-Value, and Advantage Functions

Value:

`V^\pi(s)=\mathbb{E}_\pi[G_t^\gamma\mid S_t=s]`.

Action-value:

`Q^\pi(s,a)=\mathbb{E}_\pi[G_t^\gamma\mid S_t=s,A_t=a]`.

Advantage:

`A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)`.

These match the local notation guide and are sufficient for PPO/GAE, actor-critic methods, and the planned GRPO-control discussion.

### Evaluation Mean Across Seeds

If the report needs an equation for the plotted mean curve, use:

`\bar{J}_{\alpha,e}(n)=\frac{1}{|\mathcal{Z}|M}\sum_{z\in\mathcal{Z}}\sum_{m=1}^{M}G^{\mathrm{eval}}_{\alpha,e,z,n,m}`.

Definitions:

- `\alpha`: algorithm index, e.g. PPO, SAC, or TD3;
- `e`: environment;
- `z \in \mathcal{Z}`: seed, with `\mathcal{Z}=\{0,1,2,3,4\}` at midway;
- `n`: training step at evaluation;
- `M`: number of evaluation episodes per evaluation block;
- `G^{\mathrm{eval}}_{\alpha,e,z,n,m}`: undiscounted return from evaluation episode `m`.

Use only if needed. The Experiments section may be the better place for this equation if Methodology becomes too dense.

## 5. Safe Claims

Safe claims for the Methodology section:

- At the midway stage, the implemented methodology establishes notation and a reproducible baseline protocol rather than a completed GRPO-control algorithm.
- The project treats the benchmark tasks as episodic reinforcement-learning/control problems.
- The report can formulate the tasks using MDP notation with state or observation space, action space, transition kernel, reward function, discount factor, policy, trajectory, and return.
- PPO, SAC, and TD3 are used as baseline algorithms for the current midway matrix.
- PPO and SAC can be described with stochastic-policy notation; TD3 can be described with deterministic-policy notation.
- The baseline matrix consists of 3 algorithms x 3 environments x 5 seeds = 45 runs.
- The environments are `cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous`.
- The seed list is `0,1,2,3,4`.
- The final notebook validates 45 expected CSV files, 45 present combinations, and 900 evaluation rows.
- The logged comparison metric is undiscounted evaluation episode return.
- Vector-control environments are run through ObjectRL via the project-side environment bridge.
- The DM Control observations for CartPole Swingup and Acrobot Swingup are flattened into 1-D `float32` vectors by `DMCGymAdapter`.
- CarRacing uses project-side CNN implementations because the image observation setup is not directly supported by ObjectRL's vector-observation baseline assumptions.
- CarRacing observations are preprocessed from HWC `uint8` images to CHW `float32` tensors in `[0,1]` with shape `(3,96,96)`.
- CarRacing CNN runs were executed on Google Colab with CUDA and copied back into the local repository.
- No hyperparameter optimization was performed for the midway baselines.
- The current implementation validates the pipeline and comparison surface, not final algorithmic superiority.
- The final project stage should add the GRPO-control variant and evaluate it against the completed baseline matrix.

## 6. Claims to Avoid

Avoid these claims in Methodology:

- Do not say CarRacing is deferred.
- Do not say PPO-CNN is not implemented.
- Do not say the final notebook is only a dry run.
- Do not imply that final GRPO-control experiments are already completed.
- Do not present a full GRPO-control algorithm as implemented.
- Do not claim state-of-the-art performance.
- Do not claim that PPO, SAC, or TD3 is generally superior based on the midway matrix.
- Do not claim that all algorithms have converged.
- Do not claim that no further experimental work is needed.
- Do not claim that hyperparameters are optimal or tuned.
- Do not cite debug runs as the main experiment.
- Do not hide the implementation split between vector-control and CarRacing.
- Do not describe ObjectRL as directly supporting the CarRacing image-observation setup used here.
- Do not introduce heavy Bellman contraction proofs or SSP properness arguments in Methodology; those belong in Theory if included at all.
- Do not over-specify implementation details that are not visible in the notebook or scripts.

## 7. Implementation Facts to Extract

Concrete facts visible from the notebook, scripts, and summary files:

- Main validation notebook: `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`.
- The notebook is the report-facing validation artifact and does not train or write results in its validation cells.
- Completed midway matrix: PPO, SAC, TD3 x `cartpole_swingup`, `acrobot_swingup`, `car_racing_continuous` x seeds `0..4`.
- Number of runs: 45.
- Number of expected CSV files: 45.
- Number of present combinations: 45 / 45.
- Number of missing combinations: 0.
- Number of evaluation rows: 900.
- Canonical result columns: `algorithm`, `project_env`, `seed`, `train_step`, `eval_episode`, `eval_return`, with `wall_time_seconds`, `status`, `observation_mode`, and `model_type` as bookkeeping.
- Evaluation metric: undiscounted evaluation episode return.
- No hyperparameter optimization was performed.

Vector-control pipeline:

- Script: `scripts/run_project_objectrl_baseline.py`.
- Environments: `cartpole_swingup`, `acrobot_swingup`.
- Environment factory: `scripts/project_envs.py`.
- Registration helper: `scripts/register_project_envs.py`.
- DM Control tasks: `cartpole/swingup` and `acrobot/swingup`.
- `DMCGymAdapter` wraps DM Control environments as Gymnasium-style environments.
- DM Control observation dictionaries are flattened into 1-D `float32` vectors.
- Actions are exposed as Gymnasium `Box` spaces.
- The ObjectRL runner monkey-patches the `make_env` symbol inside `objectrl.experiments.base_experiment` at runtime, without modifying files under `external/objectrl`.
- ObjectRL handles the training/evaluation loop, replay buffer, agent updates, and logging for vector-control baselines.
- Midway preset in `scripts/run_project_objectrl_baseline.py`: seeds `0..4`, `max_steps=20_000`, `eval_episodes=3`, `eval_frequency=5_000`, `warmup_steps=1_000` for off-policy algorithms; PPO warmup is forced to 0.
- Visible loaded rows for vector-control runs have evaluation steps from 0 to 19,999 in the notebook/summary outputs. CHECK NOTEBOOK OUTPUT before writing exact evaluation-step wording in final LaTeX, because ObjectRL's internal step indexing appears to use 0/4,999/9,999/14,999/19,999 rather than 5,000/10,000/15,000/20,000.

CarRacing CNN pipeline:

- Script: `scripts/run_carracing_cnn_baseline.py`.
- CNN implementation: `scripts/carracing_cnn.py`.
- Environment: `car_racing_continuous`, using `gymnasium.make("CarRacing-v3", continuous=True)`.
- Supported algorithms: SAC-CNN, TD3-CNN, PPO-CNN.
- Observation preprocessing: HWC `uint8` image to CHW `float32` tensor in `[0,1]`, shape `(3,96,96)`.
- CNN trunk: three convolutional layers followed by a linear layer to a 256-dimensional feature representation.
- SAC-CNN: tanh-Gaussian actor, twin Q-functions, automatic entropy, Polyak target updates, image replay buffer.
- TD3-CNN: deterministic actor, Gaussian exploration, twin critics, target smoothing, delayed policy updates.
- PPO-CNN: shared CNN trunk, Gaussian actor head, scalar value head, rollout buffer, GAE-lambda advantage estimation, clipped surrogate objective, value loss, entropy bonus, gradient clipping.
- Midway preset in `scripts/run_carracing_cnn_baseline.py`: seeds `0..4`, `max_steps=10_000`, `eval_episodes=3`, `eval_frequency=1_000`, `warmup_steps=500`.
- CarRacing evaluation steps in notebook outputs run from 1,000 to 10,000.
- CarRacing CNN runs were executed on Google Colab with CUDA and copied back into the local repository.

Why ObjectRL is not used directly for CarRacing:

- `scripts/run_project_objectrl_baseline.py` blocks CarRacing for the ObjectRL vector runner because the CarRacing image observation space is `(96,96,3)` and the ObjectRL baseline path expects 1-D vector observations.
- `scripts/register_project_envs.py` notes that custom Gymnasium IDs cannot simply be passed to ObjectRL's CLI without modifying `external/objectrl`, because ObjectRL's own environment factory applies internal environment-name constraints.
- The project-side CNN path avoids modifying the third-party ObjectRL checkout while still completing the CarRacing part of the baseline matrix.

Evaluation seed detail to standardize:

- `scripts/carracing_cnn.py` constructs CarRacing evaluation environments with `seed + 10_000`.
- `scripts/run_project_objectrl_baseline.py` constructs vector-control evaluation environments with `seed + 100`.
- The final notebook prose states that evaluation environments are constructed with `seed + 10_000`.
- Recommendation: Avoid specifying the exact offset in Methodology unless it is standardized. Use "separate seeded evaluation environments" in the draft prose. Before final LaTeX, decide whether to align the notebook prose with the scripts or to describe each pipeline separately.

## 8. Danish Draft

Ved midtvejsrapporten er formålet med metodologien at fastlægge den formelle ramme og den eksperimentelle protokol, som de foreløbige baseline-resultater bygger på. Projektet betragter de valgte kontrolopgaver som episodiske reinforcement-learning-problemer, hvor en agent interagerer med et miljø gennem observationer, kontinuerte handlinger og skalære belønninger. På dette stadie er den centrale metodiske leverance ikke en færdig GRPO-baseret kontrolalgoritme, men en reproducerbar baseline-protokol, som den endelige GRPO-udvidelse kan sammenlignes imod.

Vi beskriver et miljø som en Markov decision process
`\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0)`, hvor `\mathcal{S}` er tilstands- eller observationsrummet, `\mathcal{A}` er handlingsrummet, `P(s'\mid s,a)` er overgangskernen, `r(s,a)` er den forventede et-trinsbelønning, `\gamma` er diskonteringsfaktoren, og `\rho_0` er initialfordelingen. En politik beskrives enten som en stokastisk politik `\pi_\theta(a\mid s)` eller som en deterministisk politik `\mu_\theta(s)`. Denne skelnen passer til baseline-valget: PPO og SAC kan formuleres som stokastiske politikmetoder, mens TD3 naturligt beskrives som en deterministisk aktør-kritiker-metode med eksplorationsstøj under træning.

En episode beskrives som en trajektorie
`\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots,s_H)`. Til den teoretiske læringsformulering anvendes den diskonterede return
`G_t^\gamma=\sum_{k=0}^{H-t-1}\gamma^k R_{t+k+1}`. I de rapporterede eksperimenter bruges der derimod udiskonteret evalueringsreturn,
`G^{\mathrm{eval}}(\tau)=\sum_{t=0}^{H-1}R_{t+1}`. Denne adskillelse er vigtig, fordi træningsalgoritmerne kan anvende diskontering internt, mens læringskurverne i midtvejsrapporten viser observeret udiskonteret episode-return under evaluering.

Midtvejsprotokollen består af tre algoritmer, tre miljøer og fem seeds. Algoritmerne er PPO, SAC og TD3, og miljøerne er `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous`. Dermed består den fulde baseline-matrix af 45 kørsler. Den rapportvendte notebook validerer, at alle forventede kombinationer findes på disk, og at de tilsammen giver 900 evalueringsrækker. Ingen hyperparameteroptimering er udført, så resultaterne bør forstås som foreløbige baseline-resultater ved den valgte midtvejsprotokol.

Implementeringen bruger to forskellige, men bevidst adskilte, pipelines. De to vektorbaserede kontrolmiljøer køres gennem ObjectRL. Her anvendes en projekt-side adapter, som indlæser DM Control-opgaverne og eksponerer dem som Gymnasium-kompatible miljøer med fladtrykte `float32`-observationer. Dette gør det muligt at bruge ObjectRL's eksisterende trænings- og evalueringslogik uden at ændre den eksterne ObjectRL-kode.

CarRacing behandles separat, fordi miljøet anvender billedobservationer i stedet for 1-dimensionelle vektorobservationer. Til dette formål indeholder projektet egne CNN-baserede implementationer af PPO, SAC og TD3. Observationerne preprocesses fra `uint8`-billeder i HWC-format til `float32`-tensorer i CHW-format med værdier i intervallet `[0,1]`. CarRacing-kørslerne blev udført på Google Colab med CUDA og derefter kopieret tilbage til repository'et. Denne opdeling betyder, at alle tre miljøer indgår i midtvejsmatrixen, uden at den eksterne ObjectRL-kode ændres.

Den endelige GRPO-relaterede kontrolmetode er planlagt til næste projektfase. Metodologien i midtvejsrapporten skal derfor ikke præsentere en færdig GRPO-algoritme, men gøre det klart, hvilket problem, hvilken notation og hvilken evalueringsflade den senere metode skal bygges oven på. De nuværende resultater validerer primært den eksperimentelle pipeline og baseline-sammenligningen; de bør ikke tolkes som endelige rangeringer af algoritmerne.

## 9. English Academic Draft

At the midway stage, the purpose of the methodology is to fix the formal reinforcement-learning notation and to document the experimental protocol used to produce the baseline matrix. The selected tasks are treated as episodic control problems in which an agent interacts with an environment through observations, continuous actions, and scalar rewards. The present contribution is not yet a completed GRPO-control algorithm. Instead, the methodology establishes the reproducible baseline protocol that the final GRPO extension will later be compared against.

We describe each task using Markov decision process notation,
`\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0)`, where `\mathcal{S}` denotes the state or observation space, `\mathcal{A}` the action space, `P(s'\mid s,a)` the transition kernel, `r(s,a)` the expected one-step reward, `\gamma` the discount factor, and `\rho_0` the initial-state distribution. Policies are written either as stochastic policies `\pi_\theta(a\mid s)` or deterministic policies `\mu_\theta(s)`. This distinction matches the baseline set: PPO and SAC are naturally described with stochastic-policy notation, while TD3 is naturally described as a deterministic actor-critic method with exploration noise during training.

An episode is represented by a trajectory
`\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots,s_H)`. For the learning objective, the discounted return may be written as
`G_t^\gamma=\sum_{k=0}^{H-t-1}\gamma^k R_{t+k+1}`. For reporting, however, the experiments use the undiscounted evaluation episode return,
`G^{\mathrm{eval}}(\tau)=\sum_{t=0}^{H-1}R_{t+1}`. This separation is important: the algorithms may use discounted objectives internally, whereas the midway learning curves report observed undiscounted evaluation returns.

The midway protocol covers PPO, SAC, and TD3 on three environments: `cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous`. Each algorithm-environment pair is run with five seeds, producing a full matrix of 45 runs. The report-facing notebook validates that all expected combinations are present and that the completed matrix contains 900 evaluation rows. No hyperparameter optimization was performed, so the results should be interpreted as preliminary baselines under the selected midway protocol.

Two implementation paths are used. The vector-control environments are trained through ObjectRL using a project-side environment bridge. The bridge loads the DM Control tasks, flattens their observation dictionaries into one-dimensional `float32` vectors, and exposes a Gymnasium-compatible interface. This allows the project to reuse ObjectRL's training, evaluation, replay-buffer, and logging infrastructure without modifying the external ObjectRL checkout.

CarRacing is handled separately because the environment uses image observations rather than one-dimensional vector observations. The project therefore includes CNN-based PPO, SAC, and TD3 implementations for `CarRacing-v3` with continuous actions. The preprocessing converts HWC `uint8` image observations to CHW `float32` tensors in `[0,1]`. The CarRacing CNN runs were executed on Google Colab with CUDA and copied back into the local repository. This project-side path completes the CarRacing portion of the baseline matrix while keeping the ObjectRL dependency unmodified.

The GRPO-control extension is planned for the final project stage. Consequently, the midway methodology should be read as defining the notation, implementation boundaries, and comparison surface for that later extension. The completed baseline matrix validates the experimental pipeline, but it should not be interpreted as establishing final algorithm superiority or convergence.

## 10. Suggested Equations

### Equation 1: MDP Definition

Purpose: Define the formal problem object used throughout Methodology.

Plain explanation: A task is represented by state or observation space, action space, transition law, reward function, discount factor, and initial-state distribution.

LaTeX form:

```latex
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0).
```

One-sentence explanation: This defines the reinforcement-learning environment that the agent interacts with.

Placement: Methodology.

### Equation 2: Markov Transition Property

Purpose: State the Markov assumption briefly if needed.

Plain explanation: Given the present state and action, the next-state distribution does not depend on the earlier history.

LaTeX form:

```latex
\Pr(S_{t+1}=s' \mid S_t=s,A_t=a,S_{t-1},A_{t-1},\ldots)
= P(s'\mid s,a).
```

One-sentence explanation: This is the property that lets the report model the control tasks as MDPs.

Placement: Methodology if space allows; otherwise Theory.

### Equation 3: Trajectory

Purpose: Define the sequence used for returns and evaluation episodes.

Plain explanation: A trajectory is one realized episode of states, actions, and rewards.

LaTeX form:

```latex
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots,s_H).
```

One-sentence explanation: The reward index `r_{t+1}` follows the local RL notation notebooks.

Placement: Methodology.

### Equation 4: Discounted Return

Purpose: Define the learning-objective return.

Plain explanation: Future rewards are geometrically discounted by `\gamma`.

LaTeX form:

```latex
G_t^\gamma=\sum_{k=0}^{H-t-1}\gamma^k R_{t+k+1}.
```

One-sentence explanation: This return is used to define value functions and the standard RL objective.

Placement: Methodology.

### Equation 5: Undiscounted Evaluation Return

Purpose: Define the metric plotted and summarized in the midway experiments.

Plain explanation: The evaluation score is the sum of rewards observed over one evaluation episode.

LaTeX form:

```latex
G^{\mathrm{eval}}(\tau)=\sum_{t=0}^{H-1}R_{t+1}.
```

One-sentence explanation: This distinguishes the reported metric from the discounted training objective.

Placement: Methodology or Experiments. Include in Methodology if the section defines evaluation protocol.

### Equation 6: Value and Action-Value Functions

Purpose: Define the notation needed for actor-critic baselines and later GRPO-control discussion.

Plain explanation: `V` evaluates a state under a policy, while `Q` evaluates a state-action pair.

LaTeX form:

```latex
V^\pi(s)=\mathbb{E}_\pi[G_t^\gamma \mid S_t=s],
\qquad
Q^\pi(s,a)=\mathbb{E}_\pi[G_t^\gamma \mid S_t=s,A_t=a].
```

One-sentence explanation: These functions provide the common language for PPO, SAC, TD3, and advantage-based updates.

Placement: Methodology.

### Equation 7: Advantage Function

Purpose: Define the improvement signal used by policy-gradient and advantage-based methods.

Plain explanation: The advantage compares an action-value to the baseline value of the state.

LaTeX form:

```latex
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
```

One-sentence explanation: This is especially useful for explaining PPO/GAE and later GRPO-related objectives.

Placement: Methodology.

### Equation 8: Empirical Mean Evaluation Return

Purpose: Define how evaluation curves are aggregated across seeds and evaluation episodes.

Plain explanation: At a given training step, average evaluation return across seeds and evaluation episodes.

LaTeX form:

```latex
\bar{J}_{\alpha,e}(n)
=
\frac{1}{|\mathcal{Z}|M}
\sum_{z\in\mathcal{Z}}\sum_{m=1}^{M}
G^{\mathrm{eval}}_{\alpha,e,z,n,m}.
```

One-sentence explanation: This describes the mean curve for algorithm `\alpha` on environment `e` at evaluation step `n`.

Placement: Methodology if the protocol subsection includes aggregation; otherwise Experiments.

Equations to postpone to Theory:

- Bellman expectation and Bellman optimality equations unless the Methodology section needs them for notation.
- Contraction arguments for discounted MDPs.
- Finite-horizon backward-induction results.
- SSP properness conditions from the episodic notebook.
- Full GRPO objective or pseudocode until the final GRPO-control method is implemented.

## 11. Suggested Citation Placeholders

Existing BibTeX keys in `report/references.bib` that can be used:

- PPO: `\citep{schulman2017proximal}`
- GAE, if advantage estimation is discussed: `\citep{schulman2015gae}`
- SAC: `\citep{haarnoja2018sacapps}`
- TD3: `\citep{fujimoto2018td3}`
- Deterministic policy gradient background, if needed: `\citep{silver2014deterministic}`
- DDPG as TD3 predecessor, if needed: `\citep{lillicrap2015continuous}`
- GRPO/DeepSeekMath: `\citep{shao2024deepseekmath}`
- DeepSeek-R1, if connecting GRPO to later reasoning-model work: `\citep{deepseekai2025deepseekr1}`
- ObjectRL: `\citep{baykal2025objectrl}`
- Gymnasium: `\citep{towers2024gymnasium}`
- DM Control Suite: `\citep{tunyasuvunakool2020dmcontrol}`

Where to place citations:

- MDP/RL problem formulation: use a descriptive placeholder such as `[MDP/RL textbook or course material]` unless a suitable BibTeX key is added. I did not find an obvious Sutton/Barto or Puterman-style key in `references.bib`.
- PPO/SAC/TD3 baseline paragraph: cite PPO, SAC, and TD3 keys together.
- Advantage/GAE paragraph: cite `schulman2015gae` if GAE is mentioned explicitly.
- Environment paragraph: cite Gymnasium and DM Control where the environment sources are introduced.
- ObjectRL implementation paragraph: cite `baykal2025objectrl`.
- Planned GRPO-control paragraph: cite `shao2024deepseekmath`, but make clear that the final control variant is planned and not yet implemented.

Avoid inventing BibTeX keys. If a desired source is not already in `references.bib`, add it later through the references plan rather than using a guessed key.

## 12. Final Author Notes

Manual decisions to make:

- Decide whether the report's main MDP tuple should be `(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0)` or whether to mirror the finite-horizon tuple from the theory notebook. I recommend the compact discounted tuple with `\rho_0`, plus a separate finite episode length `H`.
- Decide whether the report uses `R_{t+1}` or `r_t` reward indexing. The local notation guide uses `R_{t+1}` in the main RL notation, so I recommend using that consistently in final LaTeX.
- Decide whether `\tau` is reserved for trajectories. I recommend using `\tau` for trajectories and `H` or `T_{\mathrm{ep}}` for episode length, because one local theory notebook also uses `\tau` as a hitting time.
- Decide whether to state the exact evaluation seed offset. The notebook prose and scripts are not fully aligned: CarRacing uses `seed+10_000`, while the ObjectRL vector runner uses `seed+100`. Unless standardized, write only that evaluation uses separate seeded evaluation environments.
- Decide how much algorithm detail belongs in Methodology versus Related Work and Theory. Methodology should define roles and protocol; detailed objectives can be kept in Theory or later final-stage method text.

Check against the assignment PDF:

- The assignment asks for MDP notation in the interim/midway report. Ensure the final Methodology explicitly satisfies this.
- The assignment frames final evaluation around GRPO, PPO, SAC, and TD3; the midway report should state that PPO/SAC/TD3 baselines are complete and GRPO is planned for the final stage.
- The assignment expects five seeds and learning curves of observed undiscounted evaluation episode return versus training steps. Ensure Methodology and Experiments use the same wording.
- The assignment mentions Python/PyTorch; include this only if useful and supported by Methodology/Experiments context.

Align later with Related Work, Experiments, and Theory:

- Related Work should motivate PPO, SAC, TD3, ObjectRL, Gymnasium/DM Control, and GRPO citations without duplicating the protocol.
- Theory should contain any deeper MDP, Bellman, policy-gradient, or actor-critic equations that would make Methodology too heavy.
- Experiments should hold the numeric result interpretation, rankings, figure discussion, stability observations, and limitations.
- Methodology should stay focused on the formal setup and the reproducible protocol.

Postpone until the final report:

- full GRPO-control objective;
- GRPO-control pseudocode;
- implementation details of the final GRPO-control variant;
- final GRPO-vs-baseline result matrix;
- hyperparameter optimization claims;
- statistical significance claims;
- final algorithm ranking claims.

Notation standardization from local theory notebooks:

- The notebooks mix finite-horizon MDP notation, discounted infinite-horizon notation, and episodic/SSP notation. This is normal for learning notes, but the report should choose one notation pathway.
- The symbol `\tau` is overloaded across notebooks. Standardize it before writing final LaTeX.
- The notebooks use both `p_0` and, in planning docs, `\rho_0`-style initial distributions. Pick one.
- The local value-function notation is compatible with `V^\pi`, `Q^\pi`, and `A^\pi`; keep these unchanged.
