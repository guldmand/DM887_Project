# Experiments — Inspiration Draft (claude)

This document is **inspiration material** for the Experiments section of the
DM887 midway report. It is not final report text. The final wording lives in
`report/sections/05_experiments.tex` and must be written manually.

Project: *Group Relative Policy Optimization for Control* (DM887, Spring 2026).
Stage: midway / interim report.

---

## 1. Experiments section goal analysis

In this **midway** report the Experiments section has to do four things, and it
has to do them without overclaiming:

1. **Document the experimental setup.** State the algorithms (PPO, SAC, TD3),
   environments (`cartpole_swingup`, `acrobot_swingup`,
   `car_racing_continuous`), seeds (0..4), training budgets, and evaluation
   protocol (undiscounted episode return, fixed eval cadence).
2. **Document reproducibility and the two-track implementation.** Vector
   environments are run through ObjectRL via the project-side runner
   (`scripts/run_project_objectrl_baseline.py`). CarRacing is run through a
   project-side PyTorch CNN runner (`scripts/run_carracing_cnn_baseline.py`,
   `scripts/carracing_cnn.py`) on Google Colab with CUDA, with CSVs copied
   back into the local repo. The reader must understand why the project uses
   two tracks and that no `external/objectrl` code was modified.
3. **Validate the completed midway baseline matrix.** State that the matrix
   3 algorithms × 3 environments × 5 seeds = 45 runs is complete, that
   processed CSVs were aggregated by `scripts/summarize_project_baselines.py`
   into `results/processed/project_baselines/midway_vector_summary.csv`, and
   that the report-facing notebook
   `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` validates the matrix.
4. **Interpret the baseline learning curves cautiously.** Describe the
   qualitative pattern visible in the existing midway figures, but explicitly
   limit the interpretation to the midway training budget and to the absence
   of hyperparameter optimization. The midway runs are evidence about the
   pipeline, not a final algorithm ranking.

Length target: roughly **1.5–3 pages** in the NeurIPS template, with three or
four figures. Keep it tight; this is not the final experiments section.

What this section should explicitly **not** do at the midway stage:

- It must not present GRPO results.
- It must not claim that any algorithm has converged.
- It must not present per-seed numerical tables that the reader cannot
  re-derive from the public artefacts in the repo.
- It must not contain new experiments that are not already represented by
  CSVs and figures in the repo.

---

## 2. Proposed subsection structure (6 subsections)

Suggested layout for the midway Experiments section.

### 5.1 Experimental setup
- **Purpose.** Single overview paragraph that names algorithms, environments,
  seeds, evaluation metric, and the two-track implementation split (ObjectRL
  for vector control, project-side CNN for CarRacing).
- **Key message.** The midway experiments instantiate a single, reproducible
  baseline pipeline that is shared across all three algorithms.
- **Connects to.** README, `scripts/run_project_objectrl_baseline.py`,
  `scripts/run_carracing_cnn_baseline.py`, and the report-facing notebook.
- **Midway / final.** Midway. The same paragraph will be reused (with budget
  bumps) in the final report.

### 5.2 Environments
- **Purpose.** Describe `cartpole_swingup`, `acrobot_swingup`, and
  `car_racing_continuous` at the level needed to read the curves.
- **Key message.** Two vector-control swingup tasks plus an image-observation
  driving task; CarRacing is the only image-observation environment and is the
  reason the project ships its own CNN-based runner.
- **Connects to.** `scripts/project_envs.py` (registration), `scripts/carracing_cnn.py`
  (`CarRacingCNNWrapper`), Gymnasium documentation.
- **Midway / final.** Midway. Wording is final-report ready.

### 5.3 Algorithms and implementation
- **Purpose.** Specify which implementation provides each algorithm and what
  was *not* changed.
- **Key message.** PPO, SAC, and TD3 on the vector envs come from ObjectRL via
  a project-side bridge that monkey-patches `make_env` in
  `objectrl.experiments.base_experiment` so that the project envs flow
  through, without modifying any `external/objectrl/` code. PPO-CNN, SAC-CNN,
  and TD3-CNN for CarRacing live in `scripts/carracing_cnn.py` because
  ObjectRL asserts 1-D Box observations.
- **Connects to.** `scripts/run_project_objectrl_baseline.py` (in particular
  the docstring, the monkey-patch comment, and the `RUN_CONFIG` dictionary),
  `scripts/carracing_cnn.py`, `external/objectrl/`.
- **Midway / final.** Midway. The final report can simply add the GRPO entry.

### 5.4 Evaluation protocol
- **Purpose.** Specify the metric (undiscounted evaluation episode return),
  the eval cadence, the number of eval episodes per snapshot, the number of
  seeds, and how seeds are aggregated (mean across seeds; std band when
  ≥2 seeds, as in `summarize_project_baselines._plot_env`).
- **Key message.** The protocol is fixed before any run starts and applies
  uniformly to the three algorithms, so cross-algorithm comparison at a fixed
  training step is meaningful at the chosen budget — even though the absolute
  numbers should not be read as final.
- **Connects to.** `scripts/run_project_objectrl_baseline.py:RUN_CONFIG`,
  `scripts/run_carracing_cnn_baseline.py:RUN_CONFIG`,
  `scripts/summarize_project_baselines.py` (aggregation logic).
- **Midway / final.** Midway. The eval cadence will likely change in the final
  report's longer training budget.

### 5.5 Result validation and the completed baseline matrix
- **Purpose.** Demonstrate that the matrix is complete and that the artefacts
  on disk match the matrix. Mention the report-facing notebook explicitly as
  the source of truth.
- **Key message.** The 3 × 3 × 5 = 45-run midway matrix is complete; CSVs are
  present for every (algorithm, environment, seed) combination; the
  aggregation step produces `midway_vector_summary.csv`; the report-facing
  notebook validates these artefacts and renders the displayed figures.
- **Connects to.** `results/processed/project_baselines/midway_vector_summary.csv`,
  `figures/midway/`, `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`.
- **Midway / final.** Midway. The exact phrasing ("matrix complete",
  "pipeline validated") will be replaced with GRPO-extended phrasing in the
  final report.

### 5.6 Midway baseline results, interpretation, and limitations
- **Purpose.** Show the four midway figures and interpret them carefully.
- **Key message.** At the midway training budget, the curves show a clear
  qualitative pattern on `cartpole_swingup` (SAC and TD3 reach noticeably
  higher mean returns than PPO by the end of the budget), an unsolved pattern
  on `acrobot_swingup` (all three algorithms remain at low returns within the
  midway budget), and a learning-in-progress pattern on
  `car_racing_continuous` (returns remain negative for all algorithms at the
  midway budget). These observations describe the present pipeline state,
  not a definitive algorithmic ranking. No hyperparameter optimization was
  performed and the training budget is intentionally short for the midway
  stage.
- **Connects to.** All four figures in `figures/midway/` and the notebook.
- **Midway / final.** Midway. This subsection becomes much longer in the
  final report once GRPO is added and a full training budget is run.

(Optional 5.7 — *Reproduction*. A two-line note pointing to the exact bash
incantations in the README. Useful for graders, optional for length.)

---

## 3. Safe claims (supported by current repo state)

These claims are safe in the Experiments section.

**Setup**

- Three algorithms (PPO, SAC, TD3) are evaluated on three environments
  (`cartpole_swingup`, `acrobot_swingup`, `car_racing_continuous`) using
  five seeds (0..4).
- The completed midway baseline matrix consists of 3 × 3 × 5 = 45 runs.
- Each run produces one `midway_<algo>_<env>_seed<seed>_eval.csv` under
  `results/processed/project_baselines/`.
- Vector environments (`cartpole_swingup`, `acrobot_swingup`) are run through
  ObjectRL using `scripts/run_project_objectrl_baseline.py`.
- The vector path uses a project-side bridge that monkey-patches the `make_env`
  symbol already imported by `objectrl.experiments.base_experiment`. No
  `external/objectrl/` code is modified.
- CarRacing (`car_racing_continuous`) is run through the project-side CNN
  pipeline (`scripts/run_carracing_cnn_baseline.py`,
  `scripts/carracing_cnn.py`) because ObjectRL asserts 1-D Box observations
  and CarRacing returns `(96, 96, 3)` uint8 image observations.
- The CarRacing CNN runs were executed on Google Colab with CUDA and copied
  back into the local repository.

**Protocol**

- The evaluation metric is the undiscounted episode return.
- For the vector environments, the midway training budget is `max_steps =
  20000` per run, with `eval_frequency = 5000` and `eval_episodes = 3` per
  snapshot, plus an evaluation at step 0; PPO uses `learn_frequency = 256`
  and `warmup_steps = 0`, and SAC/TD3 use `warmup_steps = 1000` (see
  `RUN_CONFIG["midway"]` in `scripts/run_project_objectrl_baseline.py`).
- For CarRacing, the midway budget is `max_steps = 10000`, with
  `eval_frequency = 1000`, `eval_episodes = 3`, and `warmup_steps = 500`
  (see `RUN_CONFIG["midway"]` in `scripts/run_carracing_cnn_baseline.py`).
- Curves in the figures show the per-seed mean return across the three
  evaluation episodes per snapshot, then the cross-seed mean and std band
  produced by `_plot_env` in `scripts/summarize_project_baselines.py`.

**Validation**

- Aggregating the 45 CSVs with `scripts/summarize_project_baselines.py
  --prefix midway` produces
  `results/processed/project_baselines/midway_vector_summary.csv`.
- The same script writes the per-environment plots and the combined
  vector-env panel under `figures/midway/`.
- The notebook `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` validates the
  matrix and renders the displayed figures and summary tables.

**Cautious results language**

- At the midway training budget, the qualitative ordering on
  `cartpole_swingup` shows SAC and TD3 reaching higher mean evaluation
  returns than PPO by the end of the budget.
- At the midway training budget, none of the three algorithms appears to
  solve `acrobot_swingup`; mean returns remain low across seeds.
- At the midway training budget, mean evaluation returns on
  `car_racing_continuous` remain negative for all three algorithms.
- The observed differences should not be interpreted as final algorithm
  rankings. No hyperparameter optimization was performed and the training
  budget is intentionally short.

---

## 4. Claims to avoid (would overclaim the current midway state)

Do **not** write any of the following in the Experiments section:

- "The midway results show that algorithm X is superior in general."
- "PPO/SAC/TD3 were tuned for these environments."
- "We performed a grid/Bayesian/sweep over hyperparameters."
- "All algorithms converged."
- "These are state-of-the-art results."
- "CarRacing was deferred to the final report."
- "PPO-CNN was not implemented" or "TD3-CNN was not implemented".
- "The notebook is a dry-run only."
- "GRPO outperforms / underperforms the baselines" (no GRPO yet).
- "The variance estimates are tight" (only 5 seeds, eval band is descriptive).
- "The midway training budget is sufficient to draw final conclusions."
- Any statement that the midway results disprove or prove a hypothesis about
  the GRPO project.
- Any per-seed numbers presented as definitive performance evidence rather
  than as illustration.

When in doubt, frame each result as "at the midway budget" and "with no
hyperparameter optimization", and let the qualitative pattern speak.

---

## 5. Result facts to extract

The following facts are **directly visible** in the repository artefacts. Use
them for the section. Where exact numbers depend on the notebook output,
the entry is marked **CHECK NOTEBOOK OUTPUT** so that you verify against
`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` before printing them.

**Counting the matrix**

- Number of CSV files in `results/processed/project_baselines/` matching
  `midway_*_eval.csv`: **45**.
  - PPO: 3 envs × 5 seeds = 15 files.
  - SAC: 3 envs × 5 seeds = 15 files.
  - TD3: 3 envs × 5 seeds = 15 files.
- Number of `(algorithm, environment, seed)` combinations expected:
  3 × 3 × 5 = **45** (matches `EXPECTED_COMBINATIONS` in
  `scripts/summarize_project_baselines.py`).
- Number of evaluation snapshots per run:
  - Vector envs: 5 snapshots per seed (steps 0, 5000, 10000, 15000, 19999),
    so 5 snapshots × 5 seeds × 3 algorithms × 2 envs = **150 vector
    snapshots**.
  - CarRacing: 10 snapshots per seed (steps 1000..10000 in steps of 1000),
    so 10 snapshots × 5 seeds × 3 algorithms × 1 env = **150 CarRacing
    snapshots**.
  - Aggregated total in `midway_vector_summary.csv`: **300 rows**
    (150 vector + 150 CarRacing). Each row is the mean over
    `n_repeats = 3` evaluation episodes.
- "Total result rows: 900" reported in the project status corresponds to the
  raw per-eval-episode rows (300 snapshots × 3 evaluation episodes per
  snapshot). **CHECK NOTEBOOK OUTPUT** for the exact way the notebook
  reports this number.

**Algorithms, environments, seeds**

- Algorithms: `ppo`, `sac`, `td3`.
- Environments: `cartpole_swingup`, `acrobot_swingup`,
  `car_racing_continuous`.
- Seeds: 0, 1, 2, 3, 4.

**Training budgets and evaluation cadence**

- Vector envs (`cartpole_swingup`, `acrobot_swingup`):
  - `max_steps = 20000`, `eval_frequency = 5000`, `eval_episodes = 3`.
  - PPO: `warmup_steps = 0`, `learn_frequency = 256`.
  - SAC, TD3: `warmup_steps = 1000`, `learn_frequency = 1`.
- CarRacing (`car_racing_continuous`):
  - `max_steps = 10000`, `eval_frequency = 1000`, `eval_episodes = 3`,
    `warmup_steps = 500`.
- Source: `RUN_CONFIG["midway"]` in
  `scripts/run_project_objectrl_baseline.py` and
  `scripts/run_carracing_cnn_baseline.py`.

**Aggregation**

- `scripts/summarize_project_baselines.py` groups by
  `(algorithm, project_env, seed, train_step, status)` and reports the mean
  `eval_return` together with `n_repeats`. The midway summary uses
  `n_repeats = 3` for every snapshot.

**Figure files to reference**

- `figures/midway/midway_cartpole_swingup_baselines.png`
- `figures/midway/midway_acrobot_swingup_baselines.png`
- `figures/midway/midway_car_racing_continuous_baselines.png`
- `figures/midway/midway_vector_env_baselines.png`

**Qualitative patterns visible in the summary CSV**

These are *qualitative* observations from `midway_vector_summary.csv`. Cross-
check the exact phrasing against the notebook's tables before citing
absolute numbers.

- `cartpole_swingup` at step 19999:
  - SAC mean returns are noticeably higher than PPO and reach the highest
    end-of-budget values across seeds.
  - TD3 reaches comparable or moderately lower end-of-budget returns to SAC.
  - PPO end-of-budget returns are more variable and lower on average.
- `acrobot_swingup` at step 19999:
  - All three algorithms remain at low mean returns; no algorithm appears to
    solve the task at this budget.
- `car_racing_continuous` at step 10000:
  - PPO mean returns sit near roughly -90 across seeds.
  - SAC mean returns are higher (less negative) than PPO and TD3 on most
    seeds, with substantial seed-to-seed variability.
  - TD3 is variable across seeds, ranging from large-negative on some seeds
    to less-negative on others.

**Stability / progress facts (CHECK NOTEBOOK OUTPUT)**

- The notebook may compute per-environment progress tables (Δ between first
  and last eval), per-environment stability tables (cross-seed std at the
  final eval), and a ranking table. Use the notebook's wording and numbers
  rather than re-deriving them in the section. **CHECK NOTEBOOK OUTPUT.**

---

## 6. Danish draft

> **Bemærk.** Akademisk sprog, men læsbart. Brug dette som udgangspunkt og
> redigér selv. Citationsmarkeringer er sat med `\citep{...}` og figur-
> referencer er kun pladsholdere.

**5. Eksperimenter**

**5.1 Eksperimentel opsætning.**
Midtvejseksperimenterne instantierer en enkelt, reproducerbar baseline-
pipeline for tre algoritmer (PPO \citep{schulman2017proximal} med GAE
\citep{schulman2015gae}, SAC \citep{haarnoja2018sacapps} og TD3
\citep{fujimoto2018td3}) på tre miljøer fra projektets
påkrævede sæt: `cartpole_swingup`, `acrobot_swingup` og
`car_racing_continuous`. Hver kombination af algoritme og miljø køres med
fem seeds (0–4), således at den fuldførte midtvejsmatrix består af
3 × 3 × 5 = 45 kørsler. De vektorbaserede miljøer afvikles gennem
ObjectRL \citep{baykal2025objectrl} via en projekt-egen bro, der ikke
modificerer `external/objectrl/`. CarRacing afvikles gennem en projekt-egen
CNN-pipeline, da ObjectRL forudsætter 1-D `Box`-observationer, mens
CarRacing returnerer `(96, 96, 3)` uint8-billeder. CarRacing CNN-kørslerne
blev udført på Google Colab med CUDA og kopieret tilbage til det lokale
repository.

**5.2 Miljøer.**
`cartpole_swingup` og `acrobot_swingup` er klassiske swing-up-opgaver med
kontinuerlige handlinger og lav-dimensionelle vektor-observationer.
`car_racing_continuous` (CarRacing-v3 \citep{towers2024gymnasium}) er en
billedbaseret kontinuerlig kontrolopgave; observationer er `(96, 96, 3)`
uint8-billeder, der i CNN-pipelinen normaliseres og transponeres til
`(3, 96, 96)` float32 i `[0, 1]` af `CarRacingCNNWrapper`.

**5.3 Algoritmer og implementation.**
PPO, SAC og TD3 på de vektorbaserede miljøer leveres af ObjectRL gennem
`scripts/run_project_objectrl_baseline.py`. Bridge-strategien er at
monkey-patche det `make_env`-symbol, som
`objectrl.experiments.base_experiment` allerede har importeret, sådan at
træning og evaluering bygger projektets miljøer i stedet for ObjectRL's
hvidlistede sæt. Selve trænings- og evalueringsløkkerne er ObjectRL's;
algoritmerne reimplementeres ikke. CarRacing kræver en CNN-baseret
politik. Da ObjectRL's `base_agent` har en `assert` på 1-D
observationer, leverer projektet PPO-, SAC- og TD3-varianter med en
delt CNN-feature-extraktor i `scripts/carracing_cnn.py`, og kørslerne
styres af `scripts/run_carracing_cnn_baseline.py`.

**5.4 Evaluerings-protokol.**
Den primære måleenhed er udiskonteret evalueringsepisode-return. For de
vektorbaserede miljøer trænes hver kørsel i 20\,000 skridt med evaluering
hver 5\,000 skridt og tre evalueringsepisoder per snapshot. PPO bruger
`learn_frequency = 256` og ingen warm-up; SAC og TD3 bruger 1\,000 warm-up-
skridt. For CarRacing trænes hver kørsel i 10\,000 skridt med evaluering
hver 1\,000 skridt og 500 warm-up-skridt. Læringskurver beregnes som
gennemsnit på tværs af seeds; et standardafvigelsesbånd vises, når mindst
to seeds bidrager (jf. `_plot_env` i
`scripts/summarize_project_baselines.py`).

**5.5 Resultatvalidering og fuldført baseline-matrix.**
De 45 kørsler producerer hver én CSV i
`results/processed/project_baselines/` på formen
`midway_<algoritme>_<miljø>_seed<seed>_eval.csv`. Aggregering med
`scripts/summarize_project_baselines.py --prefix midway` skriver
sammenfatningen `midway_vector_summary.csv` og figurene under
`figures/midway/`. Den rapport-rettede notebook
`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` indlæser disse artefakter,
verificerer at alle 45 kombinationer er til stede og gengiver de figurer og
sammenfatningstabeller, der vises her. Den fuldførte matrix etablerer
sammenligningsgrundlaget for den endelige GRPO-fase.

**5.6 Midtvejs-baseline-resultater og fortolkning.**
Figur~\ref{fig:midway-vector-panel} viser de tre baseline-læringskurver
side om side. På `cartpole_swingup` (Figur~\ref{fig:midway-cartpole}) når
SAC og TD3 markant højere middel-evalueringsreturns end PPO ved slutningen
af midtvejs-budgettet; PPO viser større seed-variation og lavere
slut-returns. På `acrobot_swingup` (Figur~\ref{fig:midway-acrobot})
forbliver alle tre algoritmer på lave middel-returns inden for
20\,000-skridt-budgettet, hvilket indikerer, at ingen af baselines løser
opgaven på dette budget. På `car_racing_continuous`
(Figur~\ref{fig:midway-carracing}) er middel-returns negative for alle
algoritmer ved 10\,000-skridt-budgettet; SAC opnår mindre-negative returns
end PPO og TD3 på de fleste seeds, men variansen på tværs af seeds er
betydelig. Disse observationer skal læses som et øjebliksbillede af den
nuværende pipeline ved midtvejs-budgettet og under den eksplicitte
betingelse, at *ingen hyperparameteroptimering blev udført*. De udgør ikke
en endelig rangering af algoritmerne. Den endelige projektfase vil udvide
træningsbudgettet, tilføje GRPO-control-varianten og foretage den endelige
sammenligning.

---

## 7. English academic draft

> **Note.** Concise, report-facing. Citation placeholders use `\citep{...}`
> and figure references use `Figure~\ref{...}` placeholders only.

**5 Experiments**

**5.1 Experimental setup.**
The midway experiments instantiate a single, reproducible baseline pipeline
for three algorithms — PPO \citep{schulman2017proximal} with GAE
\citep{schulman2015gae}, SAC \citep{haarnoja2018sacapps}, and TD3
\citep{fujimoto2018td3} — on the three required environments
`cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous`. Every
algorithm-environment pair is trained with five seeds (0..4), so the
completed midway matrix consists of $3 \times 3 \times 5 = 45$ runs. The
vector environments are run through ObjectRL \citep{baykal2025objectrl} via
a project-side bridge that does not modify any code under
`external/objectrl/`. CarRacing is run through a project-side CNN pipeline,
because ObjectRL asserts 1-D `Box` observations while CarRacing returns
`(96, 96, 3)` uint8 images. The CarRacing CNN runs were executed on Google
Colab with CUDA and copied back into the local repository.

**5.2 Environments.**
`cartpole_swingup` and `acrobot_swingup` are continuous-action swing-up
tasks with low-dimensional vector observations. `car_racing_continuous`
wraps `CarRacing-v3` \citep{towers2024gymnasium}, a continuous-action
driving task with `(96, 96, 3)` uint8 image observations. The project-side
`CarRacingCNNWrapper` normalizes and transposes the observations to
`(3, 96, 96)` float32 in $[0, 1]$ before they enter the CNN feature
extractor.

**5.3 Algorithms and implementation.**
The vector-environment baselines reuse the PPO, SAC, and TD3 implementations
shipped with ObjectRL through `scripts/run_project_objectrl_baseline.py`.
The bridge monkey-patches the `make_env` symbol that
`objectrl.experiments.base_experiment` has already imported, so that
ObjectRL's training and evaluation loops construct the project's
environments instead of its whitelisted set. None of the algorithms is
reimplemented in this path. CarRacing requires a CNN-based policy, so the
project ships a small PyTorch implementation of PPO, SAC, and TD3 with a
shared CNN trunk in `scripts/carracing_cnn.py`, driven by
`scripts/run_carracing_cnn_baseline.py`. This separation is a deliberate
consequence of the project's no-modification policy on `external/objectrl/`.

**5.4 Evaluation protocol.**
The primary metric is the undiscounted evaluation episode return. For the
vector environments each run trains for $20\,000$ steps with an evaluation
every $5\,000$ steps and three evaluation episodes per snapshot. PPO uses
`learn_frequency = 256` and no warm-up; SAC and TD3 use $1\,000$ warm-up
steps. For CarRacing each run trains for $10\,000$ steps with evaluation
every $1\,000$ steps, three evaluation episodes per snapshot, and $500$
warm-up steps. Learning curves are computed as the per-seed mean across the
three evaluation episodes per snapshot, then averaged across seeds; a
standard-deviation band is shown when at least two seeds contribute (see
`_plot_env` in `scripts/summarize_project_baselines.py`).

**5.5 Result validation and the completed baseline matrix.**
Each of the $45$ runs writes a per-run CSV into
`results/processed/project_baselines/` of the form
`midway_<algorithm>_<environment>_seed<seed>_eval.csv`. The aggregation step
`scripts/summarize_project_baselines.py --prefix midway` produces
`midway_vector_summary.csv` and the per-environment plots under
`figures/midway/`. The report-facing notebook
`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` loads these artefacts,
verifies that all $45$ algorithm-environment-seed combinations are present,
and renders the figures and summary tables shown below. The midway matrix
is therefore complete in the operational sense: the pipeline runs end to
end, the artefacts are present on disk, and they have been validated by the
notebook.

**5.6 Midway baseline results and discussion.**
Figure~\ref{fig:midway-vector-panel} shows the three baseline learning
curves side by side. On `cartpole_swingup`
(Figure~\ref{fig:midway-cartpole}), SAC and TD3 reach noticeably higher
mean evaluation returns than PPO by the end of the midway training budget;
PPO shows larger seed-to-seed variation and lower end-of-budget returns. On
`acrobot_swingup` (Figure~\ref{fig:midway-acrobot}), all three algorithms
remain at low mean returns within the $20\,000$-step budget, indicating
that none of the baselines solves the task at this budget. On
`car_racing_continuous` (Figure~\ref{fig:midway-carracing}), mean returns
remain negative for all three algorithms at the $10\,000$-step budget; SAC
attains less-negative mean returns than PPO and TD3 on most seeds, with
substantial cross-seed variability. These observations should be read as a
snapshot of the present pipeline at the midway training budget and under
the explicit condition that *no hyperparameter optimization was performed*.
They do not constitute a final ranking of the algorithms. The next project
stage will increase the training budget, introduce the GRPO-control
variant, and perform the final head-to-head comparison.

---

## 8. Suggested figure usage

All four files exist under `figures/midway/`. Suggested usage in the
midway report:

### `midway_vector_env_baselines.png` — **include**
- **Where.** Open Subsection 5.6.
- **Caption message.** Combined per-environment view of the midway baselines
  for PPO, SAC, and TD3 across five seeds. Mean evaluation return on the
  y-axis; training steps before evaluation on the x-axis. Shaded regions
  indicate cross-seed standard deviation.
- **Avoid in caption.** Do not assert "best algorithm" or "convergence".
  Do not omit the "midway training budget" qualifier.

### `midway_cartpole_swingup_baselines.png` — **include**
- **Where.** Inside Subsection 5.6, after the panel figure if used, or as
  a standalone subfigure.
- **Caption message.** Per-environment learning curves on
  `cartpole_swingup` for PPO, SAC, and TD3 across five seeds. Use this
  figure to support the qualitative observation that SAC and TD3 reach
  higher end-of-budget mean returns than PPO at the midway budget.
- **Avoid in caption.** Do not say SAC or TD3 is "the best on
  cartpole-swingup in general".

### `midway_acrobot_swingup_baselines.png` — **include**
- **Where.** Inside Subsection 5.6, alongside the cartpole figure.
- **Caption message.** Per-environment learning curves on
  `acrobot_swingup` for PPO, SAC, and TD3 across five seeds. Use this
  figure to support the observation that no baseline solves the task at
  the midway training budget.
- **Avoid in caption.** Do not say the algorithms "fail" on the task. Do
  say that they have not solved it within the chosen midway budget.

### `midway_car_racing_continuous_baselines.png` — **include**
- **Where.** Inside Subsection 5.6, after the vector environments.
- **Caption message.** Per-environment learning curves on
  `car_racing_continuous` for PPO, SAC, and TD3 across five seeds, run
  through the project-side CNN pipeline on Google Colab with CUDA. Use the
  figure to support the observation that returns remain negative at the
  $10\,000$-step midway budget.
- **Avoid in caption.** Do not say CarRacing is "deferred" or
  "unimplemented". Do not say PPO-CNN is missing — it is present in
  `scripts/carracing_cnn.py`.

If the page budget is tight, drop one of the per-environment figures and
keep only the combined panel figure plus the CarRacing single-env figure
(since CarRacing is the only image-observation env and is on a different
training budget).

---

## 9. Suggested citation placeholders

Citations to insert in the Experiments section (keys already present in
`report/references.bib`):

| Where in Experiments | Cite | Existing key |
|---|---|---|
| First mention of PPO (Subsection 5.1 or 5.3) | PPO paper | `\citep{schulman2017proximal}` |
| Alongside PPO when mentioning advantage estimation | GAE paper | `\citep{schulman2015gae}` |
| First mention of SAC | SAC paper | `\citep{haarnoja2018sacapps}` |
| First mention of TD3 | TD3 paper | `\citep{fujimoto2018td3}` |
| First mention of ObjectRL (Subsection 5.1 or 5.3) | ObjectRL paper | `\citep{baykal2025objectrl}` |
| First mention of CarRacing-v3 / Gymnasium (Subsection 5.2) | Gymnasium paper | `\citep{towers2024gymnasium}` |
| Optional: when describing `cartpole_swingup` / `acrobot_swingup` as DM Control–style tasks | DM Control paper | `\citep{tunyasuvunakool2020dmcontrol}` |

Notes:

- All keys above already exist in `report/references.bib`. Do not invent
  new keys.
- **Do not** cite GRPO (`shao2024deepseekmath`) in the Experiments section
  of the midway report; GRPO results are not yet reported. The first
  GRPO citation belongs in the Introduction or Related Work.
- DDPG, TRPO, DPG, Kakade-Langford, RLHF, and DeepSeek-R1 do not belong
  in the Experiments section.

---

## 10. Final author notes

**To decide manually**

- Whether to keep Subsection 5.6 as one block or split it into "Results"
  and "Discussion" subsections. The midway scope makes a single block
  defensible; the final report will probably split.
- Whether to include the combined panel figure
  (`midway_vector_env_baselines.png`) in addition to the three
  per-environment figures, or instead of them. Choose based on page budget.
- Whether to include a small "matrix complete" run-status table to make the
  3 × 3 × 5 = 45 claim visually obvious. The notebook may already render
  one — if so, prefer to mirror its content rather than re-derive it.
- Tone: the Danish draft is slightly more discursive than the English draft.
  Keep them in sync with the version you submit.

**To check against the assignment PDF (`DM887_Project.pdf`)**

- Required environment names. The PDF should be the authority on whether to
  write `CarRacing-v3`, `car_racing_continuous`, "continuous Car Racing",
  or "CarRacing continuous". Match the PDF's spelling throughout.
- Required evaluation metric. The drafts use "undiscounted evaluation
  episode return"; confirm this matches the assignment's wording.
- Required minimum number of seeds. The drafts cite five seeds; confirm
  this is what the assignment requires.
- Whether the assignment asks for a specific set of plots or tables (e.g.,
  curves vs. final-step bar charts).
- Whether the assignment requires a stated training budget. The midway
  budgets here are short by design and should be presented as such.

**To align later with Methodology and Theory**

- Notation in Subsection 5.4 (training step, evaluation episode, return)
  should match Methodology's MDP notation (e.g., $t$ vs. step counters,
  $G_t$ vs. "episode return").
- The pipeline-validation phrasing in Subsection 5.5 ("matrix is complete",
  "pipeline validated") should not contradict the Theory section's
  forward-looking scope statement.
- Ensure the algorithm short-names used here are the same ones used in
  Related Work and Methodology.

**What should be postponed until the final report**

- The GRPO-control results subsection.
- A formal head-to-head comparison between GRPO and the baselines.
- Hyperparameter-sensitivity analysis.
- Ablations on CNN architecture choices for CarRacing.
- A longer training-budget rerun of all 45 baselines if needed for the
  final algorithmic claims.
- Any statistical significance test on the seed-level returns; with five
  seeds and no HPO this is premature in the midway report.

**Reminders**

- This file is **inspiration only**. Do not paste the Danish or English
  drafts into `report/sections/05_experiments.tex` verbatim; rewrite in
  your own voice.
- Re-check every concrete number against the notebook
  (`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`) and the relevant CSV
  before printing it. If a rerun changes a number, update the section.
- Keep figure captions self-contained and conservative. The captions
  carry as much weight as the prose for graders skimming the section.
