# Experiments Draft Material for the DM887 Midway Report

This document is inspiration material for `report/sections/05_experiments.tex`.
It is not final report text. It is based on the assignment PDF, README, the
final report-facing notebook, the current LaTeX scaffold, reference plans,
scientific-writing notes, relevant runner/summarizer scripts, existing result
CSVs, and the existing midway figures.

Important framing assumption: `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`
is the main source of truth for midway result validation, tables, rankings,
progress summaries, stability summaries, and displayed figures.

## 1. Experiments Section Goal Analysis

The Experiments section must document the completed PPO/SAC/TD3 midway baseline
matrix in a way that is useful for the final GRPO project, but does not pretend
that the final GRPO method is already implemented. The section should show that
the experimental pipeline is complete enough to support the final comparison:
the required environments are available, the required baseline algorithms have
been run, the seeds and evaluation metric are fixed, result files are validated,
and learning curves can be generated from the saved artefacts.

The main message should be reproducibility and readiness, not final performance.
At the midway stage, the completed matrix is already substantial:
3 algorithms x 3 environments x 5 seeds = 45 runs, with 900 evaluation rows.
This should be stated clearly because the assignment reserves interim credit for
complete PPO/SAC/TD3 results. However, these results should be described as
baseline and preliminary. No hyperparameter optimization was performed, and the
training budgets are short relative to what would normally be used for final
convergence claims.

The section should also explain the two implementation paths. The vector-control
environments (`cartpole_swingup`, `acrobot_swingup`) use ObjectRL via the
project-side runner and environment adapter. `car_racing_continuous` uses
project-side CNN agents because the project uses image observations and ObjectRL
does not directly support that default setup through its vector-observation
baseline path. This is an important reproducibility point, not a weakness to
hide.

The interpretation should be cautious. It is safe to say that SAC has the
highest final-block mean return on CartPole-Swingup and CarRacing at the midway
budget, while PPO has the highest final-block mean return on Acrobot-Swingup in
the notebook ranking table. It is not safe to say that this proves general
algorithm superiority, that all algorithms converged, or that the observed
ordering will remain under longer budgets or tuned hyperparameters.

## 2. Proposed Subsection Structure

### 5.1 Experimental Setup

Purpose: define the scope of the midway experiments before giving results.

Key message: the midway experiments cover PPO, SAC, and TD3 across the three
required environments and five shared seeds.

Connect to: notebook Sections 3, 5, 7, and 11; assignment PDF evaluation
requirements; README current status.

Midway or final: belongs in the midway report. In the final report, extend this
subsection to include GRPO-control.

### 5.2 Environments

Purpose: describe the three benchmark tasks and their observation/setup
differences.

Key message: CartPole-Swingup and Acrobot-Swingup are vector-control tasks
handled through the ObjectRL path; CarRacing-v3 continuous uses RGB image
observations and therefore uses project-side CNN baselines.

Connect to: notebook Sections 3 and 6; `scripts/run_project_objectrl_baseline.py`;
`scripts/run_carracing_cnn_baseline.py`; `scripts/carracing_cnn.py`; citations
for Gymnasium and DMC.

Midway or final: belongs in the midway report. Final report can add any
environment changes, longer budgets, or preprocessing changes.

### 5.3 Baseline Algorithms and Implementations

Purpose: explain which algorithm implementations were used and why there are two
code paths.

Key message: PPO/SAC/TD3 for vector-control are run through ObjectRL without
modifying `external/objectrl`; PPO-CNN/SAC-CNN/TD3-CNN for CarRacing are
project-side PyTorch implementations.

Connect to: notebook Section 6; `scripts/run_project_objectrl_baseline.py`;
`scripts/run_carracing_cnn_baseline.py`; `scripts/carracing_cnn.py`;
`report/references.bib` for PPO/SAC/TD3/ObjectRL.

Midway or final: belongs in the midway report. GRPO implementation details
should wait for the final report.

### 5.4 Evaluation Protocol and Artefacts

Purpose: state the seeds, training budgets, evaluation cadence, return metric,
and output format.

Key message: the canonical comparison surface is a tidy table with algorithm,
environment, seed, training step, evaluation episode, and undiscounted
evaluation return.

Connect to: notebook Sections 3, 5, 7, and 12; `midway_*_eval.csv` files;
`midway_vector_summary.csv`; assignment PDF metric requirement.

Midway or final: belongs in the midway report. In the final report, keep this
protocol stable or explicitly justify any changes.

### 5.5 Result Validation

Purpose: show that the complete baseline matrix is present and loaded from disk.

Key message: the notebook validates 45 matched CSV files, 900 rows, all expected
algorithm/environment/seed combinations, and no missing combinations.

Connect to: notebook Section 7 output; status table from notebook cell 10;
`results/processed/project_baselines/midway_vector_summary.csv`.

Midway or final: belongs strongly in the midway report because it is direct
evidence for interim completion.

### 5.6 Midway Baseline Learning Curves

Purpose: present the observed learning curves and final-block summaries.

Key message: the figures show mean evaluation return across five seeds with
variation bands. At the midway budget, the curves validate learning/evaluation
behaviour and pipeline coverage; they should not be read as final performance.

Connect to: notebook Sections 8, 9, 10, and 11; figure files in
`figures/midway/`.

Midway or final: belongs in the midway report. The final report should replace
or extend it with GRPO-control curves.

### 5.7 Interpretation and Limitations

Purpose: give cautious interpretation and explain what remains.

Key message: the observed differences are preliminary, no hyperparameter
optimization was performed, the budgets are short, and CarRacing is especially
limited by the small CNN-RL training budget.

Connect to: notebook Sections 9, 11, 12, and 14; scientific-writing notes on
separating results from discussion.

Midway or final: belongs in the midway report. Final report should add GRPO
comparison, statistical analysis, longer budgets, and theory-linked discussion.

## 3. Safe Claims

- The midway baseline matrix is complete.
- The matrix contains PPO, SAC, and TD3 across `cartpole_swingup`,
  `acrobot_swingup`, and `car_racing_continuous`.
- The matrix uses five seeds: 0, 1, 2, 3, and 4.
- The matrix contains 45 runs and 900 evaluation rows.
- The notebook found 45 midway CSV files and all 45 expected
  algorithm/environment/seed combinations.
- The notebook reports 0 missing combinations.
- All status values in the loaded midway rows are `completed`.
- Vector-control environments use ObjectRL through the project-side bridge.
- CarRacing uses project-side CNN implementations because the image-observation
  setup does not fit ObjectRL's default vector-observation actor/critic path.
- PPO-CNN, SAC-CNN, and TD3-CNN are implemented and completed for CarRacing.
- The CarRacing matrix contains 15 runs, from 3 algorithms x 5 seeds.
- CarRacing CNN runs were executed on Google Colab with CUDA and copied back
  into the local repository.
- Evaluation returns are undiscounted episode returns.
- Each evaluation block uses three evaluation episodes in the midway setup.
- Vector-control runs use a 20,000-step midway budget and evaluation every 5,000
  steps, with final logged step 19,999 in the notebook output.
- CarRacing CNN runs use a 10,000-step midway budget and evaluation every 1,000
  steps.
- No hyperparameter optimization was performed.
- The observed rankings are conditional on the midway budget and default
  hyperparameters.
- The completed baseline matrix establishes the comparison basis for the final
  GRPO stage.

## 4. Claims to Avoid

- Do not claim that GRPO-control experiments are complete.
- Do not include GRPO in the midway learning curves unless explicitly labelled
  as future work.
- Do not claim that SAC, PPO, or TD3 is generally superior based on the midway
  results.
- Do not claim state-of-the-art performance.
- Do not claim that all algorithms converged.
- Do not claim that the tasks are solved in a final-performance sense.
- Do not say CarRacing is deferred, missing, or only planned.
- Do not say PPO-CNN is not implemented.
- Do not say the report-facing notebook is dry-run only.
- Do not imply that ObjectRL directly handled the CarRacing image observations.
- Do not cite debug runs as the main experiment.
- Do not compare wall-clock times across vector-control and CarRacing as if the
  hardware were identical.
- Do not present hyperparameter choices as tuned or optimized.
- Do not hide the short budgets or the lack of tuning.
- Do not use old script/docstring wording that says PPO/TD3 CNN are not
  implemented; the notebook and later code sections show the current midway
  status.

## 5. Result Facts to Extract

### Matrix and Validation Facts

- Final notebook status line: PPO, SAC, TD3 x CartPole-Swingup,
  Acrobot-Swingup, CarRacing-v3 continuous x 5 seeds = 45 runs / 900 evaluation
  rows.
- Notebook cell 9 output: `midway CSV files found : 45 (expected 45)`.
- Notebook cell 9 output: `total rows : 900 (expected 900)`.
- Notebook cell 9 output: algorithms present are `ppo`, `sac`, `td3`.
- Notebook cell 9 output: environments present are `acrobot_swingup`,
  `car_racing_continuous`, `cartpole_swingup`.
- Notebook cell 9 output: seeds present are 0, 1, 2, 3, 4.
- Notebook cell 9 output: present combinations are 45 / 45.
- Notebook cell 9 output: missing combinations are 0.
- Notebook cell 10 output: every `(environment, algorithm)` pair has 5 seeds.
- Notebook cell 10 output: all 900 loaded rows have status `completed`.
- `results/processed/project_baselines/midway_vector_summary.csv` has 301 lines
  including header, i.e. 300 aggregated rows.

### Evaluation Budget and Cadence Facts

- Notebook Section 3: vector-control environments use 20,000 environment steps
  per run, evaluation every 5,000 steps with 3 episodes.
- Notebook Section 3: CarRacing CNN environments use 10,000 environment steps
  per run, evaluation every 1,000 steps with 3 episodes.
- Notebook cell 10 output: vector envs have `min_train_step = 0` and
  `max_train_step = 19999`.
- Notebook cell 10 output: CarRacing has `min_train_step = 1000` and
  `max_train_step = 10000`.
- Notebook Section 12: the evaluation environment is constructed with
  `seed + 10_000`.
- Notebook Section 12: each evaluation row records undiscounted episode return.
- Notebook Section 12: canonical comparison columns include `algorithm`,
  `project_env`, `seed`, `train_step`, `eval_episode`, and `eval_return`.

### Row Counts by Environment and Algorithm

From notebook cell 10:

- `acrobot_swingup`: PPO 75 rows, SAC 75 rows, TD3 75 rows.
- `cartpole_swingup`: PPO 75 rows, SAC 75 rows, TD3 75 rows.
- `car_racing_continuous`: PPO 150 rows, SAC 150 rows, TD3 150 rows.

Reason: vector runs have 5 evaluation steps x 3 episodes x 5 seeds = 75 rows
per algorithm/environment pair; CarRacing runs have 10 evaluation steps x 3
episodes x 5 seeds = 150 rows per algorithm.

### Final-Block Summary Facts

Notebook cell 12 reports the following final-block mean return summaries across
five seeds. Use these cautiously and round only if needed in the final report.

| Environment | Algorithm | Mean final return | Std | Min | Max | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| `acrobot_swingup` | PPO | 16.477074 | 13.208912 | 0.861994 | 31.991789 | 5 |
| `acrobot_swingup` | TD3 | 10.656823 | 9.526693 | 0.512053 | 22.896653 | 5 |
| `acrobot_swingup` | SAC | 2.932347 | 5.681132 | 0.092263 | 13.079141 | 5 |
| `car_racing_continuous` | SAC | -25.139777 | 15.651311 | -37.106382 | 1.898805 | 5 |
| `car_racing_continuous` | TD3 | -80.033772 | 13.654906 | -93.635174 | -57.042108 | 5 |
| `car_racing_continuous` | PPO | -91.747155 | 2.903162 | -93.635174 | -86.618877 | 5 |
| `cartpole_swingup` | SAC | 283.183651 | 73.752449 | 192.437419 | 369.872152 | 5 |
| `cartpole_swingup` | TD3 | 205.822012 | 70.615329 | 152.022924 | 329.330373 | 5 |
| `cartpole_swingup` | PPO | 86.701853 | 62.612355 | 24.767230 | 182.785105 | 5 |

### Ranking Facts

Notebook cell 15 ranks algorithms by mean final return within each environment:

- `acrobot_swingup`: PPO rank 1, TD3 rank 2, SAC rank 3.
- `car_racing_continuous`: SAC rank 1, TD3 rank 2, PPO rank 3.
- `cartpole_swingup`: SAC rank 1, TD3 rank 2, PPO rank 3.

Safe wording: "At the midway budget, the final-block ranking table places..."

Unsafe wording: "This proves that X is the best algorithm."

### Progress Facts

Notebook cell 17 reports first-to-final evaluation deltas:

- `cartpole_swingup`: SAC mean delta 270.801018, TD3 mean delta 185.505804,
  PPO mean delta 61.693625.
- `acrobot_swingup`: PPO mean delta 8.952758, TD3 mean delta 4.979103, SAC mean
  delta -4.447530.
- `car_racing_continuous`: TD3 mean delta 4.045785, PPO mean delta 1.353133,
  SAC mean delta -11.458620.

Safe wording: "The progress table suggests clear positive learning signal for
SAC and TD3 on CartPole-Swingup at the midway budget; the other environments are
more mixed."

### Stability Facts

Notebook cell 19 reports within-run and across-seed stability summaries:

- Across-seed final-return standard deviations are the same as the
  `std_final_return` values in the final-block summary table.
- Lowest across-seed std by environment:
  - `acrobot_swingup`: SAC, std 5.681132.
  - `car_racing_continuous`: PPO, std 2.903162.
  - `cartpole_swingup`: PPO, std 62.612355.
- Lowest mean within-run evaluation std by environment:
  - `acrobot_swingup`: SAC, 3.968623.
  - `car_racing_continuous`: PPO, 0.317776.
  - `cartpole_swingup`: PPO, 3.645664.

Interpretation caution: lower variance does not imply higher return or a better
final policy. For example, PPO is stable but low-return on CarRacing at the
midway budget.

### Figure Files

The notebook displays these static PNGs and does not regenerate them:

- `figures/midway/midway_cartpole_swingup_baselines.png` (900 x 600).
- `figures/midway/midway_acrobot_swingup_baselines.png` (900 x 600).
- `figures/midway/midway_car_racing_continuous_baselines.png` (900 x 600).
- `figures/midway/midway_vector_env_baselines.png` (2452 x 617).

Each per-environment figure shows mean across seeds with a +/- 1 standard
deviation band per algorithm, according to notebook Section 10 and the
summarizer script.

## 6. Danish Draft

Dette afsnit beskriver de baseline-eksperimenter, der er gennemført på
midway-stadiet af projektet. Formålet er ikke at evaluere den endelige
GRPO-baserede metode, da denne først implementeres i næste projektfase.
Formålet er i stedet at etablere en reproducerbar sammenligningsbasis for de
tre krævede baseline-algoritmer PPO, SAC og TD3 på de tre krævede miljøer.

Eksperimenterne omfatter `cartpole_swingup`, `acrobot_swingup` og
`car_racing_continuous`. De to første miljøer behandles som vektorbaserede
kontrolopgaver og køres gennem ObjectRL via projektets egne miljøadaptere.
CarRacing adskiller sig ved at bruge RGB-billedobservationer fra
CarRacing-v3 med kontinuerte handlinger. Da ObjectRL's standardsti forventer
1-dimensionelle vektorobservationer, anvendes der for CarRacing en
projektside-implementation med CNN-baserede PPO-, SAC- og TD3-agenter. Disse
CarRacing-kørsler er udført på Google Colab med CUDA, hvorefter resultaterne er
kopieret tilbage til det lokale repository.

Den fulde baseline-matrix er gennemført for 3 algoritmer, 3 miljøer og 5 seeds,
svarende til 45 kørsler. Seeds er 0, 1, 2, 3 og 4 for alle kombinationer. Den
rapportvendte notebook validerer 45 CSV-filer, 900 evalueringsrækker og ingen
manglende algoritme/miljø/seed-kombinationer. Hver evalueringsrække indeholder
blandt andet algoritme, miljø, seed, træningstrin, evaluerings-episode og
udiskonteret evaluerings-return. Vektormiljøerne er kørt med 20.000
miljøtrin pr. kørsel og evaluering hver 5.000 trin, mens CarRacing-CNN-kørslerne
er kørt med 10.000 miljøtrin og evaluering hver 1.000 trin. Hver evaluering
består af tre episoder.

Figurerne viser læringskurver for PPO, SAC og TD3, hvor kurverne er gennemsnit
over fem seeds og skyggeområder angiver variation på tværs af seeds. På
CartPole-Swingup ses det tydeligste læringssignal ved midway-budgettet, særligt
for SAC og TD3. På Acrobot-Swingup er resultaterne mere ustabile og bør primært
ses som et tegn på, at pipeline og evaluering fungerer for en vanskeligere
swingup-opgave. På CarRacing er budgettet meget lille for CNN-baseret
reinforcement learning, og figuren bør derfor især bruges til at dokumentere, at
PPO-CNN, SAC-CNN og TD3-CNN kan køres og evalueres end-to-end under den samme
overordnede protokol.

De aggregerede final-block-resultater giver en foreløbig rangering ved
midway-budgettet: PPO ligger højest på Acrobot-Swingup, mens SAC ligger højest
på CartPole-Swingup og CarRacing. Denne rangering må ikke tolkes som en generel
konklusion om algoritmernes relative styrke. Der er ikke udført
hyperparameteroptimering, træningsbudgetterne er korte, og ingen af kurverne
bør læses som bevis for endelig konvergens. Resultaterne viser først og
fremmest, at baseline-matricen, evalueringsprotokollen og artefaktstrukturen er
på plads. Dermed etablerer de det sammenligningsgrundlag, som den endelige
GRPO-udvidelse skal evalueres imod.

## 7. English Academic Draft

This section reports the baseline experiments completed at the midway stage of
the project. The purpose is not to evaluate the final GRPO-control method, which
will be introduced in the next project stage. Instead, the purpose is to
establish a reproducible baseline matrix for PPO, SAC, and TD3 on the required
control environments.

The experiments cover `cartpole_swingup`, `acrobot_swingup`, and
`car_racing_continuous`. The two swingup environments are vector-control tasks
and are run through ObjectRL using the project-side environment adapter. The
CarRacing task uses continuous actions and RGB image observations from
CarRacing-v3. Since ObjectRL's default baseline path expects one-dimensional
vector observations, the CarRacing experiments use project-side PyTorch CNN
implementations of PPO, SAC, and TD3. The CarRacing CNN runs were executed on
Google Colab with CUDA and the resulting evaluation CSVs were copied back into
the local repository.

The completed midway matrix consists of 3 algorithms, 3 environments, and 5
seeds, for a total of 45 runs. The seeds are 0, 1, 2, 3, and 4 for every
algorithm/environment combination. The report-facing notebook validates 45
midway CSV files, 900 evaluation rows, all 45 expected combinations, and no
missing combinations. Each evaluation row records the algorithm, environment,
seed, training step, evaluation episode, and undiscounted evaluation episode
return. The vector-control runs use 20,000 environment steps with evaluation
every 5,000 steps and 3 evaluation episodes. The CarRacing CNN runs use 10,000
environment steps with evaluation every 1,000 steps and 3 evaluation episodes.
No hyperparameter optimization was performed.

Figure~\ref{fig:midway-baselines-placeholder} should show the midway learning
curves for PPO, SAC, and TD3. Each curve reports the mean undiscounted
evaluation return across five seeds as a function of the number of training
steps before evaluation, with shaded bands indicating variation across seeds.
At the midway budget, CartPole-Swingup shows the clearest learning signal,
especially for SAC and TD3. Acrobot-Swingup is more variable, and CarRacing is
the most constrained setup because 10,000 environment steps is a very small
budget for CNN-based reinforcement learning from RGB observations.

The final-block summary table in the notebook gives a preliminary ranking at
the midway budget: PPO has the highest mean final return on Acrobot-Swingup,
while SAC has the highest mean final return on CartPole-Swingup and CarRacing.
These differences should not be interpreted as final algorithm rankings or
general performance claims. They are conditioned on short training budgets,
default hyperparameters, and the current implementation choices. The main
conclusion from the midway experiments is therefore that the baseline pipeline
is complete, the result artefacts are validated, and the project now has a
consistent comparison basis for the final GRPO-control stage.

## 8. Suggested Figure Usage

### `figures/midway/midway_cartpole_swingup_baselines.png`

Include: yes, either as one subfigure in a three-panel baseline figure or as an
individual figure if space allows.

Caption message: five-seed PPO/SAC/TD3 learning curves for CartPole-Swingup;
mean undiscounted evaluation return with variation across seeds; midway budget
only.

Interpretation to support: CartPole-Swingup shows the clearest learning signal
among the three tasks at the midway budget, especially for SAC and TD3.

Avoid: claiming convergence, claiming final superiority, or implying that the
same ranking will hold after tuning or longer runs.

### `figures/midway/midway_acrobot_swingup_baselines.png`

Include: yes, ideally as one subfigure in the same three-panel baseline figure.

Caption message: five-seed PPO/SAC/TD3 learning curves for Acrobot-Swingup under
the same vector-control protocol as CartPole-Swingup.

Interpretation to support: Acrobot-Swingup is more variable at the midway
budget; the curves document that the baseline pipeline runs and produces
evaluation returns.

Avoid: claiming that the task is solved, that all algorithms converged, or that
PPO's final-block rank proves general superiority on Acrobot.

### `figures/midway/midway_car_racing_continuous_baselines.png`

Include: yes. It is important because CarRacing is required by the assignment
and because earlier/old wording might otherwise make it sound deferred.

Caption message: five-seed PPO-CNN/SAC-CNN/TD3-CNN learning curves for
continuous CarRacing-v3 using project-side CNN implementations; run on Colab
CUDA and validated locally.

Interpretation to support: the image-observation baseline path works
end-to-end, but the 10,000-step budget is too small for final CNN-RL claims.

Avoid: claiming that CarRacing performance is final, that ObjectRL handled this
image setup directly, or that SAC's midway rank establishes general superiority.

### `figures/midway/midway_vector_env_baselines.png`

Include: optional. The filename says "vector_env", but the current summarizer
creates a combined overview across all expected environments. Use it only if the
layout is acceptable and the caption clarifies what it contains.

Caption message: compact overview of the midway PPO/SAC/TD3 curves across the
three environments, intended as a visual summary.

Interpretation to support: useful overview of the validated baseline matrix.

Avoid: using it as the only figure if panel readability is poor in the NeurIPS
template; avoid calling it "vector-only" if it includes CarRacing.

Recommended final report choice for midway: use a single figure with three
subpanels if you can assemble it cleanly in LaTeX from the three
per-environment PNGs. If that is too much for the page budget, include the
combined overview and refer to the notebook for per-environment plots.

Suggested caption text to adapt:

> Midway PPO/SAC/TD3 baseline learning curves. The x-axis shows the number of
> training steps before evaluation and the y-axis shows undiscounted evaluation
> episode return. Curves show the mean across five seeds, with shaded bands
> indicating variation across seeds. The vector-control tasks use ObjectRL,
> while CarRacing uses project-side CNN baselines. The curves validate the
> baseline pipeline at the midway budget and should not be interpreted as final
> convergence or tuned performance.

## 9. Suggested Citation Placeholders

Use only keys that already exist in `report/references.bib`.

- When introducing PPO as a baseline:
  `\citep{schulman2017proximal}`.
- When mentioning PPO with GAE:
  `\citep{schulman2015gae,schulman2017proximal}`.
- When introducing SAC:
  `\citep{haarnoja2018sacapps}`.
- When introducing TD3:
  `\citep{fujimoto2018td3}`.
- When explaining ObjectRL as the vector-control implementation framework:
  `\citep{baykal2025objectrl}`.
- When referring to Gymnasium/Farama CarRacing:
  `\citep{towers2024gymnasium}`.
- When referring to the DeepMind Control Suite swingup environments:
  `\citep{tunyasuvunakool2020dmcontrol}`.

Possible sentence patterns:

- "The vector-control baselines use ObjectRL implementations of PPO, SAC, and
  TD3 \citep{baykal2025objectrl}."
- "CarRacing-v3 is taken from Gymnasium/Farama, while the swingup environments
  are based on the DeepMind Control Suite
  \citep{towers2024gymnasium,tunyasuvunakool2020dmcontrol}."
- "The selected baselines represent on-policy policy optimization,
  maximum-entropy actor-critic learning, and deterministic actor-critic learning
  \citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}."

## 10. Final Author Notes

- Decide manually whether to include the exact final-block numerical table in
  the report or keep it in the notebook. In a short NeurIPS-style midway report,
  the figure plus coverage facts may be enough.
- Check the assignment PDF wording before finalizing the experiment prose. The
  final assignment asks for learning curves of GRPO, PPO, SAC, and TD3, but the
  interim credit is for complete PPO/SAC/TD3 results. The midway text should
  state this distinction explicitly.
- Align with Methodology by using the same MDP notation, same evaluation-return
  definition, same seed set, and same environment names.
- Align with Theory by avoiding theory claims in this section. Do not discuss
  GRPO convergence here except as planned final work.
- Decide whether the report should call the environments by assignment-facing
  names (`CarRacing-v3`, `cartpole-swingup-v0`, `acrobot-swingup-v0`) or
  project-facing names (`car_racing_continuous`, `cartpole_swingup`,
  `acrobot_swingup`). A short mapping sentence can prevent confusion.
- Check the LaTeX graphic path before adding figures. `DM887_Report.tex`
  includes `\graphicspath{{figures/}{figures/midway/}{figures/final/}}`, so the
  final include paths should match the report's compile location.
- Postpone GRPO learning curves, GRPO-vs-baseline statistical tests, longer
  training budgets, hyperparameter sweeps, and final claims about solving tasks
  until the final report.
- If space is tight, prioritize: validation facts, one clear figure, no tuning
  caveat, and the statement that the baseline matrix establishes the comparison
  basis for the final GRPO stage.
