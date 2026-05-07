# Introduction Draft Material for the DM887 Midway Report

This document is inspiration material for `report/sections/01_introduction.tex`.
It is not intended as final report text. It is based on the assignment PDF, the
README, the final report-facing midway notebook, the LaTeX scaffold, the current
bibliography, and the project writing/reference plans.

Important framing assumption: the notebook
`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` is treated as the main source of
truth for the current midway status.

## 1. Introduction Goal Analysis

The Introduction must establish the project problem without making the reader
wait for the motivation. The project is about adapting Group Relative Policy
Optimization (GRPO), originally motivated in language-model reinforcement
learning, to control settings. The introduction should therefore connect two
ideas: stable policy optimization in control tasks, and the open question of
whether GRPO-style group-relative advantage estimates can be useful outside the
LLM setting.

For the midway report, the Introduction must also manage expectations. The
report does not yet present the final GRPO-control algorithm, convergence
analysis, or final comparison against GRPO. Instead, the current contribution is
the experimental foundation: a reproducible PPO/SAC/TD3 baseline matrix across
the required environments, plus the implementation decisions needed to make that
matrix run end to end. This is not just administrative work. The final GRPO
comparison will only be meaningful if the baseline pipeline, environment
wrappers, evaluation metrics, seeds, and result artefacts are already validated.

The Introduction should make clear that PPO, SAC, and TD3 are not the proposed
method. They are the required reference points and represent different families
of modern deep RL baselines: on-policy stochastic policy optimization, off-policy
maximum-entropy actor-critic learning, and deterministic actor-critic learning
for continuous control. The project uses these methods to define a comparison
surface for the later GRPO extension.

The Introduction should also explain the practical scope of the midway result:
3 algorithms x 3 environments x 5 seeds = 45 runs, producing 900 evaluation
rows. The vector-control environments use ObjectRL; CarRacing uses project-side
CNN implementations because the image-observation setup does not fit ObjectRL's
default vector-observation path. These results validate the experimental
pipeline, not the final scientific hypothesis.

Finally, the Introduction should point forward. The final stage should build on
the completed baseline matrix by implementing a GRPO-control variant, deciding
how group-relative estimates should interact with temporal credit assignment and
continuous actions, and evaluating the resulting method against the same
baseline protocol.

## 2. Paragraph-by-Paragraph Outline

### Paragraph 1: Control Motivation and Project Problem

Purpose: introduce the reinforcement-learning control problem and why stable
policy optimization matters.

Key message: the project studies policy optimization for continuous-control
style benchmark environments where agents must learn from online interaction and
where evaluation should be reproducible across seeds.

Evidence/context to connect to: assignment PDF requirements for online training,
regular evaluation intervals, undiscounted evaluation return, five seeds, and
the three environments.

### Paragraph 2: Baseline Landscape

Purpose: position PPO, SAC, and TD3 as the baseline methods.

Key message: PPO, SAC, and TD3 are included because they represent established
and complementary approaches to deep RL for control. They are comparison points,
not the proposed contribution.

Evidence/context to connect to: assignment PDF requiring PPO, SAC, and TD3;
`report/references.bib` keys for PPO, GAE, SAC, and TD3; project plan saying the
baselines do not need to be implemented from scratch.

### Paragraph 3: Why GRPO Is Interesting and Nontrivial in Control

Purpose: explain why GRPO is relevant and why transfer from LLM-style RL to
control is not automatic.

Key message: GRPO's group-relative policy update is attractive because it
suggests an alternative to conventional value-estimation-heavy advantage
signals, but control tasks add temporal credit assignment, continuous action
geometry, and environment-specific dynamics.

Evidence/context to connect to: assignment PDF algorithm-design prompt; reading
list and bibliography entries for DeepSeekMath/GRPO; project plan noting that a
control-suitable GRPO variant remains future work.

### Paragraph 4: Midway Contribution

Purpose: state exactly what the midway report contributes.

Key message: at the midway stage, the project establishes the related-work
foundation, notation, and a completed baseline result matrix. The results
validate the experimental pipeline and artefact layout needed for the final GRPO
comparison.

Evidence/context to connect to: final notebook status: 45 runs, 900 rows, all
three algorithms and environments, seeds 0..4; ObjectRL vector-control path;
project-side CNN CarRacing path; no hyperparameter optimization.

### Paragraph 5: Limitations and Next Stage

Purpose: prevent overclaiming and create a natural transition to later sections.

Key message: the baseline results are preliminary because the budgets are short
and no hyperparameter optimization was performed. The final stage will implement
and evaluate the GRPO-specific extension using the same validated comparison
surface.

Evidence/context to connect to: notebook interpretation rules, report-ready
notes, assignment PDF expectation that final experiments include GRPO, PPO, SAC,
and TD3 learning curves.

### Optional Paragraph 6: Report Structure

Purpose: briefly tell the reader how the report is organized, if space allows.

Key message: the report first reviews related work, then defines the formal MDP
notation and methodology, then presents the midway baseline protocol/results,
and finally summarizes limitations and next steps.

Evidence/context to connect to: `report/DM887_Report.tex` section order:
Introduction, Related Work, Methodology, Theory, Experiments, Conclusion.

## 3. Safe Claims

- The project investigates reinforcement learning for control benchmark
  environments and aims to build toward a GRPO-control variant.
- At the midway stage, the final GRPO-specific method and final GRPO experiments
  are not yet completed.
- PPO, SAC, and TD3 are used as baselines.
- The completed midway baseline matrix is 3 algorithms x 3 environments x 5
  seeds = 45 runs.
- The midway result files contain 900 evaluation rows.
- The algorithms in the midway matrix are PPO, SAC, and TD3.
- The environments in the midway matrix are `cartpole_swingup`,
  `acrobot_swingup`, and `car_racing_continuous`.
- The seed set is 0, 1, 2, 3, 4.
- Evaluation is based on undiscounted episode return.
- The vector-control baselines use ObjectRL through the project wrapper/runner
  workflow.
- CarRacing uses project-side PyTorch CNN implementations because the
  image-observation setup is not directly supported by ObjectRL's default
  vector-observation baseline path.
- PPO-CNN, SAC-CNN, and TD3-CNN are implemented for CarRacing in the project
  code path.
- CarRacing CNN runs were executed on Google Colab with CUDA and copied back
  into the local result folders.
- No hyperparameter optimization was performed for the midway results.
- The midway results validate the experimental pipeline, result layout, and
  evaluation protocol.
- Apparent performance differences in the midway baseline matrix should be
  treated as preliminary.
- The final stage will build on the completed baseline matrix to implement and
  evaluate the GRPO/project-specific extension.

## 4. Claims to Avoid

- Avoid claiming that the final GRPO-control algorithm is already implemented
  and evaluated.
- Avoid claiming that the midway report demonstrates GRPO superiority.
- Avoid claiming state-of-the-art performance.
- Avoid claiming that any PPO/SAC/TD3 ordering proves general algorithm
  superiority.
- Avoid saying that CarRacing is deferred or missing.
- Avoid saying that PPO-CNN is not implemented.
- Avoid saying that the final notebook is dry-run only.
- Avoid implying that ObjectRL directly handled the CarRacing image-observation
  setup used in the project.
- Avoid saying that the baseline algorithms were hyperparameter optimized.
- Avoid saying that the results show convergence or asymptotic performance.
- Avoid saying that all environments were solved in a final-performance sense.
- Avoid presenting wall-clock times as directly comparable across vector-control
  and CarRacing runs, since CarRacing used Colab CUDA while vector runs were
  local.
- Avoid using assignment-goal language such as "works better than PPO and on par
  with SAC and TD3" as a completed result. That belongs to the final target, not
  the midway status.
- Avoid using old repository status language that says CarRacing outputs still
  need to be imported, unless the report is explicitly discussing historical
  workflow.

## 5. Danish Draft

Reinforcement learning i kontrolopgaver handler om at lære beslutningspolitikker
gennem interaktion med et dynamisk miljø. I sådanne opgaver er det ikke nok, at
en algoritme kan forbedre en politik i et enkelt eksempel; evalueringen skal
også være stabil, reproducerbar og sammenlignelig på tværs af miljøer og seeds.
Dette projekt undersøger derfor policy optimization for kontinuerte
kontrol-lignende benchmarkmiljøer med henblik på senere at udvikle en variant af
Group Relative Policy Optimization (GRPO), der kan anvendes uden for den
LLM-kontekst, hvor metoden især er blevet kendt.

Som referencepunkt anvendes PPO, SAC og TD3. De tre metoder repræsenterer
forskellige designvalg i moderne deep reinforcement learning: PPO er en
on-policy metode med en stabiliserende clipped policy objective, SAC er en
off-policy maximum-entropy actor-critic metode, og TD3 er en deterministisk
actor-critic metode udviklet til kontinuerte handlinger. I dette projekt bruges
de ikke som et nyt metodisk bidrag, men som nødvendige baselines for den senere
vurdering af en GRPO-baseret kontrolmetode.

GRPO er interessant, fordi metoden bruger relative signaler inden for grupper af
samples i stedet for udelukkende at basere policy-opdateringen på en traditionel
value model. Det gør metoden relevant at undersøge i en bredere RL-sammenhæng.
Overførslen til kontrolopgaver er dog ikke triviel. Kontrolmiljøer kræver
sekventiel credit assignment, håndtering af kontinuerte handlingsrum og robust
læring fra ofte støjende og seed-afhængige trajektorier. En GRPO-variant til
kontrol må derfor designes og evalueres med disse forhold for øje.

På midway-stadiet er projektets bidrag ikke den endelige GRPO-metode. Bidraget
er i stedet at etablere den eksperimentelle infrastruktur, som den endelige
sammenligning skal bygge på. Den fulde baseline-matrix er gennemført for PPO,
SAC og TD3 på `cartpole_swingup`, `acrobot_swingup` og
`car_racing_continuous` med fem seeds pr. kombination, svarende til 45 kørsler
og 900 evalueringsrækker. De vektorbaserede miljøer køres gennem ObjectRL, mens
CarRacing anvender projektets egne CNN-baserede implementationer, fordi den
anvendte billedobservation ikke passer direkte til ObjectRL's standardsti for
vektorobservationer.

Resultaterne bør tolkes forsigtigt. Der er ikke udført hyperparameteroptimering,
og træningsbudgetterne er valgt til et midway-setup snarere end til endelig
konvergens. Baseline-resultaterne viser derfor først og fremmest, at pipeline,
miljøadaptere, evalueringsprotokol og resultatformater fungerer end-to-end. Den
næste projektfase skal bygge videre på denne matrix ved at implementere den
GRPO-specifikke udvidelse, fastholde en sammenlignelig evalueringsprotokol og
undersøge, om den foreslåede metode kan vurderes meningsfuldt mod PPO, SAC og
TD3.

## 6. English Academic Draft

Reinforcement learning for control requires policy optimization methods that can
learn from online interaction while remaining reproducible across environments,
training budgets, and random seeds. This project studies such methods in
continuous-control style benchmark environments and uses them as the foundation
for a later Group Relative Policy Optimization (GRPO) extension. GRPO has
recently attracted attention in language-model reinforcement learning, but its
role in control tasks remains less established.

The project uses PPO, SAC, and TD3 as baseline methods. These algorithms provide
complementary reference points: PPO represents on-policy stochastic policy
optimization, SAC represents off-policy maximum-entropy actor-critic learning,
and TD3 represents deterministic actor-critic learning for continuous action
spaces \citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}. Their
role in this report is not to constitute the proposed method, but to define the
comparison surface on which the final GRPO-control variant will be evaluated.

The motivation for considering GRPO is that group-relative policy updates may
offer an alternative way to construct advantage-like learning signals
\citep{shao2024deepseekmath}. However, transferring this idea to control is not
automatic. Control tasks involve temporal credit assignment, continuous action
geometry, and environment-specific dynamics, so a useful GRPO-control variant
must be evaluated against strong and reproducible baselines rather than argued
for only at the level of algorithmic intuition.

At the midway stage, this report therefore focuses on the experimental and
conceptual foundation needed before the GRPO-specific extension is introduced.
The completed baseline matrix covers PPO, SAC, and TD3 on `cartpole_swingup`,
`acrobot_swingup`, and `car_racing_continuous` with five seeds per combination,
for a total of 45 runs and 900 evaluation rows. The vector-control experiments
use ObjectRL, while CarRacing uses project-side CNN implementations because the
image-observation setup used here is not directly supported by ObjectRL's
default vector-observation path \citep{baykal2025objectrl}.

These baseline results should be interpreted as preliminary. No hyperparameter
optimization was performed, and the midway budgets are intended to validate the
pipeline rather than establish asymptotic performance. The main contribution of
the midway work is therefore a reproducible baseline protocol, validated result
artefacts, and a clear implementation path for the final project stage. The
final report will build on this baseline matrix by introducing the
GRPO-control extension, analyzing its assumptions, and comparing it against the
same PPO, SAC, and TD3 protocol.

## 7. Suggested Citation Placeholders

Use only keys that already exist in `report/references.bib`.

- When introducing PPO as the on-policy baseline:
  `\citep{schulman2017proximal}`.
- When mentioning PPO with GAE or advantage estimation:
  `\citep{schulman2015gae,schulman2017proximal}`.
- When introducing SAC as the maximum-entropy actor-critic baseline:
  `\citep{haarnoja2018sacapps}`.
- When introducing TD3 as the deterministic actor-critic baseline:
  `\citep{fujimoto2018td3}`.
- When mentioning deterministic policy-gradient background, if needed:
  `\citep{silver2014deterministic,lillicrap2015continuous}`.
- When introducing GRPO from DeepSeekMath:
  `\citep{shao2024deepseekmath}`.
- When discussing the broader reasoning/RL context around DeepSeek-R1, only if
  useful:
  `\citep{deepseekai2025deepseekr1}`.
- When discussing RLHF/LLM policy optimization background, only if useful:
  `\citep{ouyang2022training}`.
- When stating that the vector-control baselines use ObjectRL:
  `\citep{baykal2025objectrl}`.
- When describing CarRacing as a Gymnasium/Farama environment:
  `\citep{towers2024gymnasium}`.
- When describing the DeepMind Control Suite swingup environments:
  `\citep{tunyasuvunakool2020dmcontrol}`.

Suggested citation locations in the Introduction:

- Paragraph 2, after the sentence naming PPO, SAC, and TD3:
  `\citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}`.
- Paragraph 3, after the sentence introducing GRPO:
  `\citep{shao2024deepseekmath}`.
- Paragraph 4, after the sentence explaining ObjectRL usage:
  `\citep{baykal2025objectrl}`.
- If the environment list is included in the Introduction, cite Gymnasium and
  DMC in the same sentence:
  `\citep{towers2024gymnasium,tunyasuvunakool2020dmcontrol}`.

## 8. Final Author Notes

- Decide manually whether the Introduction should include the exact numbers
  "45 runs" and "900 evaluation rows" or leave those details for the
  Experiments section. Including them can strengthen the midway contribution,
  but the Introduction should not become a results section.
- Check the assignment PDF wording before finalizing the first paragraph. The
  assignment asks for motivation of the studied problem and the main takeaways
  of the proposed solution; at midway, the "proposed solution" should be framed
  as planned future GRPO-control work, not completed evidence.
- Align environment names consistently. The report can use readable assignment
  names such as CarRacing-v3, cartpole-swingup-v0, and acrobot-swingup-v0, while
  methodology/experiments can map them to project names like
  `car_racing_continuous`, `cartpole_swingup`, and `acrobot_swingup`.
- Check that the Introduction does not contradict the final notebook. In
  particular, do not use older wording that says CarRacing is deferred or that
  the notebook is only a dry run.
- Align later with Methodology by defining exactly where ObjectRL is used and
  where the project-side CNN code is used.
- Align later with Experiments by repeating the cautious interpretation rules:
  at the midway budget, results validate the pipeline; they do not prove
  algorithm superiority.
- Decide later whether to mention specific step budgets in the Introduction.
  They are useful for transparency but may fit better in Methodology or
  Experiments.
- Before moving text into LaTeX, verify that every citation key used in the
  final Introduction exists in `report/references.bib` and compiles.
