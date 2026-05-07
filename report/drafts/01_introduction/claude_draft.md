# Introduction — Inspiration Draft (claude)

This document is **inspiration material** for the Introduction section of the DM887
midway report. It is not final report text. The final wording lives in
`report/sections/01_introduction.tex` and must be written manually.

Project: *Group Relative Policy Optimization for Control* (DM887, Spring 2026).
Stage: midway / interim report.

---

## 1. Introduction goal analysis

The Introduction in this **midway** report has to do four jobs at once and avoid
several traps. Specifically:

1. **Frame the project.** State that the project investigates whether a GRPO-style
   policy optimization scheme can be adapted from its LLM-reasoning origin to
   continuous-control benchmarks, and that PPO, SAC, and TD3 are the chosen
   baselines.
2. **Motivate the gap.** GRPO emerged in the LLM/reasoning setting, where
   group-relative advantages replace a learned value baseline. Whether this
   group-relative structure transfers to control, where credit assignment is
   sequential and rewards are dense or shaped, is not obvious.
3. **Be honest about the midway scope.** The report does not yet present the
   final GRPO-control variant. It establishes the experimental infrastructure,
   environment setup, and a complete PPO/SAC/TD3 baseline matrix that the final
   stage will build on.
4. **State the interim contribution clearly.** The contribution at this stage is
   a reproducible baseline pipeline: 3 algorithms × 3 environments × 5 seeds
   = 45 runs (900 evaluation rows), with vector-control environments handled
   through ObjectRL and CarRacing handled through a project-side CNN
   implementation executed on Google Colab with CUDA.

Traps to avoid in the Introduction:

- Do not treat this as a textbook chapter on RL.
- Do not promise theoretical results that are not in the midway report.
- Do not claim that the baseline runs already demonstrate algorithmic
  superiority — no hyperparameter optimization was performed.
- Do not say CarRacing is deferred. It is included in the baseline matrix.
- Do not say the report-facing notebook is dry-run only. The midway baseline
  matrix is completed and the notebook validates those results.
- Do not present GRPO as already implemented or evaluated.

Length target: roughly **0.5–1 page** in the NeurIPS template. The Introduction
should be tight, not exhaustive.

---

## 2. Paragraph-by-paragraph outline (5 paragraphs)

### Paragraph 1 — Context and problem
- **Purpose.** Open with the broader RL-control setting and the role of stable
  policy optimization.
- **Key message.** Continuous-control benchmarks remain a useful proving ground
  for new policy-optimization schemes because they expose stability,
  exploration, and credit-assignment issues that synthetic tasks hide.
- **Connects to.** Generic RL background; sets up the need for several
  algorithm families.

### Paragraph 2 — Baseline algorithm landscape (PPO, SAC, TD3)
- **Purpose.** Introduce the three baseline families used in the report.
- **Key message.** PPO is the canonical on-policy clipped-surrogate baseline;
  SAC is the off-policy maximum-entropy actor-critic; TD3 is the deterministic
  twin-critic actor-critic. They are picked because they cover different design
  axes (on/off-policy, stochastic/deterministic, entropy/Q-learning style).
- **Connects to.** PPO, SAC, TD3 references; sets up later related-work
  section; foreshadows the experimental matrix.

### Paragraph 3 — Why GRPO, why control
- **Purpose.** Motivate the project-specific direction.
- **Key message.** GRPO removes the learned value baseline by computing
  advantages relative to a sampled group, which has worked well in LLM/reasoning
  settings. It is not obvious that this carries over to continuous-control
  problems with sequential rewards and continuous action spaces. This open
  question is the project's eventual focus.
- **Connects to.** GRPO/DeepSeekMath reference; sets up the gap that the final
  project will address.

### Paragraph 4 — What the midway report establishes
- **Purpose.** State the interim contribution honestly.
- **Key message.** At the midway stage the project delivers:
  (i) a reproducible experimental pipeline,
  (ii) environment setup for `cartpole_swingup`, `acrobot_swingup`, and
  `car_racing_continuous`, with vector-control envs run through ObjectRL and
  CarRacing run through a project-side PyTorch CNN implementation executed on
  Google Colab with CUDA,
  (iii) a complete baseline matrix of 3 algorithms × 3 environments × 5 seeds
  = 45 runs (900 evaluation rows),
  (iv) a report-facing notebook
  (`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`) that loads, validates, and
  visualises this matrix.
  No hyperparameter optimization was performed; results validate the pipeline
  and are not intended as definitive performance comparisons.
- **Connects to.** ObjectRL reference; Gymnasium reference; methodology and
  experiments sections.

### Paragraph 5 — Scope limits and report structure
- **Purpose.** Mark what is *not* in the midway report and outline the rest of
  the document.
- **Key message.** The final GRPO-control variant, its convergence-style
  analysis, and the head-to-head comparison against the baselines belong to the
  final project stage. The remainder of the report covers related work,
  methodology and notation, a forward-looking theory scope, the baseline
  experiments, and a brief conclusion with next steps.
- **Connects to.** Sections 2–6 of the report.

(Optional 6th paragraph: an explicit numbered contribution list, if you want a
"Contributions:" bullet block. NeurIPS-style introductions often include this;
keep it to three or four bullets if used.)

---

## 3. Safe claims (supported by current repo state)

These claims are safe to make in the Introduction:

- The project investigates reinforcement learning for continuous-control style
  benchmark environments.
- The selected baselines are PPO, SAC, and TD3.
- The selected environments are `cartpole_swingup`, `acrobot_swingup`, and
  `car_racing_continuous` (continuous CarRacing).
- Vector-control environments are run through ObjectRL.
- CarRacing uses a project-side CNN implementation because ObjectRL does not
  directly support the image-observation setup used here.
- CarRacing CNN runs were executed on Google Colab with CUDA and copied back
  into the local repo.
- Five seeds (0..4) were used per algorithm/environment combination.
- The completed midway matrix consists of 3 algorithms × 3 environments
  × 5 seeds = 45 runs.
- The summarised result table contains 900 evaluation rows.
- The midway report-facing notebook is
  `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`.
- No hyperparameter optimization was performed at the midway stage.
- The midway results are preliminary and primarily validate the pipeline.
- The project follows the NeurIPS-style report layout required by the
  assignment.
- The report is structured as Introduction, Related Work, Methodology, Theory,
  Experiments, Conclusion, with proofs in an appendix.
- The final GRPO-related extension is planned for the next stage.

---

## 4. Claims to avoid (would overclaim the current midway state)

Do not write any of the following:

- "The proposed GRPO-control method outperforms baselines."
- "The baseline results show state-of-the-art performance on these tasks."
- "PPO/SAC/TD3 are ranked by the midway results" (no HPO was performed).
- "CarRacing is deferred" / "CarRacing was not included" (it is included).
- "PPO-CNN is not implemented" / "We could not implement PPO with image
  observations" (PPO-CNN is implemented in `scripts/carracing_cnn.py`).
- "The notebook is a dry-run only" (it loads and validates the completed
  baseline matrix).
- "GRPO experiments are reported" / "GRPO is benchmarked here" (GRPO is for
  the next stage).
- "Hyperparameters were tuned" / "We searched hyperparameters" (none of this
  occurred).
- "We prove convergence of the proposed method" (theory is forward-looking
  at the midway stage).
- Any wording that implies the midway baselines settle which algorithm is
  best in general; they do not.

Use cautious, scoped wording such as:
*"At the midway stage..."*,
*"These results validate the experimental pipeline..."*,
*"The baseline results are preliminary..."*,
*"No hyperparameter optimization was performed..."*,
*"The final project stage will build on this baseline matrix..."*

---

## 5. Danish draft

> **Bemærk.** Akademisk sprog, men læsbart. Brug dette som udgangspunkt og
> redigér selv. Citationsmarkeringer er sat med `\citep{...}`.

**Introduktion.**

Dyb forstærkningslæring (RL) har gennem de seneste år leveret stærke baselines
for kontinuerlig kontrol. Disse opgaver er fortsat et nyttigt testmiljø for nye
policy-optimeringsmetoder, fordi de eksponerer stabilitets-, eksplorations- og
credit-assignment-udfordringer, som mere syntetiske benchmarks ofte skjuler.

Tre metoder definerer et bredt udsnit af nuværende baselines.
Proximal Policy Optimization \citep{schulman2017proximal} bruger et clippet
surrogate-objektiv kombineret med generaliseret advantage-estimering
\citep{schulman2015gae} og repræsenterer den on-policy stokastiske familie.
Soft Actor-Critic \citep{haarnoja2018sacapps} er off-policy med
maksimal-entropi-regulering. Twin Delayed Deep Deterministic Policy Gradient
\citep{fujimoto2018td3} er en deterministisk dobbelt-kritiker-metode. De tre
algoritmer dækker således forskellige designvalg langs aksen on-/off-policy,
stokastisk/deterministisk og entropi-/Q-baseret.

Group Relative Policy Optimization \citep{shao2024deepseekmath} er senere
blevet udbredt i sprogmodel- og reasoning-konteksten, hvor advantages
beregnes relativt inden for en sampleret gruppe i stedet for at lære en
separat værdifunktion. Det er dog ikke åbenlyst, hvordan denne
gruppe-relative struktur bedst overføres til kontinuerlig kontrol, hvor
belønninger er sekventielle og handlingsrum er kontinuerte. Dette spørgsmål
er projektets endelige fokus.

På midtvejsstadiet er bidraget primært infrastrukturelt. Rapporten etablerer
en reproducerbar baseline-pipeline for PPO, SAC og TD3 på tre miljøer:
`cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous`. De
vektorbaserede miljøer afvikles gennem ObjectRL \citep{baykal2025objectrl},
mens CarRacing afvikles gennem en projekt-egen PyTorch CNN-implementering, da
ObjectRL ikke direkte understøtter den anvendte billed-observation. CarRacing
CNN-kørslerne blev udført på Google Colab med CUDA og kopieret tilbage til
det lokale repository. Den fuldførte midtvejsmatrix består af 3 algoritmer
× 3 miljøer × 5 seeds = 45 kørsler, og det aggregerede datasæt indeholder
900 evaluerings-rækker. Der er ikke udført hyperparameteroptimering, og
resultaterne valideres i den rapport-rettede notebook
`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`. Resultaterne er derfor
foreløbige og bør læses som en bekræftelse af pipelinen, ikke som en endelig
sammenligning af algoritmernes ydeevne.

Den foreslåede GRPO-variant for kontrol, dens teoretiske analyse og den
endelige sammenligning med baselines hører til den næste fase af projektet.
Resten af denne midtvejsrapport præsenterer relateret arbejde
(afsnit~\ref{sec:related}), formel MDP-notation og metode
(afsnit~\ref{sec:methodology}), en forward-looking teori-scope
(afsnit~\ref{sec:theory}), de nuværende baseline-eksperimenter
(afsnit~\ref{sec:experiments}) og en kort konklusion med næste skridt
(afsnit~\ref{sec:conclusion}).

---

## 6. English academic draft

> **Note.** Concise, report-facing. No LaTeX commands beyond citation
> placeholders. Edit freely before pasting into LaTeX.

**Introduction.**

Deep reinforcement learning has produced a small set of strong baselines for
continuous control. These tasks remain a useful proving ground for new
policy-optimization schemes, because they expose stability, exploration, and
credit-assignment issues that more synthetic benchmarks tend to hide.

Three algorithms cover a broad slice of the current baseline landscape.
Proximal Policy Optimization \citep{schulman2017proximal} pairs a clipped
surrogate objective with Generalized Advantage Estimation
\citep{schulman2015gae} and represents the on-policy stochastic family. Soft
Actor-Critic \citep{haarnoja2018sacapps} is an off-policy actor-critic with
entropy regularization. Twin Delayed Deep Deterministic Policy Gradient
\citep{fujimoto2018td3} is a deterministic twin-critic method. Together, the
three algorithms span the on-/off-policy, stochastic/deterministic, and
entropy-/Q-learning design axes that any new policy-optimization method for
control should be measured against.

Group Relative Policy Optimization \citep{shao2024deepseekmath} has more
recently become prominent in language-model and reasoning settings, where
advantages are computed relative to a sampled group rather than against a
learned value baseline. Whether this group-relative structure transfers
cleanly to continuous control — where rewards are sequential, action spaces
are continuous, and trajectories are long — is not obvious. This question
motivates the project as a whole.

At the midway stage the contribution is primarily infrastructural. This
report establishes a reproducible baseline pipeline for PPO, SAC, and TD3 on
three environments: `cartpole_swingup`, `acrobot_swingup`, and
`car_racing_continuous` \citep{towers2024gymnasium}. The vector-control
environments are run through the ObjectRL framework
\citep{baykal2025objectrl}; CarRacing is run through a project-side PyTorch
CNN implementation, because ObjectRL does not directly support the
image-observation setup used here, and the CarRacing CNN runs were executed
on Google Colab with CUDA and then copied back into the local repository.
The completed midway baseline matrix consists of three algorithms, three
environments, and five seeds (3 × 3 × 5 = 45 runs), aggregating to 900
evaluation rows. No hyperparameter optimization was performed. The
report-facing notebook
`notebooks/DM887_Project_GRPO_Midway_PoC.ipynb` validates this matrix. The
midway results should therefore be read as a validation of the experimental
pipeline rather than as a definitive comparison of algorithmic performance.

The proposed GRPO-control variant, its theoretical scope, and the final
head-to-head comparison against the PPO, SAC, and TD3 baselines are deferred
to the final project stage. The remainder of this report covers related work
(Section 2), MDP notation and methodology (Section 3), a forward-looking
theory scope (Section 4), the current baseline experiments (Section 5), and
a short conclusion with next steps (Section 6).

---

## 7. Suggested citation placeholders

Citations to insert in the Introduction (keys already present in
`report/references.bib`):

| Where in the Introduction | Cite | Existing key |
|---|---|---|
| First mention of PPO (Paragraph 2) | PPO paper | `\citep{schulman2017proximal}` |
| Alongside PPO, mentioning advantage estimation | GAE paper | `\citep{schulman2015gae}` |
| First mention of SAC (Paragraph 2) | SAC paper | `\citep{haarnoja2018sacapps}` |
| First mention of TD3 (Paragraph 2) | TD3 paper | `\citep{fujimoto2018td3}` |
| First mention of GRPO (Paragraph 3) | GRPO / DeepSeekMath paper | `\citep{shao2024deepseekmath}` |
| Optional: broader LLM reasoning context for GRPO (only if you discuss it) | DeepSeek-R1 paper | `\citep{deepseekai2025deepseekr1}` |
| Mention of the environments (Paragraph 4) | Gymnasium paper | `\citep{towers2024gymnasium}` |
| Mention that vector-control baselines run through ObjectRL (Paragraph 4) | ObjectRL paper | `\citep{baykal2025objectrl}` |

Notes:

- All keys above are already in `report/references.bib`. Do not invent new
  keys.
- TRPO (`schulman2015trpo`), DDPG (`lillicrap2015continuous`), DPG
  (`silver2014deterministic`), and `kakade2002approximately` are also in the
  bib file but are better placed in **Related Work / Methodology**, not in
  the Introduction.
- `tunyasuvunakool2020dmcontrol` is in the bib file. Only cite it in the
  Introduction if the cartpole-swingup / acrobot-swingup tasks here are
  presented as DM Control–style tasks (they are commonly described as such).
  Decide manually based on how Methodology will frame these environments.

---

## 8. Final author notes

**To decide manually**

- Whether to add an explicit "Contributions:" bullet block at the end of the
  Introduction, NeurIPS-style. The current draft does not include one; you
  may prefer the cleaner prose form for an interim report.
- Whether to cite Gymnasium and DM Control jointly when introducing
  `cartpole_swingup` and `acrobot_swingup`, or to defer environment
  citations entirely to the Experiments section. Both are defensible.
- Whether to mention CarRacing's image observations in the Introduction at
  all, or leave that detail to Methodology. The current draft mentions it
  briefly to justify the Colab/CUDA workflow.
- Tone: the Danish draft is slightly more discursive than the English draft.
  Pick one as the lead and keep the other in sync.
- Citation command style: the existing bib + NeurIPS template loads
  `natbib` with `plainnat`. `\citep` and `\citet` both work. Stay
  consistent with whatever you choose for the rest of the report.

**To check against the assignment PDF (`DM887_Project.pdf`)**

- Required environment names. The PDF should be the authority on whether to
  write `CarRacing-v3`, `car_racing_continuous`, "continuous CarRacing", or
  similar. Match the PDF's spelling.
- Required number of seeds and the evaluation metric (undiscounted
  evaluation episode return). Confirm the Introduction's wording matches.
- Whether the assignment requires an explicit "Use of AI Tools" section
  (separate from the Introduction).
- Whether the midway/interim report has its own length or scope rules in
  the PDF that would change Introduction sizing.

**To align later with Methodology and Experiments**

- The exact terminology used for the algorithms (e.g., "PPO with GAE" vs
  just "PPO") should match between the Introduction, Methodology, and
  Experiments sections.
- The result-matrix numbers (3 × 3 × 5 = 45 runs, 900 evaluation rows) must
  be repeated consistently in Experiments. If the row count changes after
  any rerun, update the Introduction too.
- The split between ObjectRL (vector-control) and the project-side CNN
  (CarRacing) is mentioned in the Introduction; Methodology should describe
  the wrappers and Experiments should describe the Colab+CUDA execution.
  Avoid duplicating the same text in three places — let each section give
  the level of detail appropriate to its role.
- The forward-looking statement about the GRPO-control variant in the
  Introduction must agree with what the Theory and Conclusion sections
  promise for the final stage.

**Reminders**

- This file is **inspiration only**. Do not paste the Danish or English
  drafts into `report/sections/01_introduction.tex` verbatim; rewrite in
  your own voice.
- Keep the Introduction at roughly half a page to one page in the NeurIPS
  template.
- Re-check every claim against the current state of the repository and the
  notebook before submission. If any item drifts (e.g., a rerun changes
  seed counts), update the Introduction accordingly.
