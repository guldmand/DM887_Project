# Experiments draft inspiration for the DM887 midway report

This document is inspiration material for `report/sections/05_experiments.tex`. It is not final report text and should be manually refined before being copied into the LaTeX report.

The source of truth for concrete result facts is `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`, especially its validation, summary, ranking, progress, stability, and figure-display sections.

## 1. Experiments section goal analysis

The Experiments section in the midway report should document the completed PPO/SAC/TD3 baseline matrix and explain why it is sufficient as a foundation for the final GRPO stage. Its main purpose is not to prove algorithmic superiority, but to show that the project now has a reproducible experimental pipeline, validated result files, and baseline learning curves for all required environments.

The section should accomplish four things:

1. **Define the experimental matrix clearly.** State the algorithms, environments, seeds, training budgets, evaluation intervals, and metric. The reader should be able to understand what was run without opening the notebook.
2. **Explain the implementation split.** The vector-control environments use ObjectRL, while CarRacing uses project-side CNN baselines because the ObjectRL path used here assumes 1-D vector observations and does not directly support the `(96, 96, 3)` image observation setup.
3. **Validate completeness.** The midway notebook reports 45 expected CSV files, 900 expected rows, 45/45 present algorithm-environment-seed combinations, and 0 missing combinations. This is the strongest factual claim in the section.
4. **Interpret results cautiously.** The results can be discussed as preliminary learning-curve and end-of-budget signals under short midway budgets and default hyperparameters. They should not be described as convergence results, state-of-the-art results, or final rankings.

The core framing should be:

> These results validate the experimental pipeline and establish the comparison basis for the final GRPO-control experiments. They do not yet constitute a final performance study.

## 2. Proposed subsection structure

### 5.1 Experimental setup

- **Purpose:** Introduce the experiment matrix and midway scope.
- **Key message:** The midway experiments evaluate PPO, SAC, and TD3 on the three required environments with seeds 0--4.
- **Connect to:** Notebook Section 3 and Section 5; assignment PDF evaluation requirement; README experiment status.
- **Midway or final:** Belongs in the midway report. For the final report, this subsection will be extended with the GRPO-control method.

### 5.2 Environments and implementation paths

- **Purpose:** Explain the three environments and why two implementation paths are used.
- **Key message:** `cartpole_swingup` and `acrobot_swingup` are vector-observation control tasks run through ObjectRL; `car_racing_continuous` is an image-observation task run through project-side CNN implementations.
- **Connect to:** Notebook Section 6.1--6.3; `scripts/run_project_objectrl_baseline.py`; `scripts/run_carracing_cnn_baseline.py`; `scripts/carracing_cnn.py`.
- **Midway or final:** Belongs in the midway report. Final report may add any GRPO-specific environment wrappers or modifications.

### 5.3 Algorithms and evaluation protocol

- **Purpose:** State what is compared and how evaluation returns are recorded.
- **Key message:** PPO, SAC, and TD3 are evaluated with five seeds, undiscounted evaluation episode return, and regular evaluation intervals. Vector runs use 20,000 steps with evaluation every 5,000 steps; CarRacing CNN runs use 10,000 steps with evaluation every 1,000 steps.
- **Connect to:** Notebook Section 3; script `RUN_CONFIG` values; result CSV columns; summary CSV.
- **Midway or final:** Belongs in the midway report. Final report should add GRPO and any revised final budgets.

### 5.4 Result validation and artefacts

- **Purpose:** Demonstrate that the result matrix is complete and reproducible.
- **Key message:** The report-facing notebook validates 45 CSV files, 900 total rows, all expected algorithms/environments/seeds, 45/45 present combinations, and 0 missing combinations.
- **Connect to:** Notebook Section 7 validation output; per-environment status table; `results/processed/project_baselines/midway_vector_summary.csv`; figure files under `figures/midway/`.
- **Midway or final:** Essential for the midway report. For the final report, repeat validation after GRPO results are added.

### 5.5 Midway baseline results

- **Purpose:** Present the observed learning curves and compact end-of-budget result summaries.
- **Key message:** The curves show that the pipeline produces evaluation trajectories for all algorithms and environments. At the midway budget, SAC and TD3 show clear progress on CartPole-Swingup; Acrobot-Swingup remains harder and noisy; CarRacing validates the CNN pipeline but remains a very short-budget image-control experiment.
- **Connect to:** Notebook Sections 8--10; final-block summary table; ranking table; progress table; `midway_cartpole_swingup_baselines.png`, `midway_acrobot_swingup_baselines.png`, `midway_car_racing_continuous_baselines.png`.
- **Midway or final:** Belongs in the midway report, but should remain concise and cautious.

### 5.6 Interpretation and limitations

- **Purpose:** Prevent overclaiming and explain how the results should be used.
- **Key message:** The results are preliminary because no hyperparameter optimization was performed, training budgets are short, and the final GRPO method is not yet included.
- **Connect to:** Notebook Section 9 interpretation caveat and Section 12 limitations; scientific-writing notes about separating results from discussion.
- **Midway or final:** Belongs in the midway report. Final report should replace this with a fuller comparison including GRPO, longer budgets, and theory-aligned interpretation.

### 5.7 Planned final extension

- **Purpose:** Briefly connect the midway baselines to the final project.
- **Key message:** The completed matrix establishes the baseline comparison surface for the final GRPO-control experiments.
- **Connect to:** Assignment PDF final requirements; report Theory section; final-stage notebook/script plans.
- **Midway or final:** Include only a short forward-looking paragraph in the midway report. Detailed GRPO comparisons should wait for the final report.

## 3. Safe claims

The following claims are safe to make in the Experiments section:

- The midway report evaluates PPO, SAC, and TD3 as baselines.
- The evaluated environments are `cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous`.
- The full midway baseline matrix is complete: 3 algorithms x 3 environments x 5 seeds = 45 runs.
- The report-facing notebook validates 45 CSV files and 900 result rows.
- All expected combinations are present: 45/45 present and 0 missing combinations.
- Seeds are `0, 1, 2, 3, 4` for every algorithm-environment combination.
- Vector-control runs use ObjectRL through the project runner.
- CarRacing uses project-side CNN implementations: PPO-CNN, SAC-CNN, and TD3-CNN.
- CarRacing CNN runs were executed on Google Colab with CUDA and copied back into the local repository for validation and analysis.
- The evaluation metric is undiscounted evaluation episode return.
- Vector-control runs use 20,000 environment steps per run, evaluation every 5,000 steps, and 3 evaluation episodes per evaluation block.
- CarRacing CNN runs use 10,000 environment steps per run, evaluation every 1,000 steps, and 3 evaluation episodes per evaluation block.
- No hyperparameter optimization was performed.
- The learning curves are averaged across seeds, with variation shown as a standard-deviation band in the generated figures.
- The baseline results are preliminary and conditioned on the midway budgets and default hyperparameters.
- The completed matrix validates the experimental pipeline and establishes the comparison basis for the final GRPO stage.

## 4. Claims to avoid

Avoid the following claims in the midway Experiments section:

- Do not claim that the GRPO-control method has already been implemented or evaluated.
- Do not claim that the experiments are the final project experiments.
- Do not claim state-of-the-art performance.
- Do not claim that PPO, SAC, or TD3 superiority is proven in general.
- Do not claim that all algorithms converged.
- Do not claim that the end-of-budget ranking is a statistically robust final ranking.
- Do not imply that hyperparameters were optimized.
- Do not claim that CarRacing was deferred.
- Do not claim that PPO-CNN is missing; PPO-CNN, SAC-CNN, and TD3-CNN are implemented in the project-side CarRacing path.
- Do not describe the report-facing notebook as a dry-run notebook.
- Do not present debug figures or debug runs as the main experiment.
- Do not invent missing result numbers.
- Do not compare wall-clock performance across vector environments and CarRacing as if hardware were controlled.
- Do not imply that ObjectRL directly supports the exact CarRacing image-observation setup used here.
- Do not overinterpret a low standard deviation as better learning; for example, a method can be stable because it consistently performs poorly.

## 5. Result facts to extract

Concrete facts visible in the final notebook and result files:

### Matrix and validation

- Report-facing notebook status: complete midway baseline matrix.
- Total matrix: PPO, SAC, TD3 x CartPole-Swingup, Acrobot-Swingup, CarRacing-v3 continuous x 5 seeds.
- Total runs: 45.
- Total result rows: 900.
- CSV files found: 45 expected, 45 found.
- Algorithms present: `ppo`, `sac`, `td3`.
- Environments present: `acrobot_swingup`, `car_racing_continuous`, `cartpole_swingup`.
- Seeds present: `[0, 1, 2, 3, 4]`.
- Present combinations: 45 / 45.
- Missing combinations: 0.
- Global status value counts: `completed = 900`.

### Per-environment and per-algorithm coverage

- `acrobot_swingup`: PPO/SAC/TD3 each have 5 seeds, 75 evaluation rows, min train step 0, max train step 19,999, status completed.
- `cartpole_swingup`: PPO/SAC/TD3 each have 5 seeds, 75 evaluation rows, min train step 0, max train step 19,999, status completed.
- `car_racing_continuous`: PPO/SAC/TD3 each have 5 seeds, 150 evaluation rows, min train step 1,000, max train step 10,000, status completed.

### Evaluation protocol

- Vector-control environments: 20,000 environment steps per run, evaluation every 5,000 steps, 3 evaluation episodes.
- CarRacing CNN environments: 10,000 environment steps per run, evaluation every 1,000 steps, 3 evaluation episodes.
- No hyperparameter optimisation.
- Vector environments use ObjectRL defaults.
- CarRacing CNN trainers use project-side defaults based on the SAC/TD3/PPO literature.
- The aggregated summary CSV uses columns `algorithm`, `project_env`, `seed`, `train_step`, `status`, `eval_return`, and `n_repeats`.
- In the summary CSV, `n_repeats = 3`, corresponding to three evaluation episodes per evaluation block.

### End-of-budget final-return summary

These values are from the notebook final-block summary. Use cautiously and phrase as "at the midway budget".

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

### Ranking facts visible in the notebook

The notebook ranks algorithms by mean final return within each environment, with lower standard deviation used only as a tie-breaker:

- `acrobot_swingup`: PPO rank 1, TD3 rank 2, SAC rank 3.
- `car_racing_continuous`: SAC rank 1, TD3 rank 2, PPO rank 3.
- `cartpole_swingup`: SAC rank 1, TD3 rank 2, PPO rank 3.

These are safe only as **midway-budget descriptive rankings**, not final algorithm rankings.

### Learning-progress facts visible in the notebook

Mean delta from first to final evaluation block:

| Environment | Algorithm | Mean first | Mean final | Mean delta | Std delta | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| `acrobot_swingup` | PPO | 7.524316 | 16.477074 | 8.952758 | 21.318654 | 5 |
| `acrobot_swingup` | TD3 | 5.677720 | 10.656823 | 4.979103 | 14.351922 | 5 |
| `acrobot_swingup` | SAC | 7.379877 | 2.932347 | -4.447530 | 15.135600 | 5 |
| `car_racing_continuous` | TD3 | -84.079557 | -80.033772 | 4.045785 | 10.083123 | 5 |
| `car_racing_continuous` | PPO | -93.100288 | -91.747155 | 1.353133 | 3.213798 | 5 |
| `car_racing_continuous` | SAC | -13.681157 | -25.139777 | -11.458620 | 19.113688 | 5 |
| `cartpole_swingup` | SAC | 12.382633 | 283.183651 | 270.801018 | 72.725484 | 5 |
| `cartpole_swingup` | TD3 | 20.316208 | 205.822012 | 185.505804 | 83.345453 | 5 |
| `cartpole_swingup` | PPO | 25.008228 | 86.701853 | 61.693625 | 62.518543 | 5 |

Suggested cautious wording:

> CartPole-Swingup shows the clearest learning progress within the midway budget, especially for SAC and TD3. Acrobot-Swingup and CarRacing show weaker or noisier progress, consistent with the notebook's warning that the midway budgets are short.

### Stability facts visible in the notebook

The notebook reports within-run final-block evaluation standard deviation and across-seed final-return standard deviation:

| Environment | Algorithm | Mean within-run eval std | Across-seed final-return std | Seeds |
|---|---:|---:|---:|---:|
| `acrobot_swingup` | SAC | 3.968623 | 5.681132 | 5 |
| `acrobot_swingup` | TD3 | 14.696409 | 9.526693 | 5 |
| `acrobot_swingup` | PPO | 20.407896 | 13.208912 | 5 |
| `car_racing_continuous` | PPO | 0.317776 | 2.903162 | 5 |
| `car_racing_continuous` | TD3 | 2.134390 | 13.654906 | 5 |
| `car_racing_continuous` | SAC | 6.867185 | 15.651311 | 5 |
| `cartpole_swingup` | PPO | 3.645664 | 62.612355 | 5 |
| `cartpole_swingup` | TD3 | 18.223746 | 70.615329 | 5 |
| `cartpole_swingup` | SAC | 21.501631 | 73.752449 | 5 |

Suggested cautious wording:

> Stability should be interpreted together with return level. For example, low variance can reflect consistently low performance rather than robust task solution.

### Figure files to reference

- `figures/midway/midway_cartpole_swingup_baselines.png`
- `figures/midway/midway_acrobot_swingup_baselines.png`
- `figures/midway/midway_car_racing_continuous_baselines.png`
- `figures/midway/midway_vector_env_baselines.png`

The notebook states that each per-environment figure shows the mean across seeds with a +/- 1 standard-deviation band per algorithm.

## 6. Danish draft

Dette afsnit dokumenterer de baseline-eksperimenter, der er gennemført ved midway-stadiet. Formålet er ikke at evaluere den endelige GRPO-baserede metode, men at etablere en reproducerbar sammenligningsbasis for den afsluttende projektfase. Baseline-matricen består af tre algoritmer, PPO, SAC og TD3, evalueret på tre miljøer: `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous`. Hver kombination er kørt med seeds `0, 1, 2, 3, 4`, hvilket giver 45 kørsler i alt.

De to vektorbaserede kontrolmiljøer køres gennem ObjectRL. CarRacing behandles separat med projektets egne CNN-baserede implementeringer af PPO, SAC og TD3, fordi CarRacing-v3 returnerer billedobservationer med form `(96, 96, 3)`, mens den anvendte ObjectRL-baseline er målrettet 1-dimensionelle vektorobservationer. CarRacing-kørslerne blev udført på Google Colab med CUDA og efterfølgende kopieret tilbage til det lokale repository, hvor de indgår i samme validerings- og aggregeringspipeline som de øvrige resultater.

Evalueringsmetrikken er undiscounted evaluation episode return. For `cartpole_swingup` og `acrobot_swingup` anvendes 20.000 miljøskridt per kørsel med evaluering hver 5.000 skridt og 3 evaluerings-episoder per evalueringspunkt. For `car_racing_continuous` anvendes 10.000 miljøskridt per kørsel med evaluering hver 1.000 skridt og 3 evaluerings-episoder per evalueringspunkt. Der er ikke udført hyperparameteroptimering; resultaterne bruger standardkonfigurationerne fra de respektive implementeringer.

Den rapportvendte notebook validerer, at hele midway-matricen er til stede. Den finder 45 CSV-filer, 900 evalueringsrækker, alle tre algoritmer, alle tre miljøer og alle fem seeds. Samtidig rapporterer den 45 ud af 45 forventede algoritme-miljø-seed-kombinationer og 0 manglende kombinationer. Dette er den centrale midway-konklusion for eksperimenterne: den eksperimentelle pipeline er komplet nok til at understøtte den senere GRPO-sammenligning.

Læringskurverne viser middelværdien på tværs af fem seeds med variation visualiseret som standardafvigelse. Ved midway-budgettet ses den tydeligste læringsfremgang på `cartpole_swingup`, hvor især SAC og TD3 forbedrer sig markant fra første til sidste evalueringspunkt. På `acrobot_swingup` er signalet mere støjfyldt, og ingen af algoritmerne bør betragtes som konvergeret. På `car_racing_continuous` er budgettet på 10.000 skridt meget kort for CNN-baseret reinforcement learning fra RGB-billeder; figuren bør derfor primært læses som en validering af, at PPO-CNN, SAC-CNN og TD3-CNN kan køres end-to-end.

Ved sidste evalueringsblok har notebooken følgende deskriptive rangeringer efter gennemsnitlig return: PPO, TD3, SAC på `acrobot_swingup`; SAC, TD3, PPO på `car_racing_continuous`; og SAC, TD3, PPO på `cartpole_swingup`. Disse forskelle bør ikke fortolkes som endelige algoritmerangeringer. De gælder kun under de korte midway-budgetter og uden hyperparameteroptimering. Den vigtigste eksperimentelle konklusion er derfor, at baseline-matricen er gennemført og valideret, ikke at én algoritme generelt er bedre end de andre.

I den afsluttende projektfase skal denne baseline udvides med den GRPO-relaterede kontrolmetode. De samme miljøer, seeds, evalueringsmetrikker og figurpipeline kan dermed bruges til at sammenligne GRPO-varianten med PPO, SAC og TD3 på en konsistent måde.

## 7. English academic draft

This section reports the baseline experiments completed at the midway stage. The purpose is not to evaluate the final GRPO-control method, which remains planned work, but to establish a reproducible comparison basis for the final project. The baseline matrix consists of PPO, SAC, and TD3 evaluated on `cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous`. Each algorithm-environment pair is run with seeds `0, 1, 2, 3, 4`, giving 45 baseline runs in total.

The vector-control environments are executed through the ObjectRL-based workflow. CarRacing is handled separately using project-side PyTorch CNN implementations of PPO, SAC, and TD3, because the ObjectRL baseline path used in this project assumes one-dimensional vector observations, whereas CarRacing-v3 returns RGB image observations. The CarRacing CNN runs were executed on Google Colab with CUDA and the resulting evaluation CSV files were copied back into the local repository for validation and aggregation.

The evaluation metric is undiscounted evaluation episode return. For `cartpole_swingup` and `acrobot_swingup`, each run uses 20,000 environment steps, evaluation every 5,000 steps, and 3 evaluation episodes per evaluation block. For `car_racing_continuous`, each run uses 10,000 environment steps, evaluation every 1,000 steps, and 3 evaluation episodes per evaluation block. No hyperparameter optimization was performed; the results use the default configurations of the corresponding implementations.

The report-facing notebook validates the completeness of the midway matrix. It finds 45 CSV files, 900 total evaluation rows, all three algorithms, all three environments, and all five seeds. It also reports 45/45 expected algorithm-environment-seed combinations and 0 missing combinations. This validation is the main experimental contribution of the midway report: the project now has a complete baseline artefact layout and result pipeline for the later GRPO comparison.

The learning curves in Figure~\ref{fig:midway-cartpole-placeholder}, Figure~\ref{fig:midway-acrobot-placeholder}, and Figure~\ref{fig:midway-carracing-placeholder} show mean evaluation return across five seeds with variation indicated by a standard-deviation band. At the midway training budget, `cartpole_swingup` shows the clearest learning signal, with SAC and TD3 improving substantially from the first to the final evaluation block. `acrobot_swingup` is noisier at 20,000 steps, and the curves should be read as early learning signals rather than convergence evidence. `car_racing_continuous` is the most difficult setup because the agents learn from RGB image observations; the 10,000-step budget is therefore best interpreted as an end-to-end validation of the PPO-CNN, SAC-CNN, and TD3-CNN pipeline.

At the final evaluation block, the notebook reports the following descriptive mean-return rankings: PPO, TD3, SAC on `acrobot_swingup`; SAC, TD3, PPO on `car_racing_continuous`; and SAC, TD3, PPO on `cartpole_swingup`. These rankings are conditioned on the midway budget and default hyperparameters. They should not be interpreted as final algorithm rankings or as evidence of general superiority. The appropriate conclusion is that the PPO/SAC/TD3 baseline matrix is complete and reproducible, establishing the comparison basis for the final GRPO-control stage.

## 8. Suggested figure usage

### `figures/midway/midway_cartpole_swingup_baselines.png`

- **Include:** Yes, if space permits.
- **Caption message:** Five-seed midway baseline learning curves for PPO, SAC, and TD3 on CartPole-Swingup. The x-axis is training steps before evaluation; the y-axis is undiscounted evaluation episode return; curves show mean across seeds with a +/- 1 standard-deviation band.
- **Supports:** CartPole-Swingup has the clearest learning progress within the midway budget, especially for SAC and TD3.
- **Avoid:** Do not claim that the algorithms have converged or that SAC/TD3 are generally superior.

### `figures/midway/midway_acrobot_swingup_baselines.png`

- **Include:** Yes, if space permits.
- **Caption message:** Five-seed midway baseline learning curves for PPO, SAC, and TD3 on Acrobot-Swingup under the same vector-control ObjectRL workflow.
- **Supports:** Acrobot-Swingup is noisier and harder at the 20,000-step midway budget; the curves show early engagement with the task rather than final performance.
- **Avoid:** Do not claim robust convergence or stable task solution.

### `figures/midway/midway_car_racing_continuous_baselines.png`

- **Include:** Yes. This figure is important because it shows that CarRacing is not deferred and that all three CNN baselines are implemented.
- **Caption message:** Five-seed midway baseline learning curves for PPO-CNN, SAC-CNN, and TD3-CNN on continuous CarRacing-v3. Runs were executed on Colab CUDA and imported into the local result pipeline.
- **Supports:** The CarRacing image-observation pipeline runs end-to-end and produces comparable CSV/figure artefacts.
- **Avoid:** Do not claim that 10,000 steps is enough to solve CarRacing or compare asymptotic driving skill.

### `figures/midway/midway_vector_env_baselines.png`

- **Include:** Optional.
- **Caption message:** Compact overview of the midway PPO/SAC/TD3 baseline curves across environments.
- **Supports:** A quick visual sanity check of the full baseline matrix.
- **Avoid:** The notebook describes this as a navigation aid, not a primary report figure. Before using it as the main report figure, inspect whether its title/labels match the intended report wording. If it still says "vector-env" while showing CarRacing, prefer the three per-environment figures or correct the final LaTeX caption carefully.

### Practical recommendation

If page space allows, include the three per-environment figures because they are clear and avoid ambiguity. If page space is tight, use the combined overview as the main figure only after checking that the image itself is suitable for the report. The assignment eventually asks for a single figure with three subpanels for the final report, so the combined figure format is useful, but the per-environment figures are safer for the midway draft.

## 9. Suggested citation placeholders

Only use citation keys that already exist in `report/references.bib`.

- **PPO:** When introducing PPO as an on-policy clipped policy-gradient baseline, cite `\citep{schulman2017proximal}`.
- **GAE/PPO advantage estimation:** If mentioning GAE in relation to PPO-CNN or PPO baselines, cite `\citep{schulman2015gae}`.
- **SAC:** When introducing SAC as an off-policy maximum-entropy actor-critic baseline, cite `\citep{haarnoja2018sacapps}`.
- **TD3:** When introducing TD3 as a deterministic actor-critic continuous-control baseline, cite `\citep{fujimoto2018td3}`.
- **ObjectRL:** When explaining the vector-control baseline implementation framework, cite `\citep{baykal2025objectrl}`.
- **Gymnasium / CarRacing:** When naming CarRacing-v3 from Gymnasium/Farama, cite `\citep{towers2024gymnasium}`.
- **DeepMind Control environments:** When naming cartpole-swingup and acrobot-swingup as DMC tasks, cite `\citep{tunyasuvunakool2020dmcontrol}`.

Possible compact citation sentence:

> PPO, SAC, and TD3 are used as complementary baseline families for policy-gradient, maximum-entropy actor-critic, and deterministic actor-critic learning, respectively \citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}. The vector-control baselines use ObjectRL \citep{baykal2025objectrl}, while CarRacing-v3 is based on the Gymnasium/Farama environment interface \citep{towers2024gymnasium}.

## 10. Final author notes

- Decide manually how many numeric tables to include in the LaTeX report. The safest compact version is one validation/status table plus the learning-curve figure(s); detailed ranking/progress/stability tables can remain in the notebook unless space permits.
- Check the assignment PDF: the final experiments section is expected to include learning curves for GRPO, PPO, SAC, and TD3 in a single figure with three subpanels. The midway report can explain that this section currently covers only the completed PPO/SAC/TD3 baseline matrix.
- Align this section with Methodology by using the same notation for environments, seeds, evaluation returns, and training steps.
- Align with Theory by avoiding claims about convergence until the final GRPO method and assumptions are specified.
- Postpone any claims about GRPO performance, GRPO convergence, or GRPO comparison against PPO/SAC/TD3 until the final report.
- If the final report later increases training budgets or tunes hyperparameters, clearly separate those final experiments from the midway baseline numbers.
- When discussing rankings, always include the caveat: "at the midway training budget and without hyperparameter optimization."
