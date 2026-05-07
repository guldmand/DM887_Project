# Introduction draft inspiration for the DM887 midway report

This document is inspiration material for the Introduction section. It is not final report text and should be manually refined before being copied into `report/sections/01_introduction.tex`.

## 1. Introduction goal analysis

The Introduction for this midway report should quickly orient the reader around three things: the final project ambition, the current midway contribution, and the evidence that the project is on a reproducible path toward the final GRPO comparison.

The final project ambition is to investigate whether Group Relative Policy Optimization (GRPO), originally motivated in large-language-model reinforcement learning, can be adapted to control tasks. The Introduction should therefore motivate continuous-control reinforcement learning as a setting where stable policy optimization, reliable advantage estimation, and reproducible evaluation matter. It should not become a general reinforcement learning textbook introduction.

At the midway stage, the report should be honest that the GRPO-control method is not yet the completed contribution. The current contribution is the foundation needed before the GRPO variant can be evaluated: the literature framing, MDP notation, environment setup, baseline execution pipeline, result validation, and preliminary PPO/SAC/TD3 baseline matrix.

The Introduction should also make clear why PPO, SAC, and TD3 are included. They are not arbitrary algorithms: PPO represents an on-policy clipped policy-gradient family, SAC represents off-policy maximum-entropy actor-critic learning, and TD3 represents deterministic actor-critic learning for continuous control. Together they form the baseline comparison that the final GRPO-related extension must build on.

The main midway takeaway should be framed as pipeline validation, not algorithmic superiority. The completed matrix contains 3 algorithms x 3 environments x 5 seeds = 45 runs and 900 evaluation rows. These results validate that the repository can run, collect, aggregate, and visualize the required baseline experiments. However, because no hyperparameter optimization was performed and the budgets are deliberately limited, the Introduction should avoid strong performance conclusions.

The Introduction should also align with the assignment PDF: the final report is expected to motivate the significance of the studied problem and the main takeaways of the proposed solution. Since this is a midway report, the "main takeaway" is not yet a finished GRPO-control method, but the establishment of a reproducible experimental and notation foundation for the final method.

## 2. Paragraph-by-paragraph outline

### Paragraph 1 - Problem context: RL for control

- **Purpose:** Introduce the project area and why control tasks are a meaningful testbed.
- **Key message:** Continuous-control reinforcement learning requires stable learning algorithms and careful empirical evaluation because policies interact sequentially with dynamic environments.
- **Evidence/context to connect to:** The assignment requires CarRacing-v3, cartpole-swingup-v0, and acrobot-swingup-v0; the evaluation metric is undiscounted evaluation episode return over regular evaluation intervals.

### Paragraph 2 - Baseline landscape

- **Purpose:** Explain why PPO, SAC, and TD3 are the relevant baseline algorithms.
- **Key message:** The project compares against three established families: on-policy clipped policy optimization, off-policy entropy-regularized actor-critic learning, and deterministic actor-critic learning.
- **Evidence/context to connect to:** The report bibliography already contains PPO, SAC, and TD3 references; the notebook validates PPO/SAC/TD3 results for all three environments and seeds 0..4.

### Paragraph 3 - GRPO motivation and gap

- **Purpose:** Position GRPO as the final project motivation without claiming final results.
- **Key message:** GRPO is interesting because it uses group-relative information for policy optimization, but adapting this idea to control is non-trivial due to temporal credit assignment, continuous actions, environment dynamics, and the role of value estimation.
- **Evidence/context to connect to:** The assignment asks for a GRPO variant for control and suggests considering value-function or multi-step TD ideas; the report references include DeepSeekMath / GRPO context.

### Paragraph 4 - Midway scope and contribution

- **Purpose:** State clearly what this midway report accomplishes.
- **Key message:** At the midway stage, the contribution is not a completed GRPO-control algorithm; it is the reproducible baseline pipeline, formal setup, environment handling, and completed baseline matrix needed for the final comparison.
- **Evidence/context to connect to:** The report-facing notebook validates 45 CSV files, 900 rows, 0 missing combinations; vector-control baselines use ObjectRL, while CarRacing uses project-side CNN implementations because image observations are not directly supported by the ObjectRL baseline path used here.

### Paragraph 5 - Result framing and limitations

- **Purpose:** Prevent overclaiming while still making the midway progress meaningful.
- **Key message:** The preliminary baseline results validate the experimental pipeline and provide a reference matrix, but they should not be interpreted as state-of-the-art claims or general algorithm rankings.
- **Evidence/context to connect to:** No hyperparameter optimization was performed; the midway budgets are limited; CarRacing CNN runs were executed on Colab CUDA and copied back into the repository for aggregation.

### Paragraph 6 - Report structure and next stage

- **Purpose:** End the Introduction by guiding the reader through the report.
- **Key message:** The report reviews related work, defines MDP notation and experimental methodology, presents the midway baseline results, and identifies the final-stage work: implementing and evaluating the GRPO-related extension.
- **Evidence/context to connect to:** The LaTeX template contains Introduction, Related Work, Methodology, Theory, Experiments, Conclusion, and appendix sections; theory and final GRPO experiments remain forward-looking at the midway stage.

## 3. Safe claims

The following claims are safe to make in the Introduction based on the current repository status and the report-facing notebook:

- The project investigates reinforcement learning for control-style benchmark environments with a final focus on adapting GRPO to control.
- PPO, SAC, and TD3 are used as baseline algorithms.
- The midway baseline matrix covers 3 algorithms, 3 environments, and 5 seeds.
- The completed midway matrix contains 45 runs and 900 evaluation rows.
- The three environments are `cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous`.
- The vector-control environments are handled through the ObjectRL-based workflow.
- CarRacing uses project-side PyTorch CNN baseline implementations because the ObjectRL baseline path used here targets 1-D vector observations, while CarRacing uses image observations.
- The CarRacing CNN runs were executed on Google Colab with CUDA and copied back into the local repository.
- The current report-facing notebook validates that all expected algorithm-environment-seed combinations are present.
- The baseline results are preliminary and should be interpreted under the midway training budgets.
- No hyperparameter optimization was performed.
- The current midway contribution is a reproducible experimental pipeline and validated baseline matrix, not the final GRPO method.
- The final project stage will build on this matrix by implementing and evaluating the GRPO-related control extension.

## 4. Claims to avoid

The Introduction should avoid the following claims because they overstate or misrepresent the current midway state:

- Do not claim that the final GRPO-control algorithm has already been implemented or evaluated.
- Do not claim that GRPO already outperforms PPO, SAC, or TD3 in this project.
- Do not claim state-of-the-art performance.
- Do not claim that the baseline results prove one algorithm is generally superior to another.
- Do not claim that the results are final or fully optimized.
- Do not claim that hyperparameters were tuned.
- Do not say CarRacing is deferred; the CarRacing CNN baseline matrix has been run and imported.
- Do not say PPO-CNN is missing; the notebook documents PPO-CNN, SAC-CNN, and TD3-CNN for CarRacing.
- Do not describe the final notebook as a dry-run notebook; it is the report-facing midway notebook validating completed baseline results.
- Do not imply that ObjectRL directly supports the exact CarRacing image-observation setup used here.
- Do not imply that wall-clock comparisons between vector environments and CarRacing are meaningful, since they were run on different hardware paths.
- Do not claim that the midway budgets are sufficient for asymptotic performance.
- Do not present preliminary learning-curve ordering as a robust scientific conclusion.

## 5. Danish draft

Reinforcement learning til kontrolopgaver undersøger, hvordan en agent kan lære beslutningsstrategier gennem interaktion med et dynamisk miljø. I kontinuerte kontrolproblemer er dette særligt udfordrende, fordi agentens handlinger både skal være præcise, stabile og robuste over tid. Dette projekt tager udgangspunkt i denne problemstilling og undersøger, hvordan Group Relative Policy Optimization (GRPO) kan videreudvikles til kontrolopgaver, hvor metoden senere skal sammenlignes med etablerede baselines.

Som referencepunkt anvender projektet PPO, SAC og TD3. Disse algoritmer repræsenterer forskellige tilgange til moderne deep reinforcement learning: PPO som en on-policy metode med stabiliserede policy-opdateringer, SAC som en off-policy maksimum-entropi actor-critic metode, og TD3 som en deterministisk actor-critic metode udviklet til kontinuerte handlingsrum. Baseline-sammenligningen er derfor central for at kunne vurdere en senere GRPO-baseret udvidelse på en meningsfuld måde.

GRPO er interessant i denne sammenhæng, fordi metoden anvender gruppe-relative signaler i policy-optimeringen og dermed adskiller sig fra mere traditionelle advantage-estimater. Overførslen fra sprogmodel-baseret reinforcement learning til kontrolopgaver er dog ikke direkte. Kontrolopgaver kræver sekventiel credit assignment, håndtering af kontinuerte handlinger og stabil læring i miljøer med forskellig observationsstruktur. Det motiverer først at etablere en solid eksperimentel baseline, før den endelige GRPO-variant implementeres og evalueres.

Ved midway-stadiet præsenterer denne rapport derfor ikke den færdige GRPO-kontrolalgoritme. I stedet etableres det nødvendige fundament for den afsluttende projektfase. Rapporten samler den relevante metodebaggrund, fastlægger den formelle MDP-notation og dokumenterer en reproducerbar baseline-pipeline for PPO, SAC og TD3. Den færdige midway-matrix består af 3 algoritmer, 3 miljøer og 5 seeds, i alt 45 kørsler og 900 evalueringsrækker.

De vektorbaserede kontrolmiljøer køres gennem ObjectRL, mens CarRacing behandles med projektets egne CNN-baserede implementeringer, fordi den anvendte ObjectRL-baseline ikke direkte understøtter billedobservationerne i CarRacing-opsætningen. CarRacing-kørslerne er udført på Google Colab med CUDA og efterfølgende kopieret tilbage til det lokale repository. Resultaterne validerer dermed den samlede eksperimentelle pipeline, men de skal fortolkes forsigtigt: der er ikke udført hyperparameteroptimering, og baseline-resultaterne er foreløbige.

Rapportens videre struktur er som følger. Først placeres projektet i forhold til PPO, SAC, TD3 og GRPO. Derefter introduceres den formelle MDP-notation og den eksperimentelle opsætning. Eksperimentafsnittet præsenterer de validerede midway-baselines, mens den afsluttende del beskriver begrænsningerne ved det nuværende stadie og det videre arbejde mod den endelige GRPO-relaterede udvidelse.

## 6. English academic draft

Reinforcement learning for control studies how agents can learn decision-making policies through interaction with dynamic environments. In continuous-control settings, this requires algorithms that can update policies stably while coping with sequential credit assignment, stochastic training dynamics, and environment-dependent reward structure. This project investigates this setting with the final goal of adapting Group Relative Policy Optimization (GRPO) to control tasks and comparing it against established reinforcement learning baselines.

The baseline methods used in this project are PPO, SAC, and TD3. These algorithms represent complementary families of deep reinforcement learning: PPO is an on-policy policy-gradient method based on clipped policy updates, SAC is an off-policy maximum-entropy actor-critic method, and TD3 is a deterministic actor-critic method designed for continuous-action problems. Establishing these baselines is necessary before a GRPO-related extension can be evaluated in a meaningful experimental setting.

GRPO is relevant to this project because it replaces standard value-based advantage estimation with a group-relative optimization signal. While this idea has been influential in language-model reinforcement learning, its use in control is not immediate. Control tasks involve temporally extended trajectories, continuous or structured action spaces, and environment-specific observation modalities. The project therefore first establishes the notation, software pipeline, and baseline results needed to support a later GRPO-control method.

At the midway stage, this report does not claim to present the final GRPO-control algorithm. Instead, it validates the experimental foundation for the final comparison. The completed baseline matrix consists of PPO, SAC, and TD3 evaluated on `cartpole_swingup`, `acrobot_swingup`, and `car_racing_continuous` with five seeds per combination, giving 45 runs and 900 evaluation rows. The vector-control environments use the ObjectRL-based workflow, while CarRacing uses project-side CNN implementations because the image-observation setup used here is not directly supported by the ObjectRL baseline path.

The baseline results should be interpreted as preliminary midway evidence. No hyperparameter optimization was performed, and the training budgets are limited relative to what would be expected for a final performance study. The contribution of the midway report is therefore not an algorithmic performance claim, but a reproducible baseline pipeline and validated result matrix that the final GRPO-related extension can build on.

The remainder of the report reviews the relevant background on PPO, SAC, TD3, and GRPO; defines the MDP notation used throughout the project; describes the experimental setup and implementation choices; presents the validated midway baseline results; and outlines the remaining work toward the final GRPO-control method.

## 7. Suggested citation placeholders

Use only citation keys that already exist in `report/references.bib`. The following placeholders are safe with the current bibliography:

- **PPO background:** In the sentence describing PPO as an on-policy clipped policy-gradient method, add `\citep{schulman2017proximal}`.
- **GAE / advantage estimation context:** If the Introduction mentions value-based or GAE-style advantage estimation, add `\citep{schulman2015gae}`.
- **SAC baseline:** In the sentence describing SAC as an off-policy maximum-entropy actor-critic baseline, add `\citep{haarnoja2018sacapps}`.
- **TD3 baseline:** In the sentence describing TD3 as a deterministic actor-critic method for continuous control, add `\citep{fujimoto2018td3}`.
- **GRPO source:** In the first sentence introducing GRPO, add `\citep{shao2024deepseekmath}`.
- **GRPO / reasoning RL broader context:** If discussing the broader recent prominence of GRPO-style reinforcement learning in reasoning models, optionally add `\citep{deepseekai2025deepseekr1}`.
- **ObjectRL implementation framework:** In the sentence explaining the ObjectRL-based vector-control baseline workflow, add `\citep{baykal2025objectrl}`.
- **Gymnasium / CarRacing:** In the sentence identifying CarRacing-v3 from Gymnasium/Farama, add `\citep{towers2024gymnasium}`.
- **DeepMind Control environments:** If the Introduction explicitly names cartpole-swingup-v0 and acrobot-swingup-v0 as DMC-style environments, add `\citep{tunyasuvunakool2020dmcontrol}`.

A compact citation version of the English draft could place citations like this:

> The baseline methods used in this project are PPO, SAC, and TD3, representing clipped on-policy policy optimization, off-policy maximum-entropy actor-critic learning, and deterministic actor-critic learning for continuous control, respectively \citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}. GRPO is relevant because it replaces standard value-based advantage estimation with a group-relative optimization signal \citep{shao2024deepseekmath}. The vector-control baselines use ObjectRL \citep{baykal2025objectrl}, while CarRacing is based on the Gymnasium/Farama environment interface \citep{towers2024gymnasium}.

## 8. Final author notes

- Decide manually how strongly the Introduction should emphasize the final GRPO ambition versus the current midway baseline contribution. For the midway version, the baseline foundation should probably be more prominent than the unfinished GRPO method.
- Check the assignment PDF wording carefully: the Introduction is expected to motivate the significance of the problem and the main takeaways of the proposed solution. In the midway report, phrase the takeaway as a validated experimental foundation, not as a completed solution.
- Align the Introduction with the Methodology section by using the same names for environments, algorithms, seeds, metric, and pipeline components.
- Align the Introduction with the Experiments section by repeating the exact result-matrix facts: 3 algorithms, 3 environments, 5 seeds, 45 runs, 900 rows, no missing combinations.
- Make sure the Experiments section also states the same limitation as the Introduction: no hyperparameter optimization was performed.
- Keep CarRacing wording consistent across sections: it is completed for the midway baseline matrix, uses project-side CNN implementations, was run on Colab CUDA, and should not be described as deferred.
- When the final GRPO extension is implemented later, revise the Introduction so the contribution statement changes from "baseline foundation" to the actual proposed method and its evaluated findings.
