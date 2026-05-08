# Related Work Draft Inspiration for the DM887 GRPO Midway Report

This document is inspiration material for `report/sections/02_related_work.tex`. It is not final report text. It is written for the interim/midway version of the DM887 GRPO-for-control report, where the assessed related-work task is to position the project relative to PPO, GAE, SAC, TD3, GRPO, ObjectRL, and Gymnasium without pretending that the final GRPO-control method is already implemented.

The final requested output path was `report/drafts/02_related_work/codex_draft.md`; this file follows that target path.

Primary local sources inspected:

- `DM887_Project.pdf`
- `README.md`
- `docs/project.md`
- `docs/project-structure.md`
- `plans/plan-poc.md`
- `plans/plan-midway-rapport-latex.md`
- `docs/scientific-writing/plan-scientific-writing.md`
- `docs/references/reading-list.md`
- `docs/references/plan-references.md`
- `docs/references/paper_urls.md`
- `notebooks/DM887_Project_GRPO_Midway_PoC.ipynb`
- `report/DM887_Report.tex`
- `report/DM887_Report.pdf`
- `report/sections/01_introduction.tex`
- `report/sections/01_introduction_long.tex`
- `report/sections/01_introduction_interim_compressed.tex`
- `report/sections/02_related_work.tex`
- `report/sections/03_methodology.tex`
- `report/sections/03_methodology_long.tex`
- `report/sections/03_methodology_interim_compressed.tex`
- `report/sections/05_experiments.tex`
- `report/sections/05_experiments_long.tex`
- `report/sections/05_experiments_interim_compressed.tex`
- `report/references.bib`
- `papers/ppo_schulman_2017.pdf`
- `papers/gae_schulman_2015.pdf`
- `papers/sac_haarnoja_2018.pdf`
- `papers/td3_fujimoto_2018.pdf`
- `papers/ddpg_lillicrap_2015.pdf`
- `papers/dpg_silver_2014.pdf`
- `papers/deepseekmath_grpo_2024.pdf`
- `papers/deepseek_r1_2025.pdf`
- `papers/objectrl_baykal_2025.pdf`
- `papers/gymnasium_towers_2024.pdf`
- `scripts/run_project_objectrl_baseline.py`
- `scripts/run_carracing_cnn_baseline.py`
- `scripts/carracing_cnn.py`

## 1. Related Work Section Goal Analysis

The Related Work section must do more than summarize papers. In this interim report, it has a specific job: it should explain why the project's baseline algorithms and tooling are the right reference points for the later GRPO-control comparison, and it should identify the gap that remains open at the midway stage.

The section should first motivate PPO, SAC, and TD3 as complementary baseline families. PPO represents on-policy stochastic policy optimization and connects naturally to advantage estimation through GAE. SAC represents off-policy maximum-entropy actor-critic learning, which is relevant because continuous-control tasks often benefit from sample reuse and entropy-regularized exploration. TD3 represents the deterministic actor-critic family for continuous actions, and its connection to DPG and DDPG explains why it is a natural baseline for continuous control rather than just another algorithm name.

The section should then introduce GRPO in the right scope. GRPO comes from language-model reinforcement learning for mathematical reasoning and is described as a PPO variant that uses group-relative rewards or advantages, reducing reliance on a separately trained critic/value model in that setting. This motivates the project, but it does not directly solve control. Control tasks involve online environment interaction, temporal credit assignment, continuous or structured actions, reward trajectories, and seed-sensitive learning dynamics. The related work should therefore say that GRPO motivates the final project stage, not that GRPO is already proven for control.

The section should also mention ObjectRL and Gymnasium as methodology-enabling references. ObjectRL matters because the interim vector-control baselines use ObjectRL's PPO, SAC, and TD3 implementations without modifying the external checkout. Gymnasium matters because the project uses Gymnasium/Farama-style environment interfaces and CarRacing-v3. These should be cited as reproducibility and implementation-context references, not as evidence for algorithmic performance.

Finally, the section should close with a clear gap statement. The gap is not "PPO/SAC/TD3 are bad" and not "GRPO is better". The gap is that GRPO-style group-relative optimization has been developed and popularized in language-model/reasoning contexts, while its adaptation and evaluation for online control benchmarks remains open. At the midway stage, the report establishes the baseline comparison surface; the final report will evaluate the actual GRPO-control variant against that surface.

## 2. Proposed Subsection Structure

### 2.1 Policy Gradients, PPO, and Advantage Estimation

Purpose: Introduce the on-policy policy-gradient baseline family and explain why PPO with GAE is relevant.

Key message: PPO stabilizes policy-gradient updates through a clipped surrogate objective, while GAE provides a practical advantage-estimation mechanism that trades bias and variance. This makes PPO an appropriate on-policy baseline and connects directly to the advantage notation in Methodology.

Connect to papers/files: `schulman2017proximal`, `schulman2015gae`, `report/sections/03_methodology_interim_compressed.tex`, `plans/plan-midway-rapport-latex.md`.

Interim or final: Belongs in the interim report. Keep it concise; do not derive the PPO objective unless the final report later needs more detail.

### 2.2 Off-Policy Maximum-Entropy Actor-Critic Methods

Purpose: Position SAC as the stochastic off-policy actor-critic baseline.

Key message: SAC combines actor-critic learning, off-policy sample reuse, and entropy maximization. It is relevant as a strong continuous-control baseline with a different design philosophy from PPO.

Connect to papers/files: `haarnoja2018sacapps`, `report/sections/03_methodology_interim_compressed.tex`, `scripts/run_project_objectrl_baseline.py`, `scripts/run_carracing_cnn_baseline.py`.

Interim or final: Belongs in the interim report. Avoid claiming that the project's SAC results reproduce the paper's state-of-the-art claims.

### 2.3 Deterministic Actor-Critic Methods: DPG, DDPG, and TD3

Purpose: Explain why TD3 is not just a third baseline, but the representative deterministic actor-critic baseline for continuous actions.

Key message: DPG gives the deterministic policy-gradient foundation, DDPG extends it with deep function approximation and off-policy actor-critic learning, and TD3 addresses overestimation and instability through clipped double Q-learning, delayed policy updates, and target policy smoothing.

Connect to papers/files: `silver2014deterministic`, `lillicrap2015continuous`, `fujimoto2018td3`, `scripts/run_project_objectrl_baseline.py`, `scripts/carracing_cnn.py`.

Interim or final: Belongs in the interim report. Keep the chain short; do not write a full history of deterministic policy gradients.

### 2.4 Group-Relative Policy Optimization

Purpose: Introduce GRPO as the motivation for the final project.

Key message: GRPO was introduced in DeepSeekMath as a PPO-related method for language-model reasoning RL that estimates a baseline from grouped outputs rather than a learned value model. This is the conceptual starting point for the final GRPO-control stage.

Connect to papers/files: `shao2024deepseekmath`, optionally `deepseekai2025deepseekr1`, `DM887_Project.pdf`, `docs/project.md`.

Interim or final: A concise version belongs in the interim report. The final report can expand this once the concrete GRPO-control algorithm is fixed.

### 2.5 Tooling and Benchmark Interfaces

Purpose: Cite the software and environment ecosystem that supports the baseline pipeline.

Key message: ObjectRL supports research prototyping with reusable PPO/SAC/TD3 implementations, while Gymnasium provides a standardized environment interface and includes the CarRacing task used here. These references support reproducibility, not performance conclusions.

Connect to papers/files: `baykal2025objectrl`, `towers2024gymnasium`, optionally `tunyasuvunakool2020dmcontrol`, `scripts/run_project_objectrl_baseline.py`, `scripts/run_carracing_cnn_baseline.py`, notebook Section 6.

Interim or final: Belongs in the interim report, but one compact paragraph is enough.

### 2.6 Gap and Project Positioning

Purpose: End with the reason this project exists.

Key message: Existing baselines cover standard on-policy, off-policy stochastic, and deterministic actor-critic approaches for control, while GRPO motivates a group-relative alternative. The open problem is adapting and evaluating that idea in online control tasks under a reproducible baseline protocol.

Connect to papers/files: all algorithm references, `DM887_Project.pdf`, Introduction, Methodology, Experiments.

Interim or final: Belongs in the interim report. The final report should revise this paragraph after the final GRPO-control method and results exist.

## 3. Paper-to-Role Map

| Paper/resource | BibTeX key in `references.bib` | What it supports | Where to cite | What not to claim |
|---|---|---|---|---|
| PPO paper, Schulman et al. | `schulman2017proximal` | PPO as an on-policy policy-gradient method using a clipped surrogate objective and multiple minibatch updates on sampled data. | PPO/GAE subsection; baseline selection paragraph. | Do not claim the project's PPO setup achieves the paper's reported benchmark performance, or that PPO is generally superior. |
| GAE paper, Schulman et al. | `schulman2015gae` | Advantage estimation as variance reduction with a bias-variance tradeoff; connection between policy gradients, value functions, and advantages. | PPO/advantage-estimation paragraph; Methodology/Theory cross-reference if needed. | Do not claim GAE solves all credit-assignment issues or guarantees stable improvement in this project. |
| SAC paper, Haarnoja et al. | `haarnoja2018sacapps` | SAC as an off-policy maximum-entropy actor-critic method for continuous-control and robotics-style tasks; entropy maximization for exploration/stability motivation. | SAC subsection; baseline-family paragraph. | Do not claim state-of-the-art performance in this report or that the project's short SAC runs reproduce SAC paper conclusions. |
| TD3 paper, Fujimoto et al. | `fujimoto2018td3` | TD3 as an actor-critic method addressing function-approximation error and overestimation with clipped double Q-learning, delayed policy updates, and target smoothing. | TD3 subsection; deterministic actor-critic paragraph. | Do not claim TD3 always outperforms SAC/PPO or that the project proves TD3 superiority. |
| DPG paper, Silver et al. | `silver2014deterministic` | Deterministic policy-gradient theorem and motivation for deterministic policies in continuous-action spaces. | Background sentence before DDPG/TD3. | Do not present DPG as the implemented baseline; the implemented baseline is TD3. |
| DDPG paper, Lillicrap et al. | `lillicrap2015continuous` | Deep deterministic actor-critic learning for continuous control; predecessor to TD3; use of replay and target-network ideas in deep continuous control. | Short bridge from DPG to TD3. | Do not imply DDPG is part of the current experimental matrix. |
| DeepSeekMath / GRPO paper, Shao et al. | `shao2024deepseekmath` | GRPO as a PPO-related method for mathematical reasoning RL; group-relative baseline/advantage from multiple outputs; reduced reliance on a critic in the LLM setting. | GRPO subsection and final gap paragraph. | Do not claim GRPO is proven for continuous control, robotics, or this project's environments. |
| DeepSeek-R1 paper | `deepseekai2025deepseekr1` | Broader context: GRPO used in reasoning-oriented LLM RL; useful if mentioning later prominence of GRPO-style reasoning RL. | Optional one sentence in GRPO context. | Do not overuse it; do not make the project about language-model reasoning or cite it as evidence for control performance. |
| ObjectRL paper, Baykal et al. | `baykal2025objectrl` | ObjectRL as an object-oriented RL codebase for prototyping and baseline implementations including PPO/SAC/TD3. | Tooling/reproducibility paragraph; methodology implementation context. | Do not claim ObjectRL directly supports the project CarRacing image-observation setup used here. |
| Gymnasium paper, Towers et al. | `towers2024gymnasium` | Gymnasium as standardized RL environment API and maintained ecosystem; supports CarRacing/Farama-style environment framing. | Environment/tooling paragraph. | Do not claim Gymnasium supplies the full DMC vector pipeline or proves benchmark validity. |
| DM Control paper, Tunyasuvunakool et al. | `tunyasuvunakool2020dmcontrol` | DM Control Suite as source/context for CartPole Swingup and Acrobot Swingup tasks. | Environment/tooling paragraph if space permits. | Do not over-expand environment literature if the related-work section must stay short. |
| RLHF/InstructGPT paper, Ouyang et al. | `ouyang2022training` | Optional context for PPO in language-model RLHF and why PPO-style methods are relevant in LLM post-training. | Optional one sentence if linking PPO-to-GRPO in LLM training. | Do not make the related work about RLHF broadly; it is peripheral for this interim report. |

Note: the current `report/references.bib` contains `haarnoja2018sacapps` and `fujimoto2018td3`. It does not contain the keys `haarnoja2018soft` or `fujimoto2018addressing`; final LaTeX should use the existing keys or update the bibliography deliberately.

## 4. Safe Claims

Safe related-work claims:

- PPO is an on-policy policy-gradient method based on stabilized/clipped policy updates.
- PPO is a relevant baseline because it is close in spirit to GRPO's PPO-style clipped objective.
- GAE is an advantage-estimation method that trades bias and variance in policy-gradient estimation.
- SAC is an off-policy actor-critic method based on the maximum-entropy RL framework.
- SAC is relevant because it represents a stochastic off-policy actor-critic baseline for continuous-control tasks.
- DPG provides the deterministic policy-gradient foundation for continuous actions.
- DDPG extends deterministic policy-gradient ideas to deep off-policy actor-critic learning in continuous-action spaces.
- TD3 builds on the DPG/DDPG family and addresses actor-critic function-approximation error and overestimation through twin critics/clipped double Q, delayed policy updates, and target policy smoothing.
- PPO, SAC, and TD3 represent different design families: on-policy stochastic policy-gradient, off-policy stochastic maximum-entropy actor-critic, and off-policy deterministic actor-critic.
- GRPO was introduced in DeepSeekMath in the context of mathematical reasoning for language models.
- GRPO is described as a PPO variant that estimates group-relative advantages/baselines from multiple sampled outputs and reduces reliance on a separate value model in the LLM setting.
- DeepSeek-R1 provides broader context for GRPO-style RL in reasoning-oriented language-model training, if used sparingly.
- ObjectRL is an open-source RL codebase intended for research-oriented prototyping and includes PPO/SAC/TD3 implementations.
- Gymnasium provides a standardized RL environment API and is relevant to the CarRacing-v3 environment interface.
- The current project uses PPO, SAC, and TD3 as baselines, not as the novel contribution.
- The midway report establishes the baseline comparison surface for a later GRPO-control method.
- The transfer from language-model GRPO to online control is not direct.
- The final report should evaluate the GRPO-control variant against the completed PPO/SAC/TD3 baseline matrix.

Safe repo-status claims, when needed for positioning:

- This is a midway/interim report, not the final report.
- The final GRPO-control method has not yet been implemented in the reported experiments.
- The full midway baseline matrix has been completed: 3 algorithms x 3 environments x 5 seeds = 45 runs.
- The final notebook validates 900 evaluation rows.
- Vector-control environments use ObjectRL; CarRacing uses project-side CNN implementations because the image-observation setup does not fit ObjectRL's standard vector-observation workflow.
- No hyperparameter optimization was performed.

## 5. Claims to Avoid

Avoid these claims:

- Do not claim GRPO is proven better than PPO/SAC/TD3 for control.
- Do not claim the final GRPO-control method has already been implemented or evaluated.
- Do not claim state-of-the-art performance.
- Do not claim that the midway baseline results prove algorithmic superiority in general.
- Do not claim all algorithms have converged.
- Do not claim the project reproduces the full results of the PPO, SAC, TD3, DDPG, or GRPO papers.
- Do not claim SAC or TD3 should always outperform PPO in every control setting.
- Do not claim PPO is unsuitable for continuous control; the assignment explicitly includes PPO as a baseline.
- Do not claim GRPO eliminates the need for value estimation in control. At most, say that GRPO reduces reliance on a learned critic in the LLM setting and motivates investigating whether group-relative signals can be adapted to control.
- Do not claim DeepSeek-R1 is necessary for the project. It is optional context.
- Do not say CarRacing is deferred.
- Do not say PPO-CNN is not implemented.
- Do not say the final notebook is dry-run only.
- Do not imply ObjectRL directly supports the CarRacing image-observation setup used here.
- Do not cite debug runs as main experiments.
- Do not introduce new experiments, new results, or new implementation details in Related Work.
- Do not write a long textbook survey of RL; the interim report has a five-page target and needs a focused section.

## 6. Danish Draft

Relateret arbejde i denne rapport skal primært forklare, hvorfor PPO, SAC og TD3 udgør passende baselines for den senere GRPO-baserede kontrolmetode. De tre algoritmer repræsenterer forskellige hovedretninger i moderne deep reinforcement learning: on-policy policy-gradient-metoder, off-policy stokastiske aktør-kritiker-metoder og deterministiske aktør-kritiker-metoder til kontinuerte handlingsrum. Dermed giver de ikke blot tre navne at sammenligne med, men et bredere sammenligningsgrundlag for at vurdere, hvilken type optimeringsprincip en senere GRPO-variant skal måles imod.

PPO er relevant, fordi metoden stabiliserer policy-gradient-opdateringer gennem et klippet surrogate objective \citep{schulman2017proximal}. PPO er samtidig tæt beslægtet med projektets GRPO-motivation, da GRPO i DeepSeekMath beskrives som en variant af PPO. For at bruge policy-gradient-metoder effektivt kræves der typisk et estimat af en fordel-funktion. Generalized Advantage Estimation (GAE) giver et praktisk estimat, hvor bias og varians kan afvejes gennem en eksponentielt vægtet konstruktion \citep{schulman2015gae}. I dette projekt bruges PPO derfor som den on-policy baseline, der tydeligst forbinder den klassiske policy-gradient-litteratur med den senere GRPO-diskussion.

SAC repræsenterer en anden baseline-familie. Metoden er en off-policy aktør-kritiker-algoritme baseret på maximum-entropy reinforcement learning, hvor politikken både optimerer forventet return og entropi \citep{haarnoja2018sacapps}. Dette gør SAC relevant for kontrolopgaver, hvor genbrug af erfaring og eksploration er centrale praktiske hensyn. I denne rapport bruges SAC ikke som et bevis på, hvad der er optimalt for de valgte miljøer, men som en stærk og velkendt off-policy baseline, der supplerer PPO's on-policy perspektiv.

TD3 placerer projektet i den deterministiske aktør-kritiker-linje. Deterministic Policy Gradient viser, hvordan en deterministisk politik kan optimeres i kontinuerte handlingsrum ved hjælp af gradienten af en action-value-funktion \citep{silver2014deterministic}. DDPG udvider denne idé til deep reinforcement learning med off-policy læring, replay buffer og target-netværk \citep{lillicrap2015continuous}. TD3 videreudvikler denne familie ved at adressere overestimering og funktionsapproksimationsfejl gennem blandt andet twin critics, forsinkede policy-opdateringer og target policy smoothing \citep{fujimoto2018td3}. TD3 er derfor en naturlig baseline for projektets kontinuerte kontrolopgaver.

GRPO giver den metodiske motivation for den endelige projektfase. I DeepSeekMath introduceres Group Relative Policy Optimization som en PPO-relateret metode til reinforcement learning for matematisk ræsonnement i sprogmodeller \citep{shao2024deepseekmath}. I stedet for at estimere en baseline udelukkende med en separat value model anvender GRPO relative scores inden for grupper af svar på samme prompt. DeepSeek-R1 viser senere, at GRPO også indgår i bredere reasoning-orienteret RL for sprogmodeller \citep{deepseekai2025deepseekr1}. Dette er dog kontekst, ikke evidens for kontrolopgaver. Overførslen fra sprogmodel-RL til online kontrol er ikke direkte, fordi kontrolopgaver kræver sekventiel kreditallokering, håndtering af kontinuerte handlinger og stabil læring fra miljøinteraktion over tid.

Projektet bygger også på eksisterende værktøjer og miljøstandarder. ObjectRL er relevant som et objektorienteret framework til reinforcement-learning-prototyping og bruges i dette projekt som implementeringskilde for PPO, SAC og TD3 på de vektorbaserede kontrolmiljøer \citep{baykal2025objectrl}. Gymnasium er relevant som standardiseret miljøinterface og som kilde til CarRacing-v3-miljøet \citep{towers2024gymnasium}. Disse værktøjsreferencer understøtter reproducerbarhed og eksperimentel struktur, men de skal ikke bruges som argumenter for algoritmisk overlegenhed.

Det åbne spørgsmål er derfor ikke, om PPO, SAC eller TD3 generelt er bedst, og heller ikke om GRPO allerede er bedre til kontrol. Spørgsmålet er, om den group-relative idé fra GRPO kan tilpasses online kontrolopgaver på en måde, der kan evalueres retfærdigt mod etablerede baselines. Ved midtvejsrapporten er bidraget at etablere dette sammenligningsgrundlag: PPO, SAC og TD3 er kørt på alle tre miljøer med fem seeds, mens den endelige GRPO-control-variant er planlagt til næste projektfase.

## 7. English Academic Draft

The related work in this interim report serves to justify the baseline selection and to position the later GRPO-control contribution. PPO, SAC, and TD3 are not included as arbitrary comparisons; they represent distinct families of deep reinforcement learning methods. PPO provides an on-policy stochastic policy-gradient baseline, SAC provides an off-policy maximum-entropy actor-critic baseline, and TD3 provides an off-policy deterministic actor-critic baseline for continuous action spaces. Together, they define the comparison surface on which the final GRPO-control method should be evaluated.

PPO is relevant because it stabilizes policy-gradient learning through a clipped surrogate objective \citep{schulman2017proximal}. It is also conceptually close to the GRPO motivation, since GRPO is introduced as a PPO-related method. Policy-gradient methods typically require an estimate of the advantage function, and Generalized Advantage Estimation provides a practical way to trade bias and variance when constructing such estimates \citep{schulman2015gae}. PPO with advantage estimation is therefore the natural on-policy baseline for this project and connects directly to the value and advantage notation introduced in the Methodology section.

SAC represents a different design family. It is an off-policy actor-critic method based on the maximum-entropy reinforcement learning framework, where the policy is trained to maximize both expected return and entropy \citep{haarnoja2018sacapps}. This makes SAC relevant for continuous-control settings in which sample reuse, exploration, and stability are important practical concerns. In this report, SAC is used as a standard off-policy stochastic actor-critic baseline. The midway results should not be interpreted as reproducing the full claims of the SAC paper; they only describe behaviour under the fixed interim protocol.

TD3 represents the deterministic actor-critic family. The deterministic policy-gradient framework shows how deterministic policies can be optimized in continuous action spaces using gradients of the action-value function \citep{silver2014deterministic}. DDPG extends this idea to deep off-policy actor-critic learning in continuous-control domains \citep{lillicrap2015continuous}. TD3 further addresses function-approximation error and overestimation in actor-critic learning through mechanisms such as clipped double Q-learning, delayed policy updates, and target policy smoothing \citep{fujimoto2018td3}. This makes TD3 a natural deterministic baseline for the continuous-control tasks used in the project.

GRPO provides the motivation for the final project stage. DeepSeekMath introduced Group Relative Policy Optimization as a PPO-related method for mathematical reasoning in language models, using group-relative scores to estimate the update signal and reduce reliance on a separate value model in that setting \citep{shao2024deepseekmath}. DeepSeek-R1 later provides broader context for GRPO-style reinforcement learning in reasoning-oriented language-model training \citep{deepseekai2025deepseekr1}. However, this literature does not directly establish GRPO as a control algorithm. Online control requires temporal credit assignment, continuous or structured actions, environment interaction, and evaluation across seeds and training budgets. These differences motivate the project rather than solving it.

The experimental setup also depends on reproducible tooling and environment interfaces. ObjectRL is used as the implementation framework for the vector-control PPO, SAC, and TD3 baselines, following its goal of supporting research-oriented RL prototyping through reusable components \citep{baykal2025objectrl}. Gymnasium provides the standardized environment interface relevant to the CarRacing-v3 task and the broader Farama environment ecosystem \citep{towers2024gymnasium}. In the current repository, CarRacing is handled by project-side CNN implementations because the image-observation setup does not fit directly into ObjectRL's standard vector-observation workflow.

The resulting gap is clear. Existing control baselines provide established reference points for on-policy, off-policy stochastic, and deterministic actor-critic learning, while GRPO provides a group-relative optimization idea developed in language-model reasoning settings. What remains open is the adaptation and evaluation of this idea for online control benchmarks. At the midway stage, this report establishes the baseline comparison surface; the final report will evaluate the GRPO-control variant against PPO, SAC, and TD3 under the same environment set, seed structure, and evaluation metric.

## 8. Suggested Citation Placement

Use real BibTeX keys only where they already exist in `report/references.bib`.

Suggested citation placement in the English draft:

- Paragraph 1, baseline family overview: optional combined citation after listing PPO/SAC/TD3, e.g. `\citep{schulman2017proximal,haarnoja2018sacapps,fujimoto2018td3}`. This is optional because the next paragraphs cite each method individually.
- Paragraph 2, first PPO sentence: `\citep{schulman2017proximal}`.
- Paragraph 2, GAE sentence: `\citep{schulman2015gae}`.
- Paragraph 3, SAC description: `\citep{haarnoja2018sacapps}`.
- Paragraph 4, deterministic policy-gradient foundation: `\citep{silver2014deterministic}`.
- Paragraph 4, DDPG bridge: `\citep{lillicrap2015continuous}`.
- Paragraph 4, TD3 mechanisms: `\citep{fujimoto2018td3}`.
- Paragraph 5, GRPO/DeepSeekMath: `\citep{shao2024deepseekmath}`.
- Paragraph 5, optional DeepSeek-R1 context: `\citep{deepseekai2025deepseekr1}`. Use only if the final section has enough space.
- Paragraph 6, ObjectRL tooling: `\citep{baykal2025objectrl}`.
- Paragraph 6, Gymnasium environment interface: `\citep{towers2024gymnasium}`.
- If DeepMind Control Suite is mentioned explicitly for CartPole Swingup and Acrobot Swingup, cite `\citep{tunyasuvunakool2020dmcontrol}`.
- If RLHF/PPO in LLM post-training is mentioned as context, cite `\citep{ouyang2022training}`, but this is optional and probably not needed in a compressed five-page interim report.

Recommended citation density for the final compressed section: about 7-10 citations total, not every sentence.

## 9. Gap Statement Variants

### Variant 1

GRPO provides a group-relative policy-optimization idea that has been developed in language-model reasoning settings, but its transfer to online control is not automatic. The open problem for this project is to adapt that idea to trajectory-based control tasks and evaluate it against PPO, SAC, and TD3 under the same environments, seeds, and evaluation metric.

### Variant 2

The existing literature supplies strong baselines for continuous control and a recent group-relative alternative from language-model reinforcement learning. What remains open is whether a GRPO-style update can be made suitable for control, where rewards arrive through sequential environment interaction and performance must be assessed across seeds and training budgets.

### Variant 3

At the midway stage, the literature motivates the comparison but does not close the project gap. PPO, SAC, and TD3 define established control baselines, while GRPO motivates a different way to construct policy-update signals. The final project stage must determine whether this group-relative idea can be adapted and evaluated meaningfully in the selected control benchmarks.

## 10. Suggested Final Target Length

For the current five-page interim draft, the final `02_related_work.tex` should be short:

- Target: 5-6 paragraphs.
- Approximate length: 0.6-0.8 pages in the NeurIPS template.
- Suggested paragraph allocation:
  - 1 opening/positioning paragraph;
  - 1 PPO/GAE paragraph;
  - 1 SAC paragraph;
  - 1 DPG/DDPG/TD3 paragraph;
  - 1 GRPO paragraph;
  - 1 tooling/gap paragraph, or split this into two if space allows.

If space is tight, combine ObjectRL/Gymnasium with the gap paragraph and omit DeepSeek-R1. Do not let Related Work grow beyond about one page in the interim report unless other sections are shortened.

## 11. Final Author Notes

Manual decisions:

- Decide whether the final compressed section should include DeepSeek-R1. My recommendation: include it only if there is space; DeepSeekMath is the core GRPO source.
- Decide whether to mention TRPO or conservative policy improvement. These are relevant to PPO history, but likely too much for the interim section unless Theory later needs them.
- Decide whether to cite DM Control explicitly in Related Work or leave it to Methodology/Experiments. If environments are named in Related Work, cite `tunyasuvunakool2020dmcontrol`.
- Decide how much of the ObjectRL/Gymnasium tooling paragraph belongs in Related Work versus Methodology. Related Work should cite and position; Methodology should explain the concrete implementation split.

Check against the assignment PDF:

- The Related Work section is explicitly evaluated as part of the interim report.
- The section should point to relevant existing work and state which limitation or gap the project addresses.
- The final project must compare against PPO, SAC, TD3, and GRPO; the interim related work should prepare that comparison without claiming the GRPO method is complete.
- The assignment describes a final target where the GRPO variant should work better than PPO and on par with SAC/TD3. Do not phrase this as an achieved result in the interim report.

Align later with Methodology, Theory, and Experiments:

- Methodology should define MDPs, policies, returns, value functions, and advantages; Related Work should not duplicate those definitions in detail.
- Theory should later contain deeper formal claims about the GRPO-control update and convergence assumptions; Related Work should only motivate why such a method is interesting.
- Experiments should contain the 45-run/900-row validation and learning-curve interpretation; Related Work should mention the baseline families, not the numeric results.
- The Introduction already states the project status and baseline matrix. Related Work should add literature positioning rather than repeating the full project status.

Postpone until the final report:

- Full GRPO-control algorithm comparison.
- Detailed GRPO-control objective and pseudocode.
- Convergence-theory discussion.
- Hyperparameter tuning discussion.
- Claims about GRPO outperforming PPO or matching SAC/TD3.
- Full RLHF or reasoning-model survey.

Citation/key cleanup note:

- Use `haarnoja2018sacapps` for SAC and `fujimoto2018td3` for TD3 in final LaTeX. The keys `haarnoja2018soft` and `fujimoto2018addressing` are not present in the current `report/references.bib`.
