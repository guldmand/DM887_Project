# Related Work draft material for the DM887 GRPO interim report

This document is inspiration material, not final report text. It is written to help compress and refine the final `report/sections/02_related_work.tex` later without overclaiming the current midway state.

## 1. Related Work section goal analysis

The Related Work section should explain why the project compares against PPO, SAC, and TD3, and why GRPO is an interesting but not-yet-implemented direction for the final project stage. In this interim report, the section should not read as a general reinforcement learning textbook chapter. Its main role is to position the baseline matrix and motivate the final GRPO-control question.

The section should accomplish four things:

1. **Motivate the baseline families.** PPO, SAC, and TD3 are not arbitrary baselines. They represent different standard approaches to deep reinforcement learning for control: on-policy policy-gradient learning, off-policy maximum-entropy actor-critic learning, and deterministic actor-critic learning for continuous actions.

2. **Connect advantage estimation to the later GRPO idea.** PPO and GAE are especially relevant because GRPO is described as a PPO variant and because the central methodological question concerns how policy-update signals or advantages are estimated. The section should make this connection without presenting the final GRPO-control algorithm as completed.

3. **Explain the tooling context.** ObjectRL and Gymnasium matter because the interim report emphasizes reproducibility and a validated baseline pipeline. ObjectRL supports the vector-control baseline implementation, while Gymnasium provides the standardized environment interface used in the project setup. CarRacing is included through project-side CNN implementations because its image observations do not directly fit the ObjectRL vector-observation workflow used here.

4. **State the gap.** GRPO has been introduced and used in language-model mathematical reasoning contexts. The open project question is whether a group-relative policy-optimization idea can be adapted and evaluated for online control benchmarks. At the midway stage, the answer is not yet known; the report establishes the baseline comparison surface needed to evaluate it later.

The tone should be cautious. The section can say that GRPO motivates the final project stage, but it should not claim that GRPO is already known to outperform PPO, SAC, or TD3 in control. It should also avoid claiming final algorithmic conclusions from the midway PPO/SAC/TD3 baselines.

## 2. Proposed subsection structure

| Suggested subsection | Purpose | Key message | Connects to | Interim or final? |
|---|---|---|---|---|
| Policy-gradient methods and advantage estimation | Introduce the family of methods most directly related to PPO and GRPO. | Policy-gradient methods optimize policies from sampled trajectories; advantage estimates reduce variance and shape the update signal. | GAE paper, PPO paper, Methodology notation for \(A^\pi(s,a)\). | Belongs in the interim report, but keep compact. |
| PPO and GAE | Explain why PPO is the on-policy baseline and why GAE is relevant to advantage estimation. | PPO stabilizes policy updates using a clipped surrogate objective; GAE provides a practical bias-variance tradeoff for advantage estimation. | `schulman2017proximal`, `schulman2015gae`; Introduction baseline framing; Methodology advantage notation. | Belongs in the interim report. |
| Off-policy maximum-entropy actor-critic methods | Position SAC as a different baseline family. | SAC is included because it is an off-policy actor-critic method for continuous control that optimizes both expected return and entropy. | `haarnoja2018sacapps`; Experiments baseline matrix. | Belongs in the interim report. |
| Deterministic actor-critic methods | Connect DPG, DDPG, and TD3. | TD3 belongs to the deterministic policy-gradient family and addresses overestimation issues using twin critics, delayed policy updates, and target smoothing. | `silver2014deterministic`, `lillicrap2015continuous`, `fujimoto2018td3`; TD3 baseline. | Belongs in the interim report, but DPG/DDPG should be brief. |
| GRPO and group-relative optimization | Explain the origin and motivation for GRPO. | GRPO is introduced in DeepSeekMath as a PPO variant that estimates a baseline from grouped outputs instead of using a separate critic in the LLM setting. | `shao2024deepseekmath`; optionally `deepseekai2025deepseekr1`; Introduction final-goal framing. | Belongs in the interim report, but keep it scoped and cautious. |
| Tooling and benchmark environments | Briefly justify ObjectRL and Gymnasium as reproducibility infrastructure. | The baseline work depends on standardized environments and implementation frameworks; these support reproducible comparisons rather than being the main contribution. | `baykal2025objectrl`, `towers2024gymnasium`; scripts; notebook validation. | Belongs in the interim report if space allows. |
| Gap and project positioning | Close the section by tying the literature to this project. | Established baselines cover several major control-learning families, while GRPO-style group-relative optimization remains to be adapted and evaluated for online control. | Assignment PDF, Introduction, Methodology, Experiments. | Essential in the interim report. Expand in the final report after the GRPO-control method is fixed. |

For the compressed 5-page interim report, these can be merged into 4-5 paragraphs rather than separate subsections.

## 3. Paper-to-role map

| Paper/resource | Existing BibTeX key | What it supports | Where to cite | What NOT to claim |
|---|---|---|---|---|
| PPO: Schulman et al., *Proximal Policy Optimization Algorithms* | `schulman2017proximal` | PPO as an on-policy policy-gradient method using a clipped surrogate objective; practical baseline for policy optimization. | Paragraph on PPO/GAE; first mention of PPO baseline. | Do not claim PPO is generally best for control or that the project uses exactly every implementation detail from the paper. |
| GAE: Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | `schulman2015gae` | Advantage estimation as a bias-variance tradeoff; relevance of value-based advantage estimates in policy-gradient methods. | Paragraph connecting PPO, advantage functions, and later GRPO motivation. | Do not claim GAE is the only possible advantage estimator or that it solves all stability issues. |
| SAC: Haarnoja et al., *Soft Actor-Critic Algorithms and Applications* | `haarnoja2018sacapps` | SAC as an off-policy maximum-entropy actor-critic method for continuous control; entropy-regularized objective. | Paragraph on off-policy actor-critic baselines. | Do not claim SAC is always superior or that the midway runs prove SAC's general advantage. |
| TD3: Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods* | `fujimoto2018td3` | TD3 as an actor-critic method addressing overestimation using clipped double Q-learning, delayed policy updates, and target smoothing. | Paragraph on deterministic actor-critic baselines. | Do not claim TD3 removes all function-approximation error or always outperforms DDPG/PPO/SAC. |
| DPG: Silver et al., *Deterministic Policy Gradient Algorithms* | `silver2014deterministic` | The deterministic policy-gradient foundation for continuous-action actor-critic methods. | Brief background sentence before DDPG/TD3. | Do not spend much space on derivations; this belongs more in Theory if needed. |
| DDPG: Lillicrap et al., *Continuous Control with Deep Reinforcement Learning* | `lillicrap2015continuous` | Deep deterministic actor-critic learning for continuous actions; bridge from DPG to TD3. | Same deterministic actor-critic paragraph. | Do not claim the project evaluates DDPG; it only provides background. |
| GRPO / DeepSeekMath | `shao2024deepseekmath` | Source for GRPO; GRPO as a PPO variant that uses group-relative advantage estimation and avoids a separate critic in the LLM mathematical reasoning setting. | GRPO motivation paragraph and gap statement. | Do not claim GRPO is already proven for control, robotics, CarRacing, cartpole, or acrobot. |
| DeepSeek-R1 | `deepseekai2025deepseekr1` | Broader context for RL-based reasoning and GRPO-style training in LLMs. | Optional one-sentence context only, if space allows. | Do not overuse it; do not make the project about LLM reasoning rather than control. |
| ObjectRL | `baykal2025objectrl` | ObjectRL as an object-oriented RL codebase used for baseline implementations in the project. | Tooling/reproducibility paragraph; experiment implementation context. | Do not claim ObjectRL supports this exact CarRacing image-observation setup directly. |
| Gymnasium | `towers2024gymnasium` | Standardized environment API and interoperability for RL environments. | Tooling/environment paragraph. | Do not imply Gymnasium alone guarantees reproducibility or fair algorithm comparison. |

Optional background if space remains:

| Paper/resource | Existing BibTeX key | Role | Caution |
|---|---|---|---|
| TRPO | `schulman2015trpo` | Background for trust-region-style policy optimization and PPO motivation. | Mention only if needed; the interim section can skip details. |
| Conservative policy iteration | `kakade2002approximately` | Historical/theoretical background for conservative policy improvement. | Probably too detailed for the 5-page interim report. |
| RLHF / InstructGPT | `ouyang2022training` | Context for PPO use in language-model RLHF before GRPO. | Not necessary unless adding one sentence about LLM RL context. |
| DeepMind Control Suite | `tunyasuvunakool2020dmcontrol` | Environment background for cartpole/acrobot via DM Control. | More relevant to Methodology/Experiments than Related Work. |

## 4. Safe claims

The following claims are safe for the Related Work section:

- PPO, SAC, and TD3 are standard deep reinforcement learning baselines representing different design families.
- PPO is an on-policy policy-gradient method that uses a clipped surrogate objective to limit policy-update size.
- GAE is used to estimate advantages with a bias-variance tradeoff and is relevant background for policy-gradient methods.
- SAC is an off-policy actor-critic method based on the maximum-entropy reinforcement learning framework.
- TD3 belongs to the deterministic actor-critic family and addresses overestimation effects through mechanisms such as clipped double Q-learning, delayed policy updates, and target policy smoothing.
- DPG and DDPG provide background for deterministic continuous-control actor-critic methods.
- GRPO is introduced in DeepSeekMath as a PPO variant for language-model mathematical reasoning, using group-relative signals and reducing reliance on a separate critic/value model in that setting.
- DeepSeek-R1 can be mentioned as broader evidence that RL-based training has become important for reasoning models, but it is not the central source for the original GRPO method.
- ObjectRL is relevant as the implementation framework used for the vector-control PPO/SAC/TD3 baselines.
- Gymnasium is relevant as a standardized API for RL environments and interoperability.
- The project adapts the literature question toward control: can GRPO-style group-relative optimization be adapted and evaluated in online control benchmarks?
- At the midway stage, the report establishes the PPO/SAC/TD3 baseline comparison surface rather than presenting final GRPO-control results.

## 5. Claims to avoid

Avoid these claims in the Related Work section:

- Do not claim that GRPO is proven better than PPO, SAC, or TD3 for control.
- Do not claim that the current project has already implemented or evaluated the final GRPO-control method.
- Do not claim state-of-the-art performance.
- Do not claim that the midway PPO/SAC/TD3 results prove general algorithm superiority.
- Do not claim all algorithms have converged in the current experiments.
- Do not claim CarRacing was deferred; it is included through project-side CNN baselines.
- Do not claim PPO-CNN is not implemented; the current project-side code includes PPO-CNN, SAC-CNN, and TD3-CNN.
- Do not claim the notebook is dry-run only; it is the final report-facing midway notebook and validates the completed baseline matrix.
- Do not claim ObjectRL directly supports the image-observation CarRacing setup used here.
- Do not mechanically summarize every paper in isolation; the section should build a project argument.
- Do not use DeepSeek-R1 as if it were the original GRPO paper; cite DeepSeekMath for GRPO itself.
- Do not introduce new experimental results or implementation details in Related Work.

## 6. Danish draft

Nedenstående er et udkast til indhold og formulering. Det er bevidst lidt mere udførligt end den endelige 5-siders version bør være.

Reinforcement learning til kontrol bygger på en række etablerede metoder, som på forskellige måder forsøger at lære stabile politikker fra interaktion med et miljø. I dette projekt bruges PPO, SAC og TD3 som baselines, fordi de repræsenterer tre centrale familier af moderne deep reinforcement learning: on-policy policy-gradient metoder, off-policy maksimum-entropi actor-critic metoder og deterministiske actor-critic metoder til kontinuerte aktionsrum. Formålet med Related Work er derfor ikke blot at opsummere tre algoritmer, men at forklare hvorfor netop disse metoder udgør et relevant sammenligningsgrundlag for en senere GRPO-baseret kontrolmetode.

PPO er relevant som on-policy baseline, fordi metoden stabiliserer policy-gradient opdateringer gennem et klippet surrogate objective. I stedet for at lade den nye politik bevæge sig arbitrært langt væk fra den tidligere politik, begrænser PPO den effektive opdatering gennem et sandsynlighedsforhold mellem gammel og ny politik. Dette gør PPO praktisk anvendelig i mange dybe RL-opsætninger og forklarer, hvorfor metoden ofte bruges som en robust referencealgoritme. GAE er tæt forbundet med denne familie af metoder, fordi den giver en praktisk måde at estimere advantages på med en eksplicit bias-variance tradeoff. For dette projekt er dette vigtigt, fordi både PPO og GRPO afhænger af et policy-update signal, hvor advantage-estimation spiller en central rolle.

SAC repræsenterer en anden baseline-familie. Hvor PPO er on-policy, er SAC en off-policy actor-critic metode baseret på maksimum-entropi reinforcement learning. Metoden optimerer ikke kun forventet return, men også politikkens entropi, hvilket kan understøtte exploration og gøre læringen mindre afhængig af tidlige deterministiske valg. SAC er derfor et relevant sammenligningspunkt for kontinuerte kontrolopgaver, især fordi projektets environments kræver læring i aktionsrum, hvor exploration og stabil critic-læring kan have stor betydning.

TD3 repræsenterer den deterministiske actor-critic tradition. Denne linje kan spores tilbage til deterministic policy gradients og DDPG, hvor en deterministisk actor optimeres ved hjælp af en critic i kontinuerte aktionsrum. TD3 videreudvikler denne idé ved at adressere overestimation i actor-critic metoder gennem blandt andet twin critics, delayed policy updates og target policy smoothing. Dermed supplerer TD3 PPO og SAC som en baseline, der har et andet syn på politikrepræsentation og opdatering: en deterministisk actor i stedet for en stokastisk politik.

GRPO motiverer den endelige projektretning. I DeepSeekMath introduceres Group Relative Policy Optimization som en variant af PPO til sprogmodel-baseret matematisk ræsonnement. Den centrale idé er at estimere en relativ opdateringsretning ud fra grupper af outputs, hvor gruppens scores bruges som baseline, i stedet for at afhænge af en separat critic på samme måde som PPO i traditionelle actor-critic opsætninger. Denne idé er interessant for projektet, fordi den peger på en alternativ måde at konstruere policy-update signaler på. Samtidig er overførslen til kontrol ikke direkte: kontrolopgaver involverer sekventiel interaktion, tidslig credit assignment, kontinuerte eller strukturerede aktionsrum og læring fra trajectories snarere end uafhængige tekstlige svar.

Reproducerbar eksperimentel infrastruktur er også en del af den relevante baggrund. ObjectRL bruges i projektet som baseline-implementering for de vektorbaserede kontrolmiljøer, mens Gymnasium giver en standardiseret API til reinforcement learning environments. I den aktuelle midway-opsætning kræver CarRacing en separat projekt-side CNN-implementering, fordi billedobservationerne ikke passer direkte ind i den ObjectRL-baserede vector-observation pipeline, der bruges til cartpole-swingup og acrobot-swingup. Dette er ikke en ny algoritmisk contribution, men det er vigtigt for at etablere et sammenligneligt og reproducerbart baselinegrundlag.

Samlet set placerer litteraturen projektet mellem etablerede kontrolbaselines og nyere group-relative policy optimization fra sprogmodeldomænet. PPO, SAC og TD3 dækker centrale referencepunkter for kontrol, mens GRPO motiverer en mulig alternativ policy-optimization strategi. Det åbne spørgsmål for det endelige projekt er, om en GRPO-lignende, gruppe-relativ opdateringsidé kan formuleres og evalueres meningsfuldt i online kontrolbenchmarks. I interimrapporten besvares dette spørgsmål endnu ikke; rapporten etablerer i stedet det notationelle, metodiske og eksperimentelle sammenligningsgrundlag, som den endelige GRPO-control evaluering skal bygge på.

## 7. English academic draft

The following draft is report-facing, but it should still be compressed before being inserted into the final interim LaTeX section.

Reinforcement learning for control is supported by several mature algorithmic families that make different trade-offs between stability, sample reuse, exploration, and policy representation. This project uses PPO, SAC, and TD3 as baselines because they represent complementary approaches to deep reinforcement learning for control: on-policy policy-gradient optimization, off-policy maximum-entropy actor-critic learning, and deterministic actor-critic learning for continuous action spaces. The role of the related work is therefore not to survey reinforcement learning broadly, but to motivate why these methods form an appropriate comparison surface for the later GRPO-control method.

PPO is included as the on-policy baseline because it stabilizes policy-gradient updates through a clipped surrogate objective \citep{schulman2017proximal}. Rather than allowing an unconstrained policy-gradient step, PPO limits the effect of large changes in the probability ratio between the new and old policies. This makes PPO a practical reference method for trajectory-based policy optimization. Generalized Advantage Estimation is relevant background because it estimates advantages through a bias-variance trade-off and is commonly associated with policy-gradient and actor-critic implementations \citep{schulman2015gae}. For this project, the PPO/GAE connection is important because the later GRPO motivation also concerns how policy-update signals are constructed from sampled experience.

SAC provides a contrasting off-policy baseline. It is an actor-critic method based on the maximum-entropy reinforcement learning framework, where the policy is trained to maximize expected return while also maintaining entropy \citep{haarnoja2018sacapps}. This makes SAC relevant for continuous-control tasks in which exploration and sample reuse are important. TD3 represents a different continuous-control baseline from the deterministic policy-gradient family. Deterministic policy gradients provide the theoretical basis for learning deterministic policies in continuous action spaces \citep{silver2014deterministic}, and DDPG applies this idea with deep function approximation \citep{lillicrap2015continuous}. TD3 extends this line by addressing overestimation effects in actor-critic learning using clipped double Q-learning, delayed policy updates, and target policy smoothing \citep{fujimoto2018td3}. Together, PPO, SAC, and TD3 cover distinct and widely used baseline families.

GRPO motivates the final project stage. In DeepSeekMath, Group Relative Policy Optimization is introduced as a PPO variant for language-model mathematical reasoning \citep{shao2024deepseekmath}. Its key idea is to estimate the update signal from relative scores within a group of sampled outputs, thereby reducing reliance on a separate critic in that setting. This is attractive for the present project because it suggests an alternative way to construct policy-optimization signals. However, the transfer from language-model reinforcement learning to control is not direct. Online control tasks require temporal credit assignment, interaction with environment dynamics, continuous or structured action spaces, and stable learning from trajectories. A GRPO-control method therefore needs its own formulation and evaluation rather than a direct reuse of the language-model algorithm.

The experimental infrastructure is also part of the related context. ObjectRL is used as the implementation framework for the vector-control PPO, SAC, and TD3 baselines \citep{baykal2025objectrl}, while Gymnasium provides a standardized interface for reinforcement learning environments \citep{towers2024gymnasium}. In this project, CarRacing is handled by project-side CNN implementations because the image-observation setup does not directly match the ObjectRL vector-observation workflow used for the other environments. This tooling distinction supports the reproducibility goal of the interim report, but it is not the final algorithmic contribution.

The resulting gap is that established control baselines are well defined, while GRPO-style group-relative policy optimization has primarily been motivated in language-model reasoning contexts. Whether this idea can be adapted into an effective online control method remains the central question for the final project stage. At the midway stage, this report therefore establishes the related work, notation, implementation paths, and completed baseline matrix needed for a later GRPO-control comparison, without claiming that the final GRPO method has already been evaluated.

## 8. Suggested citation placement

Use real BibTeX keys from `report/references.bib`:

- First sentence introducing PPO as the on-policy baseline: `\citep{schulman2017proximal}`.
- Sentence explaining GAE as advantage estimation with a bias-variance tradeoff: `\citep{schulman2015gae}`.
- Sentence introducing SAC as off-policy maximum-entropy actor-critic: `\citep{haarnoja2018sacapps}`.
- Sentence introducing deterministic policy-gradient background: `\citep{silver2014deterministic}`.
- Sentence introducing DDPG as deep deterministic actor-critic background: `\citep{lillicrap2015continuous}`.
- Sentence explaining TD3 mechanisms: `\citep{fujimoto2018td3}`.
- First sentence introducing GRPO from DeepSeekMath: `\citep{shao2024deepseekmath}`.
- Optional one-sentence broader LLM reasoning context: `\citep{deepseekai2025deepseekr1}`. Use only if space permits.
- Sentence stating ObjectRL is the vector-control baseline implementation framework: `\citep{baykal2025objectrl}`.
- Sentence stating Gymnasium provides a standardized RL environment interface: `\citep{towers2024gymnasium}`.

Avoid over-citing every sentence. In the compressed interim report, one citation cluster per paragraph may be enough:

- PPO/GAE paragraph: `\citep{schulman2017proximal,schulman2015gae}`.
- SAC/TD3 paragraph: `\citep{haarnoja2018sacapps,silver2014deterministic,lillicrap2015continuous,fujimoto2018td3}`.
- GRPO paragraph: `\citep{shao2024deepseekmath}`.
- Tooling paragraph: `\citep{baykal2025objectrl,towers2024gymnasium}`.

Important: the current `03_methodology.tex` appears to cite `haarnoja2018soft` and `fujimoto2018addressing`, but the existing keys in `report/references.bib` are `haarnoja2018sacapps` and `fujimoto2018td3`. If final LaTeX is edited later, align citation keys consistently.

## 9. Gap statement variants

### Variant 1: concise and report-facing

Although GRPO has been introduced as an efficient PPO-style method for language-model mathematical reasoning, its direct transfer to online control is non-trivial. The final project therefore investigates whether a group-relative policy-optimization signal can be adapted to control benchmarks and evaluated against PPO, SAC, and TD3 under the same experimental protocol.

### Variant 2: emphasizes methodology

The literature provides strong baseline methods for control and a recent group-relative optimization idea from language-model RL, but it does not by itself specify how GRPO should be formulated for continuous-control trajectories. The interim report addresses the prerequisite step: establishing notation, implementation paths, and a validated PPO/SAC/TD3 baseline matrix before the GRPO-control variant is evaluated.

### Variant 3: cautious and assignment-aligned

GRPO motivates the final project because it replaces conventional value-based advantage estimation with group-relative signals in the language-model setting. Whether this idea can be made effective for control tasks involving environment dynamics, temporal credit assignment, and continuous actions remains open in this project. The completed midway baselines define the comparison surface for answering that question in the final report.

## 10. Suggested final target length

For the current 5-page interim draft, the final `02_related_work.tex` should be compact:

- **Target paragraphs:** 4-6 paragraphs.
- **Approximate length:** 0.5 to 0.75 pages in the NeurIPS-style template.
- **Suggested compression:**
  1. Paragraph 1: baseline families and purpose of related work.
  2. Paragraph 2: PPO and GAE.
  3. Paragraph 3: SAC and TD3, with DPG/DDPG folded into one sentence.
  4. Paragraph 4: GRPO and why transfer to control is non-trivial.
  5. Optional paragraph 5: ObjectRL/Gymnasium tooling and reproducibility.
  6. Final sentence or paragraph: gap and project positioning.

If page space becomes tight, merge the tooling paragraph into the final gap paragraph and keep DeepSeek-R1 out of the main text.

## 11. Final author notes

Decide manually:

- Whether to include DeepSeek-R1 at all. It is useful context for reasoning RL, but DeepSeekMath is the core GRPO citation.
- Whether to include ObjectRL/Gymnasium in Related Work or leave most tooling discussion to Methodology/Experiments.
- Whether to mention TRPO/CPI. They are useful background for PPO, but probably too detailed for the interim 5-page report.
- How much Danish phrasing to preserve when converting to the final English LaTeX section.

Check against the assignment PDF:

- The final assignment expects GRPO learning curves and comparison against PPO, SAC, and TD3, but the interim report is currently evaluated on Related Work, MDP notation, and complete PPO/SAC/TD3 baseline results.
- The assignment wording is ambitious about GRPO performance. The interim report should not promise that this has already been achieved.
- The final report should later justify how the GRPO-control variant is designed and why its results support the project hypothesis.

Align later with Methodology, Theory, and Experiments:

- Methodology should define the MDP, policies, trajectories, returns, value functions, and advantages used by the Related Work discussion.
- Theory should contain any heavier derivations; Related Work should not prove policy-gradient or GRPO properties.
- Experiments should present the validated 45-run baseline matrix and learning curves; Related Work should only motivate why these baselines are relevant.
- Use the same citation keys consistently across all sections: `haarnoja2018sacapps` and `fujimoto2018td3`, not non-existing alternatives.

Postpone until the final report:

- Detailed GRPO-control algorithm design.
- Pseudocode for the GRPO-control update.
- Claims about GRPO-control performance.
- Convergence or stability analysis of the proposed GRPO variant.
- Final comparison between GRPO, PPO, SAC, and TD3.
- Any stronger claim that GRPO is suitable for robotics/control beyond what the final experiments support.
