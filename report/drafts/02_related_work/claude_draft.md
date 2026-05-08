# Related Work — Inspiration Draft (claude)

This document is **inspiration material** for the Related Work section of
the DM887 *interim/midway* report. It is not final report text. The final
wording lives in `report/sections/02_related_work.tex` and must be written
manually. This draft may be slightly long; you will compress it into the
final 5-page interim report.

Project: *Group Relative Policy Optimization for Control* (DM887,
Spring 2026). Stage: midway / interim report.

---

## 1. Related Work section goal analysis

In this **interim** report the Related Work section has to do five jobs at
once, and it has to do them without becoming a list of paper summaries:

1. **Justify the baseline triple PPO / SAC / TD3.** Explain *why* these
   three algorithms are the right comparison surface for a control project.
   They are not arbitrary — they cover three orthogonal design axes
   (on-policy vs. off-policy, stochastic vs. deterministic, policy-gradient
   vs. value-based actor-critic) and they are the algorithms shipped by the
   ObjectRL framework used in the project's vector-control runner.
2. **Lay out the policy-optimization lineage GRPO sits in.** Trace
   policy-gradient → trust-region/clipped surrogate (TRPO/PPO with GAE) →
   group-relative advantages (GRPO). The point is not to retell the
   theory; it is to show that GRPO is a sibling of PPO and inherits its
   policy-update geometry while replacing the value-baseline component of
   the advantage with a group-relative quantity.
3. **Explain the LLM/reasoning origin of GRPO honestly.** GRPO was
   introduced in `DeepSeekMath` for math-reasoning RL on language models
   and was later used as the optimization scaffold for `DeepSeek-R1`. This
   is the *known* setting; the project's question is whether a GRPO-style
   group-relative scheme transfers to *online control benchmarks*.
4. **Position the tooling.** ObjectRL (vector-control baselines) and
   Gymnasium (`CarRacing-v3`, plus the standard environment interface)
   are the two reproducibility anchors that the experiment matrix sits on.
5. **State the gap clearly.** GRPO-style group-relative policy optimization
   is well-motivated for LLM/reasoning RL but its adaptation and
   evaluation for online continuous-control benchmarks is open. The
   project's final stage targets that open question; the interim report
   delivers the baseline foundation needed to evaluate it.

What this section should explicitly **not** do at the interim stage:

- It must not include theorems or proofs — those belong in Theory.
- It must not include the GRPO-control algorithm — it does not exist yet.
- It must not duplicate the MDP notation block from Methodology
  (`report/sections/03_methodology.tex`).
- It must not narrate experimental results — those are in Experiments
  (`report/sections/05_experiments.tex`).
- It must not become a textbook chapter on RL.

Length target for the **interim** version: approximately one full page in
the NeurIPS template, see Section 10 of this draft.

---

## 2. Proposed subsection structure (5 subsections)

A 5-subsection structure fits the interim report. If page budget is
tight, fold 2.1 into 2.2 — but keep the gap statement (2.5) as its own
short paragraph.

### 2.1 Policy-gradient methods and advantage estimation
- **Purpose.** Set the lineage that PPO and (eventually) GRPO sit in.
- **Key message.** The policy-gradient theorem motivates updating
  $\theta$ in the direction $\mathbb{E}^{\pi_\theta}[\nabla_\theta \log
  \pi_\theta(a|s)\, A^{\pi_\theta}(s,a)]$. Variance-reduced advantage
  estimation, in particular GAE, is what makes deep policy-gradient
  methods practical.
- **Connects to.** `papers/gae_schulman_2015.pdf`, the methodology
  notation (`\ref{sec:methodology-policies-returns}`), the eventual
  GRPO discussion (group-relative *advantages*).
- **Interim / final.** Interim. Same content reused in the final report.

### 2.2 PPO and trust-region / clipped policy updates
- **Purpose.** Explain the policy-update geometry that PPO uses and that
  GRPO inherits.
- **Key message.** TRPO showed that constraining policy updates by KL
  divergence yields monotonic-improvement guarantees in idealised
  settings; PPO is the practical first-order surrogate that uses a
  clipped probability-ratio objective to keep updates inside a trust
  region. The clipped-surrogate objective is the same object that GRPO
  later modifies by replacing the value baseline with a group-relative
  estimate.
- **Connects to.** `papers/ppo_schulman_2017.pdf`,
  `papers/trpo_schulman_2015.pdf` (background only; light-touch),
  `papers/kakade_langford_2002.pdf` (background only).
- **Interim / final.** Interim. The final report can extend this if
  GRPO-control derivations require more TRPO-style background.

### 2.3 Off-policy actor-critic methods: SAC, DPG, DDPG, TD3
- **Purpose.** Describe the off-policy actor-critic family that SAC and
  TD3 belong to, and motivate them as continuous-control baselines.
- **Key message.** SAC pairs an off-policy actor with a maximum-entropy
  objective and twin soft Q-critics; this gives stable continuous-control
  learning with explicit exploration via entropy regularization. The DPG
  → DDPG → TD3 line is a deterministic-policy lineage: DPG gives the
  deterministic policy gradient theorem, DDPG turns it into a deep
  off-policy method, and TD3 fixes function-approximation overestimation
  bias via clipped double Q-learning, delayed policy updates, and target
  policy smoothing. Both SAC and TD3 are continuous-control workhorses
  and complement the on-policy PPO.
- **Connects to.** `papers/sac_haarnoja_2018.pdf`,
  `papers/td3_fujimoto_2018.pdf`, `papers/ddpg_lillicrap_2015.pdf`,
  `papers/dpg_silver_2014.pdf`, plus the methodology classification of
  PPO/SAC as stochastic-policy methods and TD3 as a deterministic-policy
  method (`\ref{sec:methodology-baselines}`).
- **Interim / final.** Interim. Compress aggressively if length is tight.

### 2.4 Group-relative policy optimization (GRPO)
- **Purpose.** Introduce GRPO as a PPO-family variant that drops the
  learned value baseline in favour of a group-relative one.
- **Key message.** GRPO was introduced in `DeepSeekMath` as the
  optimization recipe for math-reasoning RL on language models, and was
  used as the RL scaffold for `DeepSeek-R1`'s reasoning training. The
  central modification relative to PPO is that the per-sample advantage
  is computed *within a sampled group* of $G$ responses to the same
  prompt (state) — typically by mean-/std-normalizing the per-sample
  rewards within the group — rather than against a learned value
  function. The clipped-surrogate update structure inherited from PPO is
  retained. This avoids training a separate value model in language-model
  RL; whether the same trick is *helpful*, *neutral*, or *harmful* in
  online control with sequential rewards and continuous action spaces is
  not yet established.
- **Connects to.** `papers/deepseekmath_grpo_2024.pdf`,
  `papers/deepseek_r1_2025.pdf` (light-touch context only),
  Section 2.2 (PPO surrogate), Section 2.5 (gap).
- **Interim / final.** Interim. Will be expanded in the final report
  once the GRPO-control variant is fixed.

### 2.5 Tooling and reproducibility: ObjectRL and Gymnasium
- **Purpose.** Position the experimental infrastructure within the wider
  tooling landscape and motivate the project's choice of frameworks.
- **Key message.** ObjectRL is an object-oriented RL codebase that
  ships PPO/SAC/TD3 implementations on a shared training/evaluation
  loop, which makes it a natural fit for the project's vector-control
  baselines. Gymnasium provides the canonical environment interface used
  by `CarRacing-v3`. The project deliberately reuses these frameworks
  for the baselines so that the experimental contribution sits at the
  protocol/measurement level, not at the level of re-implementing
  standard algorithms. Where Gymnasium-style image observations are
  incompatible with ObjectRL's 1-D `Box` assumption, the project ships
  a small CNN-based runner; this is an implementation choice, not a gap
  in the algorithmic literature, and is documented in Methodology.
- **Connects to.** `papers/objectrl_baykal_2025.pdf`,
  `papers/gymnasium_towers_2024.pdf`,
  `scripts/run_project_objectrl_baseline.py`,
  `scripts/run_carracing_cnn_baseline.py`,
  `scripts/carracing_cnn.py`,
  `\ref{sec:methodology-implementation}`.
- **Interim / final.** Interim. The final report can drop this entirely
  or compress it to a sentence if space is at a premium.

### 2.6 Gap and project positioning
- **Purpose.** State the open question and what this project contributes
  toward closing it.
- **Key message.** GRPO and its successors have established
  group-relative policy optimization as a competitive alternative to
  value-baseline PPO in the LLM/reasoning setting; PPO, SAC, and TD3
  remain the canonical baselines for continuous control. The transfer
  of group-relative advantage estimation to online control benchmarks
  is an open question, and is the goal of the final project stage. At
  the interim stage the project contributes the reproducible PPO/SAC/TD3
  baseline matrix on the assignment's three control tasks, which
  defines the comparison surface against which a GRPO-control variant
  will be measured.
- **Connects to.** `report/sections/01_introduction.tex` (project aim),
  `\ref{sec:methodology-grpo-preparation}`,
  `\ref{sec:transition-final-grpo}`.
- **Interim / final.** Interim. The final report rewrites this with
  the GRPO-control results in hand.

---

## 3. Paper-to-role map

For each paper, the table lists the BibTeX key (verified against
`report/references.bib`), the role in the report, where to cite it, and
what *not* to claim from it.

| Paper | BibTeX key (verified) | Role in report | Where to cite | What NOT to claim |
|---|---|---|---|---|
| PPO (Schulman et al., 2017) | `schulman2017proximal` | Defines the on-policy clipped-surrogate baseline used by the project; supplies the policy-update geometry that GRPO inherits. | First mention in 2.2 ("PPO" / "clipped surrogate objective"); Methodology already cites it for the baseline triple. | Do not claim PPO is monotonic-improvement (only TRPO is, in idealised form); do not claim PPO is best-in-class for any specific task. |
| GAE (Schulman et al., 2015) | `schulman2015gae` | Variance-reduced advantage estimator that PPO and similar methods use. | First mention in 2.1, alongside the policy-gradient lineage. | Do not claim GAE removes bias entirely (it is a bias-variance tradeoff parameterised by $\lambda$). |
| TRPO (Schulman et al., 2015) | `schulman2015trpo` | Theoretical predecessor / motivation for PPO's clipped surrogate. Optional. | Light-touch in 2.2 only if needed to motivate the trust-region framing. | Do not present TRPO as a project baseline; it is not used in the experiments. |
| Kakade & Langford (2002) | `kakade2002approximately` | Conservative-policy-iteration predecessor for monotonic-improvement-style results. Optional. | Optional one-cite background in 2.2; can be omitted in the interim. | Do not include a derivation; this is background, not a project claim. |
| SAC (Haarnoja et al., 2018) | `haarnoja2018sacapps` | The off-policy maximum-entropy actor-critic baseline used by the project. | First mention in 2.3. | Do not claim SAC is "best on continuous control"; do not paraphrase the entropy temperature schedule unless needed. |
| TD3 (Fujimoto et al., 2018) | `fujimoto2018td3` | The deterministic-policy off-policy baseline used by the project; supplies overestimation-mitigation tools (clipped double Q, delayed updates, target smoothing). | First mention in 2.3. | Do not claim TD3 always outperforms DDPG; do not present it as "TD3 = DDPG + tricks" without nuance. |
| DDPG (Lillicrap et al., 2015) | `lillicrap2015continuous` | Background for the deterministic actor-critic line. Cited as predecessor to TD3, not as a project baseline. | One sentence in 2.3 lineage. | Do not present DDPG as a project baseline; the project does not use it directly. |
| DPG (Silver et al., 2014) | `silver2014deterministic` | Theoretical foundation for deterministic policy gradients (used by DDPG/TD3). | One sentence in 2.3 lineage. | Same as DDPG: not a project baseline. |
| GRPO / DeepSeekMath (Shao et al., 2024) | `shao2024deepseekmath` | Source paper for GRPO; explains the group-relative advantage construction and PPO inheritance. | Primary GRPO citation in 2.4 and the gap statement in 2.6. | Do not claim GRPO has been validated on control benchmarks; do not claim a specific transfer guarantee. |
| DeepSeek-R1 (DeepSeek-AI, 2025) | `deepseekai2025deepseekr1` | Course-provided context paper showing GRPO used as the RL scaffold for reasoning training. Light-touch only. | Optional second cite in 2.4 to anchor "GRPO has been used at scale for reasoning RL"; cite once or omit. | Do not summarise R1's training pipeline; do not claim R1 evidence about *control*. |
| InstructGPT / RLHF (Ouyang et al., 2022) | `ouyang2022training` | Background for PPO-based RLHF, useful only if the report explicitly contextualises why GRPO emerged inside the LLM-RL pipeline. | Optional, in 2.4 only if you want to show the broader LLM-RL trajectory. Otherwise omit. | Do not present RLHF as a project baseline; the project is not RLHF. |
| ObjectRL (Baykal et al., 2025) | `baykal2025objectrl` | The implementation source for the vector-control PPO/SAC/TD3 baselines. | First mention in 2.5 (and Methodology already cites it). | Do not claim that ObjectRL is the "best" RL framework; do not claim it natively supports image observations for the project's setup. |
| Gymnasium (Towers et al., 2024) | `towers2024gymnasium` | The standard environment interface; CarRacing-v3 is loaded through `gymnasium.make`. | First mention in 2.5; Methodology can also cite it where CarRacing is described. | Do not claim Gymnasium provides the swing-up tasks (those come from DM Control). |
| DM Control (Tunyasuvunakool et al., 2020) | `tunyasuvunakool2020dmcontrol` | Optional. Source of the swing-up tasks (`cartpole_swingup`, `acrobot_swingup`) loaded by `scripts/project_envs.py` via `dm_control.suite.load`. | Optional in 2.5 if the section explicitly attributes the swing-up tasks. | Do not claim DM Control was used as the runner; the project uses ObjectRL on top of a DM Control / Gymnasium adapter. |

If you choose **not** to cite TRPO, Kakade–Langford, RLHF/InstructGPT, or
DM Control in the interim Related Work, that is fine — the section is
already supported by the eight core citations.

---

## 4. Safe claims

These claims are safe in the Related Work section.

**On PPO / GAE**

- PPO is a first-order on-policy policy-gradient method that constrains
  policy updates using a clipped probability-ratio surrogate
  (`schulman2017proximal`).
- GAE provides a bias-variance-controlled advantage estimator that is
  commonly paired with PPO (`schulman2015gae`).
- TRPO (`schulman2015trpo`) is an earlier trust-region method that
  motivates the clipped-surrogate framing of PPO, but the project does
  not use TRPO directly.

**On SAC**

- SAC is an off-policy actor-critic method that adds an entropy bonus to
  the reward, which yields a maximum-entropy objective and explicit
  exploration regularisation (`haarnoja2018sacapps`).
- SAC has been widely used as a strong continuous-control baseline.

**On the deterministic actor-critic lineage**

- DPG (`silver2014deterministic`) provides the deterministic policy
  gradient theorem.
- DDPG (`lillicrap2015continuous`) extended it to a deep off-policy
  setting using replay buffers and target networks.
- TD3 (`fujimoto2018td3`) addresses function-approximation overestimation
  in continuous control via clipped double Q-learning, delayed policy
  updates, and target policy smoothing.

**On GRPO**

- GRPO was introduced in `DeepSeekMath` (`shao2024deepseekmath`) as a
  PPO-family policy-optimization method that replaces the per-sample
  value-baseline component of the advantage with a group-relative
  quantity computed across a sampled group of responses to the same
  prompt.
- GRPO retains a clipped-surrogate update structure inherited from PPO.
- GRPO was subsequently used as the RL scaffold for DeepSeek-R1's
  reasoning-RL training (`deepseekai2025deepseekr1`); the existing
  evidence about GRPO is concentrated in language-model / reasoning
  settings.

**On tooling**

- ObjectRL (`baykal2025objectrl`) provides PPO, SAC, and TD3
  implementations on a shared training/evaluation loop, which the
  project reuses for the vector-control baselines.
- Gymnasium (`towers2024gymnasium`) provides the standard environment
  interface used to construct `CarRacing-v3`.
- The project does not modify any code under `external/objectrl/`; the
  vector-control runner uses a project-side bridge.

**On the project's positioning**

- The project's contribution at the interim stage is a reproducible
  PPO/SAC/TD3 baseline matrix; the GRPO-control variant is planned for
  the final stage.
- The midway baselines define the comparison surface for the eventual
  GRPO-control evaluation; they are not intended to settle algorithmic
  superiority in general.

---

## 5. Claims to avoid

Do **not** write any of the following in Related Work:

- "GRPO is known to outperform PPO on control."
  (No public evidence on online control benchmarks; the GRPO evidence
  is in LLM/reasoning settings.)
- "PPO/SAC/TD3 have been definitively ranked on these tasks."
  (The interim baselines do not justify this; no HPO was performed.)
- "We have implemented and evaluated GRPO."
  (Not at the interim stage.)
- "CarRacing was deferred." / "PPO-CNN was not implemented."
  (Both are implemented; CarRacing is fully in the matrix.)
- "ObjectRL natively supports image observations."
  (`objectrl.agents.base_agent` asserts 1-D Box observations; CarRacing
  uses the project-side CNN runner.)
- "Our method achieves state-of-the-art performance."
  (No method has even been proposed yet at the interim stage.)
- "DDPG is a project baseline."
  (It is not; only PPO, SAC, and TD3 are.)
- "RLHF is a project baseline."
  (It is not; cite InstructGPT only as background, if at all.)
- "The midway notebook is a dry-run only."
  (It validates a complete 45-run baseline matrix.)
- Any claim that paraphrases a paper's results in a way the paper does
  not actually support (e.g., "TD3 always beats DDPG"; "SAC is the best
  off-policy method").
- Any sentence that introduces a new theorem, proof, or algorithm in
  Related Work — those belong in Theory or in the final-stage method
  section.

---

## 6. Danish draft

> **Bemærk.** Akademisk sprog, men læsbart. Brug dette som udgangspunkt
> og redigér selv. Citationer er sat med `\citep{...}` mod nøgler, der
> allerede findes i `report/references.bib`. Udkastet er bevidst lidt
> langt; komprimer det selv til den endelige version.

**2 Relateret arbejde**

**2.1 Politik-gradient-metoder og fordels-estimering.**
Klassiske politik-gradient-metoder opdaterer politikkens parametre
$\theta$ ved at følge gradienten af den forventede return,
typisk i retningen
$\mathbb{E}^{\pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s)\,
A^{\pi_\theta}(s, a)]$. I dyb forstærkningslæring er det imidlertid
nødvendigt at reducere variansen af denne estimator for at gøre den
praktisk anvendelig. *Generalized Advantage Estimation* (GAE)
\citep{schulman2015gae} introducerer en parametrisk bias–varians-
afvejning kontrolleret af $\lambda$, og er i praksis blevet
standardmetoden til fordels-estimering for on-policy policy-gradient-
opdateringer. Denne notation og denne afvejning genfindes både i PPO
og i den senere GRPO-metode.

**2.2 PPO og trust-region/clippet politik-opdatering.**
*Trust Region Policy Optimization* (TRPO) \citep{schulman2015trpo}
viste, at det er muligt at give monotone forbedringsgarantier for
politik-opdateringer ved at begrænse KL-divergensen mellem den nye og
den gamle politik. *Proximal Policy Optimization* (PPO)
\citep{schulman2017proximal} er den førsteordens-erstatning, der bruger
en clippet sandsynligheds-ratio som surrogatmål for at holde
opdateringer inden for en effektiv trust-region uden at løse en
begrænset optimering. Den clippede surrogatobjektiv er det objekt, som
GRPO senere modificerer ved at erstatte værdi-baselinjen i fordels-
estimatet med en gruppe-relativ størrelse. PPO indgår i dette projekt
som on-policy stokastisk baseline.

**2.3 Off-policy actor-critic-metoder: SAC, DPG, DDPG og TD3.**
*Soft Actor-Critic* (SAC) \citep{haarnoja2018sacapps} er en off-policy
actor-critic-metode, der tilføjer et entropi-led til belønningen og
dermed maksimerer en maksimal-entropi-mål. SAC bruger to soft Q-funktioner
og en stokastisk politik og opnår stabil indlæring i kontinuerlige
kontrol-opgaver. Den deterministiske actor-critic-linje begynder med
*deterministic policy gradient*-sætningen \citep{silver2014deterministic};
*Deep DPG* (DDPG) \citep{lillicrap2015continuous} udvider den til en
dyb off-policy-metode med replay-buffer og målnetværk. *Twin Delayed
DDPG* (TD3) \citep{fujimoto2018td3} adresserer overestimering i
funktionsapproksimation gennem clippet dobbelt-Q-læring, forsinkede
politik-opdateringer og udjævning af målpolitikken. SAC og TD3 indgår
begge i projektets baseline-matrix; DDPG og DPG indgår alene som
baggrund for TD3.

**2.4 Group Relative Policy Optimization (GRPO).**
GRPO blev introduceret som en del af *DeepSeekMath*
\citep{shao2024deepseekmath} og blev senere brugt som
optimeringsrammeværk i træningen af *DeepSeek-R1*
\citep{deepseekai2025deepseekr1}. Sammenlignet med PPO bevarer GRPO
det clippede surrogatmål, men erstatter den lærte værdi-baseline i
fordels-estimatet med en *gruppe-relativ* størrelse, der beregnes
inden for en sampleret gruppe af outputs til den samme tilstand
(typisk samme prompt i sprogmodel-konteksten). Dette fjerner behovet
for en separat værdimodel i den oprindelige LLM-/reasoning-anvendelse.
Den eksisterende evidens for GRPO er imidlertid centreret om sprogmodel-
og reasoning-RL; transferen til online kontrolopgaver med sekventielle
belønninger og kontinuerlige handlingsrum er ikke etableret og er det,
som det endelige projektstadie undersøger.

**2.5 Værktøjer og benchmark-miljøer: ObjectRL og Gymnasium.**
Projektets vektor-kontrol-baselines udnytter PPO-, SAC- og TD3-
implementeringerne fra *ObjectRL* \citep{baykal2025objectrl}, en
objekt-orienteret RL-kodebase med en delt trænings- og evaluerings-løkke
for de tre algoritmer. Dette valg betyder, at projektets bidrag på
dette stadie ligger i forsøgsprotokol og resultatvalidering frem for
i en reimplementering af standardalgoritmerne. *Gymnasium*
\citep{towers2024gymnasium} leverer den standardiserede miljø-
interface, som bruges til at konstruere `CarRacing-v3`. Da
ObjectRL’s standard arkitekturer antager én-dimensionelle vektor-
observationer, og CarRacing returnerer billed-observationer, leveres
PPO-, SAC- og TD3-CNN-varianter af projekt-egne agenter; dette er en
implementeringsdetalje og en del af metoden snarere end et nyt
algoritmisk bidrag.

**2.6 Gap og projektets positionering.**
GRPO og dens efterkommere har etableret gruppe-relativ politik-
optimering som en konkurrencedygtig alternativ til PPO-baseret
RLHF i sprogmodel- og reasoning-konteksten, mens PPO, SAC og TD3
forbliver de kanoniske baselines for kontinuerlig kontrol.
Overførslen af gruppe-relativ fordels-estimering til online
kontrolbenchmarks er imidlertid ikke etableret. Det endelige
projektstadie sigter mod at adressere dette spørgsmål. På
midtvejsstadiet leverer dette projekt det reproducerbare PPO-/SAC-/TD3-
baseline-matrix på opgavens tre kontrolmiljøer, som vil definere den
sammenligningsflade, som en GRPO-control-variant senere skal måles
imod.

---

## 7. English academic draft

> **Note.** Concise, report-facing. Citation placeholders use real
> `\citep{...}` keys verified against `report/references.bib`. Section
> labels match the existing `03_methodology.tex` and
> `05_experiments.tex`.

**2 Related Work**

**2.1 Policy-gradient methods and advantage estimation.**
Policy-gradient methods update policy parameters $\theta$ in the
direction of the gradient of expected return, typically along
$\mathbb{E}^{\pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s)\,
A^{\pi_\theta}(s, a)]$. In deep reinforcement learning, the practical
question is how to estimate $A^{\pi_\theta}$ with controllable bias
and variance. *Generalized Advantage Estimation* (GAE)
\citep{schulman2015gae} introduces a $\lambda$-parameterised
bias–variance tradeoff for advantage estimation that has become a
de-facto standard in on-policy policy-gradient training. The same
advantage object reappears in PPO and, in modified form, in GRPO.

**2.2 PPO and trust-region / clipped policy updates.**
*Trust Region Policy Optimization* (TRPO) \citep{schulman2015trpo}
showed that constraining policy updates by KL divergence yields
monotonic-improvement properties in idealised settings. *Proximal
Policy Optimization* (PPO) \citep{schulman2017proximal} is the
first-order surrogate that replaces the explicit constraint with a
clipped probability-ratio objective, keeping updates inside an
effective trust region without solving a constrained optimisation
problem. The clipped surrogate of PPO is precisely the objective that
GRPO later modifies by replacing the learned value baseline with a
group-relative quantity. PPO is included in this project as the
on-policy stochastic baseline.

**2.3 Off-policy actor-critic methods: SAC, DPG, DDPG, TD3.**
*Soft Actor-Critic* (SAC) \citep{haarnoja2018sacapps} is an off-policy
actor-critic method that adds an entropy bonus to the reward, yielding
a maximum-entropy objective and explicit exploration regularisation,
and uses twin soft Q-functions with a stochastic actor. The
deterministic actor-critic line starts with the deterministic policy
gradient theorem \citep{silver2014deterministic}; *Deep DPG* (DDPG)
\citep{lillicrap2015continuous} extends it to a deep off-policy
setting using replay buffers and target networks. *Twin Delayed
DDPG* (TD3) \citep{fujimoto2018td3} addresses function-approximation
overestimation through clipped double Q-learning, delayed policy
updates, and target policy smoothing. SAC and TD3 are both included
in this project's baseline matrix; DPG and DDPG are mentioned as
background for TD3 rather than as project baselines.

**2.4 Group Relative Policy Optimization (GRPO).**
GRPO was introduced as part of *DeepSeekMath*
\citep{shao2024deepseekmath} and was subsequently used as the
optimisation framework for the reasoning training of *DeepSeek-R1*
\citep{deepseekai2025deepseekr1}. Relative to PPO, GRPO retains the
clipped-surrogate update but replaces the learned value baseline in the
advantage estimate with a *group-relative* quantity computed within a
sampled group of outputs to the same input — typically by mean- and
std-normalising the per-sample rewards within the group. In its
original setting this removes the need for a separate value model
during reinforcement learning of language models. The existing GRPO
evidence is, however, concentrated in language-model and reasoning RL;
the transfer to online control problems with sequential rewards and
continuous action spaces is not established and is the question
addressed by the final stage of this project.

**2.5 Tooling and benchmark environments: ObjectRL and Gymnasium.**
The vector-control baselines reuse the PPO, SAC, and TD3
implementations shipped by *ObjectRL* \citep{baykal2025objectrl}, an
object-oriented RL codebase that exposes the three algorithms on a
shared training and evaluation loop. Reusing ObjectRL means that the
project's contribution at this stage sits at the level of the
experimental protocol and result validation rather than at the level
of re-implementing standard algorithms. *Gymnasium*
\citep{towers2024gymnasium} provides the standard environment interface
through which `CarRacing-v3` is constructed. Because ObjectRL's default
actor and critic architectures assume one-dimensional vector
observations and CarRacing returns image observations, the CarRacing
baselines use project-side CNN-based PPO, SAC, and TD3 agents
(see Section~\ref{sec:methodology-implementation}). This split is an
implementation choice, not a gap in the algorithmic literature.

**2.6 Gap and project positioning.**
GRPO and its successors have established group-relative policy
optimisation as a competitive alternative to value-baseline PPO in the
language-model and reasoning setting, while PPO, SAC, and TD3 remain
the canonical baselines for continuous control. The transfer of
group-relative advantage estimation to online control benchmarks is
not established, and is the question that the final project stage will
address. At the interim stage, this project contributes the
reproducible PPO/SAC/TD3 baseline matrix on the assignment's three
control tasks, which defines the comparison surface against which the
GRPO-control variant will later be evaluated.

---

## 8. Suggested citation placement

Citation placement that matches the English draft above. All keys are
verified against `report/references.bib`.

| Sentence in English draft | Suggested citation |
|---|---|
| 2.1 — first mention of "Generalized Advantage Estimation (GAE)" | `\citep{schulman2015gae}` |
| 2.2 — first mention of "Trust Region Policy Optimization (TRPO)" | `\citep{schulman2015trpo}` *(optional; drop if length is tight)* |
| 2.2 — first mention of "Proximal Policy Optimization (PPO)" | `\citep{schulman2017proximal}` |
| 2.3 — first mention of "Soft Actor-Critic (SAC)" | `\citep{haarnoja2018sacapps}` |
| 2.3 — first mention of "deterministic policy gradient theorem" | `\citep{silver2014deterministic}` *(optional in interim)* |
| 2.3 — first mention of "Deep DPG (DDPG)" | `\citep{lillicrap2015continuous}` *(optional in interim)* |
| 2.3 — first mention of "Twin Delayed DDPG (TD3)" | `\citep{fujimoto2018td3}` |
| 2.4 — first mention of "DeepSeekMath" / GRPO | `\citep{shao2024deepseekmath}` |
| 2.4 — first mention of "DeepSeek-R1" | `\citep{deepseekai2025deepseekr1}` *(optional; drop if length is tight)* |
| 2.5 — first mention of "ObjectRL" | `\citep{baykal2025objectrl}` |
| 2.5 — first mention of "Gymnasium" | `\citep{towers2024gymnasium}` |
| 2.5 (optional) — first mention of swing-up tasks as DM Control suite tasks | `\citep{tunyasuvunakool2020dmcontrol}` *(optional; could also live in Methodology)* |
| 2.6 — gap statement; second mention of GRPO | `\citep{shao2024deepseekmath}` *(optional, only if needed for the gap-statement sentence)* |

A minimal interim-version citation set (eight items) is:

```
schulman2015gae, schulman2017proximal, haarnoja2018sacapps,
fujimoto2018td3, shao2024deepseekmath, baykal2025objectrl,
towers2024gymnasium, deepseekai2025deepseekr1
```

Notes:

- All BibTeX keys above were verified against
  `report/references.bib`.
- `kakade2002approximately` and `ouyang2022training` exist in the bib
  but are **not** recommended for the interim Related Work section.
- The methodology section currently in the repository
  (`report/sections/03_methodology.tex`) cites
  `\citep{schulman2017proximal,haarnoja2018soft,fujimoto2018addressing}`.
  The latter two keys do **not** exist in `report/references.bib`. This
  is a separate issue from Related Work, but worth flagging when you
  next touch the methodology file (Final author notes).

---

## 9. Gap statement variants

Three alternative gap statements for the end of Related Work. Each is
cautious and interim-appropriate. Pick whichever tone fits.

**Variant A — emphasis on the open transfer question.**
> GRPO has been shown to be a competitive alternative to value-baseline
> PPO in language-model and reasoning RL, but its behaviour on online
> control benchmarks with sequential rewards and continuous action
> spaces is not yet established. The interim contribution of this
> project is the reproducible PPO/SAC/TD3 baseline matrix on the three
> assignment environments, which defines the comparison surface against
> which a GRPO-control variant will be evaluated in the final project
> stage.

**Variant B — emphasis on lineage and inheritance.**
> GRPO sits inside the PPO family of clipped-surrogate methods but
> replaces the learned value baseline with a group-relative quantity.
> The published evidence for this construction is concentrated in
> language-model RL; whether the same group-relative structure is
> useful in continuous-control settings remains an open question. The
> interim report establishes the PPO, SAC, and TD3 baseline matrix that
> the final project stage will use to study this question on the
> required control tasks.

**Variant C — emphasis on protocol and measurement.**
> The literature provides strong continuous-control baselines (PPO,
> SAC, TD3) and a recent group-relative policy-optimisation method
> developed for language-model RL (GRPO). What is currently missing is
> a careful, reproducible comparison between these two lines on a
> shared online-control protocol. The interim report contributes the
> shared protocol and the validated baseline matrix; the final stage
> will add the GRPO-control variant and perform the comparison.

---

## 10. Suggested final target length

The interim report has a tight overall page budget. Suggested length
for the final compressed `report/sections/02_related_work.tex` in the
**5-page interim draft**:

- **Paragraphs:** 5–6 short paragraphs, one per subsection (2.1–2.6),
  with subsection headings used sparingly. If page budget is very
  tight, drop the `\subsection` headings and keep the same five
  paragraphs as a flat section.
- **Approximate page length:** about **0.75–1.0 page** of the NeurIPS
  template (roughly 350–500 words).
- **Citation count:** 8 core citations (see Section 8), or 6 if you
  drop TRPO and DeepSeek-R1.

Concrete compression heuristics:

- Subsections 2.1 and 2.2 can be merged into a single paragraph that
  introduces GAE, TRPO, and PPO together.
- Subsection 2.3 can compress DPG and DDPG into a single sentence of
  background and spend the rest on SAC and TD3.
- Subsection 2.5 can compress to one sentence on ObjectRL plus one
  sentence on Gymnasium / CarRacing.
- Subsection 2.6 should not be cut — the gap statement is the most
  load-bearing sentence in the section for the interim grader.

For the final report (after the GRPO-control method is fixed), expand
2.4 and 2.6 first; the other subsections do not need to grow much.

---

## 11. Final author notes

**To decide manually**

- Whether to keep TRPO (`schulman2015trpo`) and DeepSeek-R1
  (`deepseekai2025deepseekr1`) citations in the interim version. Both
  are defensible; both can be cut for length without weakening the
  narrative.
- Whether to keep DPG (`silver2014deterministic`) and DDPG
  (`lillicrap2015continuous`) as background citations in 2.3, or
  reduce 2.3 to "TD3 \citep{fujimoto2018td3}, building on the
  deterministic-policy actor-critic line of work, ...". Both are fine.
- Whether to use `\subsection` headings or fold the section into a
  flat block of paragraphs. The interim length budget probably favours
  the flat form.
- Which of the three gap-statement variants (Section 9) best matches
  your voice. Variant A is the safest default.

**To check against the assignment PDF (`DM887_Project.pdf`)**

- Whether the assignment requires a specific list of related-work
  topics (e.g., does it require an explicit RLHF / DeepSeek-R1
  paragraph?).
- Required environment naming. The drafts use the project short names
  (`cartpole_swingup`, `acrobot_swingup`, `car_racing_continuous`).
  Confirm those match the PDF's spelling for any environment mention
  in Related Work (Methodology and Experiments sections already pin
  this down).
- Whether the assignment expects a specific citation style or count.

**To align later with Methodology, Theory, and Experiments**

- The notation drift between Methodology and the local theory
  notebooks (uppercase $S_t, A_t, R_{t+1}$ vs. lowercase $s_t, a_t,
  r_t$) is *not* Related Work's problem, but Related Work should not
  introduce its own notation that contradicts Methodology. The drafts
  here only mention $A^{\pi_\theta}(s,a)$ and the surrogate objective
  in prose, which is consistent with whichever convention you pick.
- The Methodology file currently cites `haarnoja2018soft` and
  `fujimoto2018addressing`, which are **not** in `references.bib`.
  When you next touch Methodology, swap those for `haarnoja2018sacapps`
  and `fujimoto2018td3`. This is outside the Related Work scope, but
  flagging it here so it is not missed.
- Whether GRPO ends up being characterised as "PPO with a
  group-relative baseline" or in a more general way (e.g., "a
  policy-optimisation method that uses group-relative scores") will
  depend on what the final-stage Theory section pins down. Either
  framing is acceptable in the interim.

**To postpone until the final report**

- Detailed technical comparison between GRPO and PPO at the level of
  update rules and gradient estimators.
- Any algorithmic specification of GRPO-control.
- Any RLHF/InstructGPT background paragraph that grows beyond a
  single sentence — only include this if the final method explicitly
  requires the framing.
- A theoretical-properties subsection on GRPO (convergence, variance
  reduction, etc.) — Theory and the Appendix are the right places.

**Reminders**

- This file is **inspiration only**. Do not paste the Danish or
  English drafts into `report/sections/02_related_work.tex`
  verbatim; rewrite in your own voice, and compress to the target
  length in Section 10.
- Re-verify every BibTeX key against `report/references.bib` before
  compiling, since the methodology file currently shows two keys
  that do not match.
- Keep the section honest about what is *known* (PPO/SAC/TD3,
  GRPO-in-LLM) and what is *open* (GRPO-in-control). The interim
  report wins by being precise about the literature gap, not by
  claiming the gap is already closed.
