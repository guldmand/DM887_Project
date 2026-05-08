# 02 Related Work – Jannik Danish v1

## Formål med Related Work

Related Work-afsnittet skal i denne interimrapport forklare, hvorfor projektet sammenligner mod netop PPO, SAC og TD3, og hvorfor GRPO er en relevant, men endnu ikke implementeret, retning for den endelige projektfase. Afsnittet skal derfor ikke være en bred lærebogsgennemgang af reinforcement learning. Det skal i stedet positionere projektet i forhold til de algoritmefamilier, som baseline-matrixen repræsenterer, og forklare det åbne gap, som den senere GRPO-control metode skal undersøge.

PPO, SAC og TD3 er ikke tre tilfældige algoritmer. De repræsenterer tre forskellige tilgange til deep reinforcement learning for control: on-policy policy-gradient metoder, off-policy stokastiske actor-critic metoder og off-policy deterministiske actor-critic metoder til kontinuerte aktionsrum. Dermed fungerer de som et bredt og relevant sammenligningsgrundlag for en senere GRPO-baseret kontrolmetode.

Samtidig skal Related Work gøre det klart, at GRPO ikke allerede er evalueret i denne midway-version. GRPO motiverer den endelige projektretning, men interimrapportens bidrag er at etablere det baseline-grundlag, som GRPO-control senere skal måles imod.

## Policy-gradient metoder, PPO og advantage estimation

Policy-gradient metoder optimerer en politik direkte ud fra samples fra miljøet. I stedet for først at lære en eksplicit model af dynamikken forsøger metoderne at ændre politikkens parametre i en retning, der øger den forventede return. I praksis afhænger denne type metoder ofte af et estimat af advantage-funktionen, fordi advantage-estimatet angiver, om en handling var bedre eller dårligere end forventet i en given tilstand.

Generalized Advantage Estimation (GAE) er relevant i denne sammenhæng, fordi metoden giver en praktisk måde at estimere advantages på med en bias-variance tradeoff. Det gør GAE til en vigtig baggrund for moderne on-policy policy-gradient metoder og for forståelsen af, hvordan policy-update signaler kan konstrueres.

PPO er projektets on-policy baseline. Metoden stabiliserer policy-gradient opdateringer ved hjælp af et clipped surrogate objective, som begrænser effekten af for store ændringer mellem den gamle og den nye politik. PPO er derfor et naturligt referencepunkt i projektet, både fordi det er en etableret baseline for trajectory-baseret policy optimization, og fordi GRPO selv er motiveret som en PPO-relateret metode.

I denne rapport skal PPO ikke fremstilles som den generelt bedste metode. Dens rolle er at give et stærkt on-policy sammenligningspunkt og samtidig forbinde klassisk policy-gradient litteratur med den senere GRPO-diskussion.

## Off-policy actor-critic metoder: SAC og TD3

SAC repræsenterer en anden baseline-familie. Hvor PPO er on-policy, er SAC en off-policy actor-critic metode baseret på maximum-entropy reinforcement learning. Det betyder, at SAC ikke kun optimerer forventet return, men også belønner højere policy-entropi. Denne formulering kan understøtte exploration og gøre læringen mindre afhængig af tidlige, for snævre valg i politikken.

SAC er relevant som baseline, fordi kontrolopgaver ofte kræver både stabil critic-læring og effektiv udnyttelse af tidligere erfaringer. I projektet bruges SAC derfor som den stokastiske off-policy actor-critic baseline, der supplerer PPO’s on-policy perspektiv.

TD3 repræsenterer den deterministiske actor-critic tradition. Denne linje bygger på deterministic policy gradients, hvor en deterministisk actor optimeres gennem gradientinformation fra en action-value funktion. DDPG udvider denne idé til deep reinforcement learning med replay buffer og target networks. TD3 videreudvikler DDPG-linjen ved at adressere overestimering og funktionsapproksimationsfejl gennem blandt andet twin critics, delayed policy updates og target policy smoothing.

TD3 er derfor en naturlig baseline for kontinuerte kontrolopgaver. Sammen med PPO og SAC giver TD3 projektet et baseline-set, der dækker både on-policy og off-policy læring, stokastiske og deterministiske politikker samt forskellige actor-critic designvalg.

## GRPO som motivation for final-stage metoden

Group Relative Policy Optimization (GRPO) motiverer den endelige projektretning. I DeepSeekMath introduceres GRPO som en PPO-relateret metode til reinforcement learning for matematisk ræsonnement i sprogmodeller. Den centrale idé er at beregne et gruppe-relativt update-signal ud fra flere sampled outputs for samme prompt, i stedet for at være afhængig af en separat value model på samme måde som traditionelle actor-critic metoder.

Denne idé er interessant for projektet, fordi den peger på en alternativ måde at konstruere policy-update signaler på. I stedet for kun at sammenligne en handling eller trajectory mod en lært value baseline, kan man undersøge, om relative sammenligninger inden for en gruppe kan give et brugbart læringssignal.

Det er dog vigtigt, at rapporten ikke overfører GRPO-resultater fra sprogmodeller direkte til control. LLM-baseret mathematical reasoning og online control er meget forskellige problemtyper. Control indebærer sekventiel miljøinteraktion, temporal credit assignment, kontinuerte eller strukturerede aktionsrum og seed-afhængige læringsforløb. Derfor er det stadig et åbent spørgsmål, om GRPO-lignende group-relative updates kan formuleres og evalueres meningsfuldt i online control benchmarks.

I denne interimrapport er GRPO-control derfor ikke præsenteret som en færdig metode. GRPO fungerer i stedet som motivationen for final-stage projektet, mens den nuværende rapport etablerer de PPO-, SAC- og TD3-baselines, som en senere GRPO-control variant skal sammenlignes med.

## Tooling, environments og reproducerbarhed

Projektet bygger på eksisterende tooling for at gøre baseline-eksperimenterne mere reproducerbare. ObjectRL bruges som implementeringsgrundlag for PPO, SAC og TD3 på de vektorbaserede control environments. Det betyder, at projektet ikke forsøger at reimplementere standardalgoritmerne fra bunden, men i stedet fokuserer på at etablere en konsistent experimental pipeline, result validation og comparison setup.

Gymnasium er relevant som standardiseret miljøinterface i moderne RL-eksperimenter og som del af projektets environment setup. CarRacing kræver dog en separat project-side CNN implementation, fordi billedobservationerne ikke passer direkte ind i den vector-observation workflow, der anvendes for ObjectRL-baserede runs. Dette er en implementation path forskel, ikke et nyt algoritmisk bidrag.

Tooling-delen bør holdes kort i den endelige rapport. Den vigtigste pointe er, at ObjectRL og Gymnasium understøtter reproducerbarhed og standardiseret eksperimentel struktur, mens projektets egentlige midway-bidrag ligger i den komplette og validerede PPO/SAC/TD3 baseline-matrix.

## Gap og projektets positionering

Samlet set placerer litteraturen projektet mellem etablerede control baselines og nyere group-relative policy optimization fra language-model/reasoning domænet. PPO, SAC og TD3 dækker centrale referencepunkter for deep reinforcement learning i control: on-policy policy optimization, off-policy stochastic actor-critic learning og deterministic actor-critic learning. GRPO introducerer en anden måde at tænke policy-update signaler på, men dens anvendelse i online control er endnu ikke etableret.

Det åbne spørgsmål er derfor ikke, om GRPO allerede er bedre end PPO, SAC eller TD3. Det åbne spørgsmål er, om en GRPO-lignende group-relative idé kan tilpasses online control tasks på en måde, der kan evalueres retfærdigt mod etablerede baselines.

Ved midway-stadiet besvarer projektet ikke dette spørgsmål endeligt. I stedet etablerer rapporten det nødvendige sammenligningsgrundlag: en komplet PPO/SAC/TD3 baseline-matrix på tre environments med fem seeds, validerede result files og learning curves. Den endelige projektfase skal bygge oven på dette grundlag ved at formulere, implementere og evaluere en GRPO-control variant under samme eksperimentelle struktur.

## Noter til senere komprimering

Den endelige `02_related_work.tex` bør sandsynligvis være omkring 0.9–1.2 sider i den nuværende interimrapport. Derfor bør denne danske v1 ikke oversættes direkte og ukritisk. Den bør først strammes til en dansk v2, hvor gentagelser fjernes, og hvor teksten samles i 4–5 stærke akademiske afsnit.

Mulig endelig struktur uden subsections:

1. Baseline framing: PPO, SAC og TD3 som tre komplementære algoritmefamilier.
2. PPO + GAE: policy-gradient, clipped updates og advantage estimation.
3. SAC + TD3: off-policy stochastic og deterministic actor-critic baselines.
4. GRPO: motivation fra language-model reasoning og hvorfor transfer til control er åben.
5. Tooling + gap: ObjectRL/Gymnasium og projektets interim-positionering.

Claims der skal bevares:
- PPO, SAC og TD3 er baselines, ikke projektets nye metode.
- GRPO-control er ikke implementeret i midway-resultaterne.
- Baseline-matrixen er etableret for at muliggøre final-stage comparison.
- Transfer fra LLM/reasoning GRPO til online control er ikke direkte.

Claims der skal undgås:
- At GRPO er bevist bedre til control.
- At midway-resultaterne viser generel algorithmic superiority.
- At alle algoritmer er konvergeret.
- At CarRacing er deferred eller ikke implementeret.
- At ObjectRL direkte understøtter projektets CarRacing image-observation setup.
