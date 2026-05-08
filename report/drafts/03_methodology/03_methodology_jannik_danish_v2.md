# 03 Methodology – Jannik Danish v2

## Formål og afgrænsning

Methodology-afsnittet skal fastlægge den formelle og eksperimentelle ramme for midway-rapporten. Introduktionen motiverer projektet som en undersøgelse af GRPO i kontrolmiljøer, mens experiments-afsnittet dokumenterer de gennemførte PPO-, SAC- og TD3-baselines. Methodology skal derfor bygge bro mellem disse to dele: først ved at definere den RL-notation, rapporten bruger, og derefter ved at beskrive den baseline-protokol, som resultaterne er baseret på.

På midway-stadiet er den endelige GRPO-control-metode endnu ikke implementeret. Afsnittet skal derfor ikke præsentere GRPO-pseudokode, komponentmotivation eller konvergensanalyse som afsluttede bidrag. Det skal i stedet etablere det metodiske fundament, som den senere GRPO-variant skal bygges og evalueres ovenpå. Det passer også med opgavebeskrivelsen, hvor midway-relevante krav især omfatter related work, MDP-notation og komplette PPO/SAC/TD3-baseline-resultater.

## MDP-formulering

De tre projektmiljøer behandles som Markov decision processes. En MDP beskriver en sekventiel beslutningsproces, hvor en agent observerer en tilstand eller observation, vælger en handling, modtager en belønning og bevæger sig til en ny tilstand. Rapporten bruger den kompakte notation

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0).
$$

```latex
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma,\rho_0).
```

Her er $\mathcal{S}$ tilstands- eller observationsrummet, $\mathcal{A}$ handlingsrummet, $P(s'\mid s,a)$ overgangskernen, $r(s,a)$ belønningsfunktionen, $\gamma\in[0,1)$ diskonteringsfaktoren, og $\rho_0$ startfordelingen. Denne notation er bevidst holdt kompakt, så den kan bruges både til de vektorbaserede swing-up-miljøer og til det billedbaserede CarRacing-miljø.

I en streng teoretisk formulering kan man skelne mellem miljøets skjulte tilstand og agentens observation. I denne rapport behandles de observerede input dog som agentens state/observation representation. Det er tilstrækkeligt for midway-formålet, fordi rapporten fokuserer på implementerede baselines og evalueringsprotokol frem for en fuld POMDP-analyse.

## Policies, trajektorier og returns

En policy beskriver, hvordan agenten vælger handlinger. For stokastiske policies bruges

$$
\pi_\theta(a\mid s),
$$

```latex
\pi_\theta(a\mid s)
```

hvor $\theta$ er policyens parametre. Denne notation passer til PPO og SAC. For deterministiske policies, som er den naturlige notation for TD3, bruges

$$
\mu_\theta(s).
$$

```latex
\mu_\theta(s)
```

En policy inducerer trajektorier gennem miljøet. En trajektorie kan skrives som

$$
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots).
$$

```latex
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots).
```

Træningsmålet i RL beskrives ofte ved en diskonteret return,

$$
G_t^\gamma
=
\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
$$

```latex
G_t^\gamma
=
\sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
```

I rapportens experiments-afsnit rapporteres der derimod **undiscounted evaluation episode return**. Den kan skrives som

$$
G^{\mathrm{eval}}(\tau)
=
\sum_{t=0}^{H-1} r_{t+1},
$$

```latex
G^{\mathrm{eval}}(\tau)
=
\sum_{t=0}^{H-1} r_{t+1},
```

hvor $H$ er episode-længden. Denne sondring er vigtig: algoritmerne kan bruge diskonterede træningsmål og interne losses, mens rapportens læringskurver viser observeret udiskonteret return under evaluering.

## Value functions og advantage

Rapporten bruger standard notation for state-value, action-value og advantage. State-value funktionen er

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t^\gamma \mid S_t=s
\right],
$$

```latex
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t^\gamma \mid S_t=s
\right],
```

og action-value funktionen er

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t^\gamma \mid S_t=s,A_t=a
\right].
$$

```latex
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t^\gamma \mid S_t=s,A_t=a
\right].
```

Advantage defineres som

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

```latex
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
```

Denne notation er nødvendig for at beskrive policy-gradient- og actor-critic-metoderne i rapporten. Den er også central for den senere GRPO-control-udvidelse, fordi GRPO motiveres af en gruppe-relativ konstruktion af policy-opdateringssignaler. I midway-rapporten bruges dette dog kun som forberedende notation; den konkrete GRPO-control-algoritme hører til final report.

## Baseline-algoritmernes rolle

PPO, SAC og TD3 fungerer som baseline-algoritmer. De er ikke rapportens nye metodiske bidrag, men udgør den sammenligningsflade, som den senere GRPO-control-metode skal vurderes imod.

PPO repræsenterer en on-policy policy-gradient tilgang med stokastisk policy. SAC repræsenterer en off-policy actor-critic tilgang med stokastisk policy og maximum-entropy-regularisering. TD3 repræsenterer en off-policy deterministisk actor-critic tilgang til kontinuerte handlingsrum. Tilsammen dækker de tre baselines centrale designvalg i moderne deep reinforcement learning: on-policy versus off-policy, stokastisk versus deterministisk policy, og policy-gradient versus actor-critic-baseret optimering.

På midway-stadiet skal resultaterne ikke tolkes som endelige udsagn om algoritmernes generelle kvalitet. Der er ikke udført hyperparameter-optimering, og træningsbudgetterne er korte. Det metodiske formål er derfor at etablere en reproducerbar baseline-matrix, ikke at bevise algoritmisk overlegenhed.

## Miljøer og implementation paths

Projektet evaluerer tre miljøer: `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous`. De to swing-up-miljøer er vektorbaserede og køres gennem ObjectRL via en project-side environment bridge. Denne bridge gør det muligt at bruge projektets miljøer i ObjectRL's trænings- og evalueringsloop uden at ændre kildekoden i `external/objectrl`.

De vektorbaserede miljøer tilpasses ved, at DM Control-observationer flades ud til 1-D `float32`-vektorer og eksponeres gennem et Gymnasium-lignende interface. Dermed matcher de ObjectRL's forventning om vektorbaserede `Box`-observationer.

`car_racing_continuous` kræver et andet implementation path, fordi miljøet leverer RGB-billedobservationer. I stedet for at ændre ObjectRL bruger projektet egne CNN-baserede implementeringer af PPO, SAC og TD3 til CarRacing. Observationerne konverteres fra HWC `uint8` til CHW `float32` i intervallet $[0,1]$:

$$
(96,96,3) \rightarrow (3,96,96).
$$

```latex
(96,96,3) \rightarrow (3,96,96).
```

Denne opdeling er et praktisk og metodisk valg. CarRacing er ikke udskudt eller udeladt; det indgår i den fulde baseline-matrix, men med en modelarkitektur der passer til billedobservationer. CarRacing-kørslerne blev udført på Google Colab med CUDA og kopieret tilbage til det lokale repository, så alle resultater kan analyseres gennem den samme report-facing pipeline.

## Trænings- og evalueringsprotokol

Midway-protokollen evaluerer PPO, SAC og TD3 på alle tre miljøer med seeds $0,1,2,3,4$. Det giver

$$
3 \times 3 \times 5 = 45
$$

```latex
3 \times 3 \times 5 = 45
```

algoritme–miljø–seed-kombinationer. Den endelige notebook validerer, at alle 45 kombinationer er til stede, og at resultatsættet indeholder 900 evalueringsrækker.

De vektorbaserede miljøer køres med et midway-budget på 20.000 training steps, mens CarRacing køres med 10.000 training steps. Evaluering foretages ved faste intervaller med tre evaluerings-episoder per evalueringspunkt. Den rapporterede metrik er undiscounted evaluation episode return.

Resultatfilerne gemmer blandt andet algoritme, miljø, seed, training step, evaluation episode, evaluation return, status og implementation metadata. Det gør det muligt at samle kørslerne i én fælles aggregation pipeline, som producerer summary tables og læringskurver. Den metodiske pointe er, at alle baselines kan evalueres og sammenlignes gennem samme result-format, selv om der anvendes to forskellige implementation paths.

Hvis rapporten skal formalisere den aggregerede læringskurve, kan gennemsnittet for algoritme $\alpha$, miljø $e$ og training step $n$ skrives som

$$
\bar{J}_{\alpha,e}(n)
=
\frac{1}{|\mathcal{Z}|M}
\sum_{z\in\mathcal{Z}}
\sum_{m=1}^{M}
G^{\mathrm{eval}}_{\alpha,e,z,n,m}.
$$

```latex
\bar{J}_{\alpha,e}(n)
=
\frac{1}{|\mathcal{Z}|M}
\sum_{z\in\mathcal{Z}}
\sum_{m=1}^{M}
G^{\mathrm{eval}}_{\alpha,e,z,n,m}.
```

Her er $\mathcal{Z}=\{0,1,2,3,4\}$ seed-mængden, $M$ er antallet af evaluerings-episoder, og $G^{\mathrm{eval}}_{\alpha,e,z,n,m}$ er den udiskonterede return for evaluerings-episode $m$. Denne ligning er nyttig, hvis methodology-afsnittet skal gøre præcist, hvad læringskurvernes gennemsnit repræsenterer.

## Forberedelse til final GRPO-stage

Den endelige GRPO-control-variant er planlagt til næste projektfase. Midway-metodologien fastlægger derfor den notation, baseline-struktur og evalueringsprotokol, som GRPO senere kan tilføjes til. I final report bør GRPO evalueres under samme MDP-formulering, samme evalueringsmetrik og samme seed-struktur som PPO, SAC og TD3.

Methodology-afsnittets bidrag er dermed at gøre den kommende sammenligning mulig: det definerer problemformuleringen, baseline-rollerne, miljøopsætningen og evalueringsprotokollen. Det er ikke et claim om, at GRPO-control allerede er færdig, eller at de nuværende baselines er endeligt optimerede.

## Anbefaling til LaTeX-versionen

Den engelske LaTeX-version bør være kortere end denne danske v2. Jeg anbefaler at bruge følgende struktur:

1. `Problem Formulation`
2. `Policies, Returns, and Value Functions`
3. `Baseline Algorithms`
4. `Environment and Implementation Setup`
5. `Training and Evaluation Protocol`
6. `Preparation for the GRPO-Control Extension`

Den formelle mean-evaluation-ligning kan medtages, hvis der er plads. Hvis rapporten bliver for lang, kan den udelades, fordi experiments-afsnittet allerede forklarer den praktiske evalueringsprotokol.
