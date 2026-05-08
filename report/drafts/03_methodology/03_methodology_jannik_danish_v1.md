# 03 Methodology – Jannik Danish v1

## Formål med methodology-afsnittet

Formålet med methodology-afsnittet er at skabe forbindelsen mellem rapportens introduktion og de konkrete baseline-eksperimenter. Introduktionen forklarer, hvorfor projektet undersøger GRPO i en kontrolkontekst, mens experiments-afsnittet dokumenterer de gennemførte PPO-, SAC- og TD3-kørsler. Methodology-afsnittet skal derfor fastlægge den formelle RL-notation og den eksperimentelle protokol, som gør resultaterne forståelige, reproducerbare og sammenlignelige.

På midway-stadiet er den endelige GRPO-control-metode endnu ikke implementeret. Metodologien skal derfor ikke beskrive en færdig GRPO-algoritme. I stedet skal afsnittet etablere det metodiske fundament for den senere GRPO-sammenligning: MDP-formulering, policies, trajektorier, returns, value functions, advantage functions, baseline-algoritmer, miljøopsætning og evalueringsprotokol.

Afsnittet skal samtidig undgå at gentage hele experiments-afsnittet. Det skal ikke give resultatrangeringer eller detaljeret figurfortolkning. I stedet skal det forklare, hvad der måles, hvordan kørslerne er struktureret, og hvorfor projektet bruger to implementation paths: ObjectRL for vektorbaserede miljøer og projektets egne CNN-baserede agenter for CarRacing.

## Problemformulering som MDP

Projektets miljøer modelleres som Markov decision processes. En MDP beskriver en sekventiel beslutningsproces, hvor en agent observerer en tilstand eller observation, vælger en handling, modtager en belønning og derefter bevæger sig til en ny tilstand. I rapporten kan den fælles problemformulering skrives som

$$
\mathcal{M}
=
(\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0).
$$

```latex
\mathcal{M}
=
(\mathcal{S}, \mathcal{A}, P, r, \gamma, \rho_0).
```

Her betegner $\mathcal{S}$ tilstands- eller observationsrummet, $\mathcal{A}$ handlingsrummet, $P(s' \mid s,a)$ overgangskernen, $r(s,a)$ belønningsfunktionen, $\gamma \in [0,1)$ diskonteringsfaktoren, og $\rho_0$ fordelingen over starttilstande.

Denne notation er en kompakt deep-RL-formulering, der passer til rapportens formål. De lokale teori-notebooks gennemgår blandt andet deterministiske beslutningsprocesser, Markov-kæder, MDP'er, diskonterede beslutningsprocesser og episodiske MDP'er. I rapporten samles disse ideer i én notation: miljøet har Markovske dynamikker, agenten påvirker overgange gennem handlinger, og træningsalgoritmerne lærer policies ud fra observerede belønninger.

De tre miljøer i projektet har forskellige observationsstrukturer, men kan beskrives med samme overordnede MDP-ramme. `cartpole_swingup` og `acrobot_swingup` behandles som vektorbaserede kontrolmiljøer, mens `car_racing_continuous` er et billedbaseret kontrolmiljø. I rapporten kan vi omtale $\mathcal{S}$ som state/observation space, fordi implementeringen arbejder direkte med observationerne fra miljøerne. Det undgår at introducere en unødigt tung POMDP-diskussion i midway-rapporten.

## Policies, trajektorier og returns

En policy beskriver, hvordan agenten vælger handlinger på baggrund af den aktuelle tilstand eller observation. For stokastiske policies bruger rapporten notation

$$
\pi_\theta(a \mid s),
$$

```latex
\pi_\theta(a \mid s)
```

hvor $\theta$ er policyens parametre. Dette passer særligt til PPO og SAC, som modellerer en fordeling over handlinger. For deterministiske policies, som er den naturlige notation for TD3, bruges

$$
\mu_\theta(s).
$$

```latex
\mu_\theta(s)
```

En policy inducerer trajektorier gennem miljøet. En trajektorie kan skrives som

$$
\tau
=
(s_0, a_0, r_1, s_1, a_1, r_2, \ldots).
$$

```latex
\tau
=
(s_0, a_0, r_1, s_1, a_1, r_2, \ldots).
```

Denne notation følger reward-indekseringen, hvor $r_{t+1}$ er belønningen efter handlingen $a_t$ i tilstanden $s_t$. Hvis pladsen i rapporten bliver trang, kan trajektoriedefinitionen holdes kort, men den er nyttig, fordi både træning og evaluering bygger på sekvenser af interaktioner.

I RL beskrives træningsmålet ofte ved en diskonteret return

$$
G_t^\gamma
=
\sum_{k=0}^{\infty}
\gamma^k R_{t+k+1}.
$$

```latex
G_t^\gamma
=
\sum_{k=0}^{\infty}
\gamma^k R_{t+k+1}.
```

Denne størrelse vægter fremtidige belønninger med diskonteringsfaktoren $\gamma$. I experiments-afsnittet rapporteres der derimod undiscounted evaluation episode return. Den kan skrives som

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

hvor $H$ er episode-længden. Dette er en vigtig sondring: træningsalgoritmerne kan bruge diskonterede mål og interne losses, mens rapportens læringskurver viser den observerede, udiskonterede return under evaluering.

## Value functions og advantage

For at forbinde baseline-algoritmerne og den senere GRPO-control-udvidelse skal rapporten definere value functions og advantage. State-value funktionen beskriver den forventede diskonterede return, når agenten starter i tilstand $s$ og følger policy $\pi$:

$$
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t^\gamma
\mid S_t=s
\right].
$$

```latex
V^\pi(s)
=
\mathbb{E}_\pi
\left[
G_t^\gamma
\mid S_t=s
\right].
```

Action-value funktionen beskriver tilsvarende den forventede return, når agenten starter i tilstand $s$, vælger handling $a$, og derefter følger policy $\pi$:

$$
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t^\gamma
\mid S_t=s, A_t=a
\right].
$$

```latex
Q^\pi(s,a)
=
\mathbb{E}_\pi
\left[
G_t^\gamma
\mid S_t=s, A_t=a
\right].
```

Advantage-funktionen defineres som forskellen mellem action-value og state-value:

$$
A^\pi(s,a)
=
Q^\pi(s,a) - V^\pi(s).
$$

```latex
A^\pi(s,a)
=
Q^\pi(s,a) - V^\pi(s).
```

Denne definition er central, fordi PPO og flere policy-gradient-metoder bruger advantage-lignende signaler i policy-opdateringen. Den er også relevant for den planlagte GRPO-control-variant, fordi GRPO netop motiveres af en gruppe-relativ måde at konstruere policy-opdateringssignaler på. I midway-rapporten bør dette kun nævnes som motivation og forberedelse. Den konkrete GRPO-control-algoritme, pseudokode og teoretiske analyse hører til den afsluttende projektfase.

## Baseline-algoritmernes metodologiske rolle

PPO, SAC og TD3 bruges som baseline-algoritmer. De er ikke det nye metodiske bidrag i denne midway-rapport, men de definerer den sammenligningsflade, som den senere GRPO-control-metode skal vurderes imod.

PPO repræsenterer en on-policy policy-gradient familie med stokastisk policy-repræsentation. SAC repræsenterer en off-policy actor-critic tilgang med stokastisk policy og maximum-entropy-regularisering. TD3 repræsenterer en off-policy deterministisk actor-critic tilgang til kontinuerte handlingsrum. Det er metodisk nyttigt at have alle tre med, fordi de dækker forskellige centrale designvalg i moderne deep reinforcement learning: on-policy versus off-policy, stokastisk versus deterministisk policy, og policy-gradient versus actor-critic-baseret optimering.

På midway-stadiet bruges algoritmerne som standardiserede referencepunkter. Der er ikke udført hyperparameter-optimering, og resultaterne skal ikke forstås som endelige udsagn om algoritmernes generelle kvalitet. Den metodiske pointe er, at alle tre baselines kan køres gennem en fælles evalueringsstruktur på alle projektets miljøer.

## Miljøer og implementation paths

Projektet anvender tre miljøer: `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous`. De to første er vektorbaserede kontrolmiljøer. De køres gennem ObjectRL via en projekt-side bridge, så projektets miljøer kan bruges i ObjectRL's trænings- og evalueringsloop uden at ændre kildekoden i `external/objectrl`.

Denne bridge er vigtig for reproducerbarheden. Projektet genbruger ObjectRL's eksisterende PPO-, SAC- og TD3-implementeringer for de vektorbaserede miljøer, men konstruerer miljøerne gennem projektets egne wrappers. `DMCGymAdapter` flader DM Control-observationer ud til 1-D `float32`-vektorer og eksponerer et Gymnasium-lignende interface. Dermed passer observationerne til ObjectRL's forventning om vektorbaserede `Box`-observationer.

`car_racing_continuous` kræver et separat implementation path. CarRacing returnerer RGB-billedobservationer med form $(96,96,3)$, mens den anvendte ObjectRL-baseline forventer 1-D vektorobservationer. Derfor bruger projektet egne CNN-baserede implementeringer af PPO, SAC og TD3 til CarRacing. Observationerne konverteres fra HWC `uint8` til CHW `float32` i intervallet $[0,1]$:

$$
(96,96,3)
\rightarrow
(3,96,96).
$$

```latex
(96,96,3)
\rightarrow
(3,96,96).
```

Dette betyder ikke, at CarRacing er udskudt eller udeladt. Tværtimod indgår CarRacing i den samme fuldførte baseline-matrix som de to vektorbaserede miljøer, men med en modelarkitektur der passer til billedobservationer. CarRacing-kørslerne blev udført på Google Colab med CUDA og efterfølgende kopieret tilbage til det lokale repository, så resultaterne kunne indgå i den samme rapport-facing result-struktur.

## Trænings- og evalueringsprotokol

Midway-protokollen evaluerer PPO, SAC og TD3 på alle tre miljøer med seeds $0,1,2,3,4$. Det giver

$$
3 \times 3 \times 5 = 45
$$

```latex
3 \times 3 \times 5 = 45
```

algoritme–miljø–seed-kombinationer. Den endelige notebook validerer, at alle 45 kombinationer er til stede, og at resultatsættet indeholder 900 evalueringsrækker.

For de vektorbaserede miljøer anvendes et midway-budget på 20.000 miljøskridt med evaluering hver 5.000 skridt og 3 evaluerings-episoder per evalueringspunkt. For CarRacing anvendes et kortere midway-budget på 10.000 miljøskridt med evaluering hver 1.000 skridt og 3 evaluerings-episoder per evalueringspunkt. Disse budgetter skal forstås som midway-budgetter, ikke som endelige konvergensbudgetter.

Resultatfilerne gemmer blandt andet algoritme, miljø, seed, training step, evaluation episode, evaluation return, status og implementation metadata. Det gør det muligt at samle alle kørsler i en fælles aggregation pipeline, som producerer summary tables og læringskurver. Den centrale evalueringsmetrik er undiscounted evaluation episode return.

Hvis rapporten skal definere den aggregerede læringskurve mere formelt, kan den gennemsnitlige evalueringsreturn for algoritme $\alpha$, miljø $e$ og træningsskridt $n$ skrives som

$$
\bar{J}_{\alpha,e}(n)
=
\frac{1}{|\mathcal{Z}|M}
\sum_{z \in \mathcal{Z}}
\sum_{m=1}^{M}
G^{\mathrm{eval}}_{\alpha,e,z,n,m},
$$

```latex
\bar{J}_{\alpha,e}(n)
=
\frac{1}{|\mathcal{Z}|M}
\sum_{z \in \mathcal{Z}}
\sum_{m=1}^{M}
G^{\mathrm{eval}}_{\alpha,e,z,n,m},
```

hvor $\mathcal{Z}=\{0,1,2,3,4\}$ er seed-mængden, $M$ er antallet af evaluerings-episoder, og $G^{\mathrm{eval}}_{\alpha,e,z,n,m}$ er den udiskonterede return for evaluerings-episode $m$. Denne ligning er ikke absolut nødvendig, men den kan være nyttig, hvis methodology-afsnittet skal forklare præcist, hvad læringskurvernes gennemsnit repræsenterer.

## Planlagt GRPO-control-udvidelse

Den endelige GRPO-control-variant er ikke implementeret på midway-stadiet. Det er vigtigt at skrive dette eksplicit, så rapporten ikke overclaimer. Midway-metodologien etablerer i stedet den notation, den softwarestruktur og den evalueringsprotokol, som den endelige GRPO-metode skal bygges ovenpå.

I final project stage kan GRPO tilføjes som en fjerde algoritme under samme MDP-formulering, samme evalueringsmetrik og samme seed-struktur. Det vil gøre sammenligningen mellem GRPO, PPO, SAC og TD3 mere konsistent. De konkrete algoritmiske detaljer, pseudokode, komponentmotivation og eventuel teori hører til final report og bør ikke præsenteres som afsluttede resultater i midway-rapporten.

Methodology-afsnittets hovedbidrag er derfor ikke at præsentere en ny algoritme, men at gøre den kommende sammenligning mulig: det fastlægger problemformuleringen, baseline-rollerne, miljøopsætningen og evalueringsprotokollen, så experiments-afsnittet kan rapportere resultaterne på et klart og reproducerbart grundlag.

## Noter til næste version

Denne version er tænkt som dansk master draft. Før den oversættes til engelsk LaTeX, bør vi beslutte tre ting:

1. Om vi vil bruge $\rho_0$ eller $p_0$ som notation for startfordelingen. Jeg anbefaler $\rho_0$, fordi det er almindeligt i deep RL og allerede passer godt til den kompakte MDP-tuple.
2. Om vi vil omtale $\mathcal{S}$ som state space eller state/observation space. Jeg anbefaler state/observation space, fordi CarRacing og DM Control praktisk set leverer observationer, og fordi vi ikke ønsker en tung POMDP-diskussion i midway-rapporten.
3. Om den formelle mean-evaluation-ligning skal med i final LaTeX. Den er metodisk præcis, men kan udelades, hvis afsnittet bliver for langt.
