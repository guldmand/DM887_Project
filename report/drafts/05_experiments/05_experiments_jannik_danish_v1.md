# 05 Experiments – Jannik Danish v1

## Formål med eksperimenterne

Formålet med eksperimenterne i midway-rapporten er ikke at evaluere den endelige GRPO-baserede metode, men at etablere det eksperimentelle fundament, som den endelige sammenligning skal bygge på. På dette stadie af projektet er fokus derfor på at gennemføre og validere en komplet baseline-matrix for PPO, SAC og TD3 på de tre miljøer, der indgår i projektet.

Eksperimenterne skal vise, at hele pipeline fungerer end-to-end: miljøerne kan initialiseres, agenternes policies kan trænes, evalueringer kan gennemføres ved faste intervaller, resultater kan gemmes som CSV-filer, og læringskurver kan genereres fra de gemte resultater. Dermed fungerer denne del af rapporten både som en eksperimentel status for midway-afleveringen og som et reproducerbart sammenligningsgrundlag for den senere GRPO-control-udvidelse.

Resultaterne skal fortolkes forsigtigt. De er gennemført med et begrænset midway-budget og uden hyperparameter-optimering. Derfor bør de ikke forstås som endelige udsagn om algoritmernes generelle performance. I stedet viser de, hvordan de valgte PPO-, SAC- og TD3-baselines opfører sig under den konkrete eksperimentelle opsætning, og om der er et tilstrækkeligt stabilt grundlag for at fortsætte med den planlagte GRPO-sammenligning.

## Eksperimentel matrix

Den gennemførte midway-matrix består af tre algoritmer, tre miljøer og fem seeds:

- Algoritmer: PPO, SAC og TD3.
- Miljøer: `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous`.
- Seeds: 0, 1, 2, 3 og 4.

Det giver i alt:

$$
3 \text{ algorithms} \times 3 \text{ environments} \times 5 \text{ seeds} = 45 \text{ runs}.
$$

Copy-paste LaTeX:

```latex
3 \text{ algorithms} \times 3 \text{ environments} \times 5 \text{ seeds} = 45 \text{ runs}.
```

Den endelige notebook validerer, at alle 45 forventede kombinationer er til stede, og at resultaterne samlet indeholder 900 evalueringsrækker. Alle rækker har status `completed`, og der mangler ingen kombinationer af algoritme, miljø og seed. Denne validering er vigtig, fordi rapportens resultater dermed ikke bygger på et ufuldstændigt subset af eksperimenterne.

## Miljøer og implementation paths

Eksperimenterne bruger to forskellige implementation paths, fordi miljøerne ikke har samme observationstype.

De to vector-control-miljøer, `cartpole_swingup` og `acrobot_swingup`, køres gennem ObjectRL. Disse miljøer bruger vektorbaserede observationer og passer derfor til ObjectRL-pipelinen for PPO, SAC og TD3. Projektet anvender en project-side environment bridge, så miljøerne kan kobles til ObjectRL uden at ændre selve ObjectRL-kildekoden.

CarRacing adskiller sig fra de to andre miljøer, fordi observationerne er billedbaserede. Derfor anvendes en separat project-side CNN-implementation for CarRacing. Det betyder, at PPO, SAC og TD3 stadig evalueres på CarRacing, men med CNN-baserede policies/actors, der kan behandle billedobservationer. CarRacing-kørslerne blev udført på Google Colab med CUDA og efterfølgende kopieret tilbage til det lokale repository, så de indgår i samme result-struktur som de øvrige baseline-kørsler.

Denne opdeling er et praktisk designvalg i midway-implementationen. Det vigtigste er, at alle tre algoritmer evalueres på alle tre miljøer, og at outputtet samles i en fælles report-facing result-pipeline.

## Evalueringsprotokol

Eksperimenterne evaluerer de lærte policies ved hjælp af undiscounted evaluation episode return. Det følger opgavebeskrivelsens krav om læringskurver, hvor x-aksen er antallet af training steps før evaluering, og y-aksen er den undiscounted return opnået under evaluering.

Det er vigtigt at skelne mellem træningsformuleringen og evalueringsmålet. I RL beskrives træningsmålet ofte som en discounted return, fx:

$$
V^\pi_\gamma(s)
=
\mathbb{E}^{\pi}
\left[
\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)
\mid s_0=s
\right].
$$

Copy-paste LaTeX:

```latex
V^\pi_\gamma(s)
=
\mathbb{E}^{\pi}
\left[
\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)
\mid s_0=s
\right].
```

I experiments-afsnittet rapporteres der derimod undiscounted evaluation episode return, fordi det er den evalueringsmetrik, der kræves i projektet. Det betyder, at resultaterne beskriver den observerede episode-return under evaluering, ikke en trænings-loss eller en klassifikationsaccuracy.

Vector-control-kørslerne evalueres over et midway-budget på 20,000 training steps, mens CarRacing-kørslerne evalueres over 10,000 training steps. Disse budgetter er relativt små og skal forstås som midway-budgetter. De er tilstrækkelige til at validere pipeline og producere foreløbige læringskurver, men ikke nødvendigvis til at opnå konvergeret eller stærkt tunet performance.

Der blev ikke udført hyperparameter-optimering. Algoritmerne skal derfor forstås som baseline-konfigurationer, ikke som endeligt optimerede modeller.

## Resultatvalidering

Den endelige report-facing notebook validerer resultaterne før fortolkning. Valideringen viser:

- 45 ud af 45 forventede CSV-filer findes.
- 900 ud af 900 forventede evalueringsrækker findes.
- Alle tre algoritmer er repræsenteret: PPO, SAC og TD3.
- Alle tre miljøer er repræsenteret: `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous`.
- Alle fem seeds er repræsenteret: 0, 1, 2, 3 og 4.
- Der mangler ingen algoritme/miljø/seed-kombinationer.
- Alle evalueringsrækker har status `completed`.

Dette er et centralt resultat i sig selv for midway-rapporten, fordi det viser, at baseline-eksperimenterne ikke blot er planlagt, men faktisk gennemført og samlet i en konsistent result-struktur. Det betyder også, at de efterfølgende figurer og tabeller kan genereres fra samme validerede datagrundlag.

## Baseline-resultater

Læringskurverne viser forskellige mønstre på tværs af miljøerne. Resultaterne skal læses som deskriptive midway-resultater og ikke som endelige konklusioner om algoritmernes generelle kvalitet.

For `cartpole_swingup` ses det tydeligste læringssignal. SAC opnår den højeste mean final return ved midway-budgettet, efterfulgt af TD3. Begge algoritmer viser en tydelig positiv udvikling fra første til sidste evalueringspunkt. PPO ligger lavere i den endelige rangering, men indgår stadig som en fuldført baseline i samme evalueringsmatrix.

For `acrobot_swingup` er resultaterne mere afdæmpede og støjfyldte. PPO har den højeste mean final return ved midway-budgettet, efterfulgt af TD3 og SAC. Progress-tabellen viser positiv udvikling for PPO og TD3, mens SAC ikke viser samme forbedring i den nuværende korte opsætning. Det indikerer, at miljøet er mere udfordrende under dette budget, og at længere træning eller tuning kan være nødvendig før stærkere konklusioner kan drages.

For `car_racing_continuous` er alle final returns fortsat negative, hvilket ikke er overraskende ved et kort CNN-baseret midway-budget. SAC opnår den bedste mean final return blandt de tre CarRacing-baselines, efterfulgt af TD3 og PPO. Det vigtigste resultat for CarRacing på dette stadie er dog ikke høj performance, men at CNN-pipelinen fungerer for alle tre algoritmer, og at CarRacing nu indgår i den samme validerede baseline-matrix som de to vector-control-miljøer.

Samlet set viser resultaterne, at baseline-pipelinen er funktionel og komplet. Der er læringssignal i dele af eksperimenterne, især for `cartpole_swingup`, mens `acrobot_swingup` og `car_racing_continuous` fremstår mere vanskelige under det valgte midway-budget.

## Stabilitet og variation

Resultaterne viser også variation på tværs af seeds og evalueringspunkter. Dette er forventeligt i RL, hvor både initialisering, exploration og miljøinteraktion kan påvirke læringsforløbet.

Stability-tabellerne bør ikke bruges til at udpege en generelt “mest stabil” algoritme på tværs af alle opgaver. I stedet skal de bruges som en indikation af, hvor meget variation der ses i de konkrete midway-kørsler. En algoritme kan godt have lav variation, men samtidig lav performance. Omvendt kan en algoritme med højere final return også have større variation på tværs af seeds eller evalueringspunkter.

For eksempel viser CarRacing-resultaterne, at SAC opnår den bedste final return blandt de tre algoritmer ved midway-budgettet, men ikke nødvendigvis den laveste variation. På CartPole viser SAC og TD3 stærkere performance end PPO, men også mere variation. Det understøtter den forsigtige fortolkning: midway-resultaterne er nyttige til at validere pipeline og identificere foreløbige mønstre, men de er ikke tilstrækkelige til endelige claims om robusthed eller generalisering.

## Begrænsninger

Der er flere vigtige begrænsninger ved experiments på midway-stadiet.

For det første er træningsbudgetterne korte. Resultaterne viser derfor tidlige læringsforløb, ikke nødvendigvis konvergerede policies. Dette gælder især CarRacing, hvor billedbaserede observationer og CNN-baseret policy learning typisk kræver mere træning end simple vector-control tasks.

For det andet er der ikke udført hyperparameter-optimering. Forskelle mellem PPO, SAC og TD3 kan derfor skyldes både algoritmiske forskelle og de valgte baseline-konfigurationer. Det ville være for stærkt at konkludere, at en algoritme generelt er bedre end en anden baseret på denne midway-opsætning alene.

For det tredje er GRPO-control-metoden endnu ikke implementeret i disse eksperimenter. Det betyder, at experiments-afsnittet ikke evaluerer projektets endelige metode, men i stedet etablerer baseline-grundlaget, som den endelige metode skal sammenlignes imod.

For det fjerde bruger projektet to implementation paths: ObjectRL for vector-control og project-side CNN-agenter for CarRacing. Det er et nødvendigt og praktisk valg på midway-stadiet, men det betyder også, at forskelle mellem vector-control og CarRacing ikke kun skyldes miljøernes sværhedsgrad, men også forskelle i observationstype og modelarkitektur.

## Overgang til final GRPO-stage

Den vigtigste konklusion fra experiments-afsnittet er, at baseline-grundlaget nu er på plads. Alle PPO-, SAC- og TD3-kørsler er gennemført på alle tre miljøer med fem seeds, og resultaterne er samlet i en valideret result-pipeline. Dermed er projektet klar til næste fase, hvor den planlagte GRPO-control-udvidelse kan formaliseres, implementeres og evalueres mod de samme baseline-resultater.

I den endelige rapport bør den samme eksperimentelle struktur genbruges til GRPO-sammenligningen. Det betyder, at GRPO-control bør evalueres på de samme miljøer, med samme seed-struktur og med samme evalueringsmetrik. På den måde kan den endelige sammenligning mellem GRPO, PPO, SAC og TD3 blive mere konsistent og reproducerbar.

Midway-eksperimenterne skal derfor forstås som et fundament: de løser ikke hele projektets forskningsspørgsmål, men de etablerer den nødvendige infrastruktur, baseline-performance og result-validering, som den endelige GRPO-analyse afhænger af.
