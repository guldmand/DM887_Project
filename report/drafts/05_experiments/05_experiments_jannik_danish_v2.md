# 05 Experiments – Jannik Danish v2

## Formål med eksperimenterne

Formålet med eksperimenterne i midway-rapporten er at etablere et reproducerbart baseline-grundlag for den senere GRPO-control-sammenligning. Den endelige GRPO-baserede metode er endnu ikke implementeret på dette stadie. Derfor fokuserer experiments-afsnittet på at gennemføre, validere og fortolke de krævede PPO-, SAC- og TD3-baselines på de tre projektmiljøer.

Eksperimenterne fungerer som en end-to-end-validering af projektets pipeline. Miljøerne kan initialiseres, policies kan trænes, evalueringer kan udføres ved faste intervaller, resultater kan gemmes som CSV-filer, og læringskurver kan genereres fra de gemte resultater. Dermed dokumenterer midway-rapporten ikke blot foreløbig performance, men også at den eksperimentelle infrastruktur er klar til den endelige GRPO-udvidelse.

Resultaterne fortolkes forsigtigt. De er produceret med et begrænset midway-budget og uden hyperparameter-optimering. De skal derfor ikke læses som endelige udsagn om algoritmernes generelle performance, men som foreløbige baseline-resultater under en konkret og reproducerbar opsætning.

## Eksperimentel matrix

Midway-eksperimenterne består af tre algoritmer, tre miljøer og fem seeds:

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

Den endelige report-facing notebook validerer, at alle 45 forventede kombinationer er til stede, og at resultaterne samlet indeholder 900 evalueringsrækker. Alle evalueringsrækker har status `completed`, og der mangler ingen kombinationer af algoritme, miljø og seed.

## Miljøer og implementation paths

Eksperimenterne bruger to implementation paths, fordi miljøerne har forskellige observationstyper.

`cartpole_swingup` og `acrobot_swingup` behandles som vector-control-miljøer og køres gennem ObjectRL. Projektet bruger en project-side environment bridge, så de valgte miljøer kan bruges med ObjectRL uden at ændre ObjectRL-kildekoden.

`car_racing_continuous` adskiller sig fra de to andre miljøer, fordi observationerne er billedbaserede. Derfor bruges en separat project-side CNN-implementation til CarRacing. PPO, SAC og TD3 evalueres dermed stadig på CarRacing, men med CNN-baserede policies/actors, der kan behandle billedobservationer. CarRacing-kørslerne blev udført på Google Colab med CUDA og efterfølgende kopieret tilbage til det lokale repository, så de indgår i samme result-struktur som de øvrige baseline-resultater.

Denne opdeling er et praktisk designvalg i midway-implementationen. Det centrale er, at alle tre algoritmer evalueres på alle tre miljøer, og at resultaterne samles i én fælles result-pipeline.

## Evalueringsprotokol

De lærte policies evalueres med undiscounted evaluation episode return. Det følger opgavebeskrivelsens krav om læringskurver, hvor x-aksen angiver training steps before evaluation, og y-aksen angiver den observerede undiscounted return under evaluering.

Det er vigtigt at skelne mellem træningsformuleringen og evalueringsmålet. I reinforcement learning beskrives træningsmålet ofte som en discounted return, eksempelvis:

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

I experiments-afsnittet rapporteres der derimod undiscounted evaluation episode return. Resultaterne beskriver derfor observeret episode-return under evaluering, ikke trænings-loss eller klassifikationsaccuracy.

Vector-control-kørslerne evalueres over et midway-budget på 20,000 training steps, mens CarRacing-kørslerne evalueres over 10,000 training steps. Disse budgetter er tilstrækkelige til at validere pipeline og producere foreløbige læringskurver, men ikke nødvendigvis til at opnå konvergerede eller stærkt optimerede policies.

Der blev ikke udført hyperparameter-optimering. Algoritmerne skal derfor forstås som baseline-konfigurationer.

## Resultatvalidering

Før resultaterne fortolkes, validerer notebooken datagrundlaget. Valideringen viser:

- 45 ud af 45 forventede CSV-filer findes.
- 900 ud af 900 forventede evalueringsrækker findes.
- PPO, SAC og TD3 er alle repræsenteret.
- `cartpole_swingup`, `acrobot_swingup` og `car_racing_continuous` er alle repræsenteret.
- Seeds 0–4 er alle repræsenteret.
- Der mangler ingen algoritme/miljø/seed-kombinationer.
- Alle evalueringsrækker har status `completed`.

Dette er et centralt midway-resultat. Det viser, at baseline-eksperimenterne er gennemført som en komplet matrix, og at figurer og tabeller i rapporten bygger på et valideret datagrundlag.

## Baseline-resultater

Læringskurverne viser forskellige mønstre på tværs af miljøerne. Resultaterne skal læses som deskriptive midway-resultater, ikke som endelige konklusioner om algoritmernes generelle kvalitet.

For `cartpole_swingup` ses det tydeligste læringssignal. SAC opnår den højeste mean final return ved midway-budgettet, efterfulgt af TD3. Begge algoritmer viser en tydelig positiv udvikling fra første til sidste evalueringspunkt. PPO ligger lavere i den endelige rangering, men indgår stadig som en fuldført baseline i samme matrix.

For `acrobot_swingup` er resultaterne mere afdæmpede. PPO opnår den højeste mean final return ved midway-budgettet, efterfulgt af TD3 og SAC. PPO og TD3 viser positiv udvikling gennem forløbet, mens SAC ikke viser samme forbedring i den nuværende korte opsætning. Det peger på, at miljøet er mere udfordrende under det valgte budget.

For `car_racing_continuous` er final returns fortsat negative for alle tre algoritmer. Det er forventeligt ved et kort CNN-baseret midway-budget. SAC opnår den bedste mean final return blandt CarRacing-baselines, efterfulgt af TD3 og PPO. Det vigtigste resultat her er dog, at CNN-pipelinen fungerer for alle tre algoritmer, og at CarRacing nu er inkluderet i den samlede validerede baseline-matrix.

Samlet viser resultaterne, at baseline-pipelinen er funktionel og komplet. Der er tydeligt læringssignal i dele af eksperimenterne, især for `cartpole_swingup`, mens `acrobot_swingup` og `car_racing_continuous` fremstår mere vanskelige under midway-budgettet.

## Stabilitet og variation

Resultaterne viser variation på tværs af seeds og evalueringspunkter. Dette er forventeligt i reinforcement learning, hvor initialisering, exploration og miljøinteraktion kan påvirke læringsforløbet.

Stability-tabellerne bør ikke bruges til at udpege en generelt mest stabil algoritme på tværs af alle opgaver. De viser i stedet variationen i de konkrete midway-kørsler. En algoritme kan have lav variation, men samtidig lav performance, mens en anden algoritme kan opnå højere final return med større variation.

For eksempel opnår SAC den bedste final return på CarRacing ved midway-budgettet, men ikke nødvendigvis den laveste variation. På CartPole opnår SAC og TD3 bedre final performance end PPO, men med mere variation. Det understøtter den overordnede fortolkning: resultaterne er nyttige til at validere pipeline og identificere foreløbige mønstre, men de er ikke tilstrækkelige til endelige claims om robusthed eller generalisering.

## Begrænsninger

Midway-eksperimenterne har fire centrale begrænsninger.

For det første er træningsbudgetterne korte. Resultaterne viser derfor tidlige læringsforløb, ikke nødvendigvis konvergerede policies. Dette gælder især CarRacing, hvor billedbaserede observationer og CNN-baseret policy learning typisk kræver længere træning.

For det andet er der ikke udført hyperparameter-optimering. Forskelle mellem PPO, SAC og TD3 kan derfor skyldes både algoritmiske forskelle og de valgte baseline-konfigurationer.

For det tredje er GRPO-control-metoden endnu ikke implementeret i disse eksperimenter. Afsnittet evaluerer derfor ikke projektets endelige metode, men etablerer baseline-grundlaget, som den endelige metode skal sammenlignes imod.

For det fjerde bruger projektet to implementation paths: ObjectRL for vector-control og project-side CNN-agenter for CarRacing. Det er et praktisk valg på midway-stadiet, men betyder også, at forskelle mellem vector-control og CarRacing både afspejler miljøernes sværhedsgrad, observationstype og modelarkitektur.

## Overgang til final GRPO-stage

Den vigtigste konklusion fra experiments-afsnittet er, at baseline-grundlaget nu er på plads. Alle PPO-, SAC- og TD3-kørsler er gennemført på alle tre miljøer med fem seeds, og resultaterne er samlet i en valideret result-pipeline.

I den endelige projektfase bør den samme eksperimentelle struktur genbruges til GRPO-control. Det betyder, at GRPO bør evalueres på de samme miljøer, med samme seed-struktur og samme evalueringsmetrik. På den måde kan den endelige sammenligning mellem GRPO, PPO, SAC og TD3 blive mere konsistent og reproducerbar.

Midway-eksperimenterne løser derfor ikke hele projektets forskningsspørgsmål, men de etablerer den nødvendige infrastruktur, baseline-performance og result-validering, som den endelige GRPO-analyse afhænger af.
