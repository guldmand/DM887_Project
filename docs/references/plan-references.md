# Plan: References, BibTeX, and Citation Management

## 0. Objective

Use proper academic references in the DM887 midway and final report through BibTeX.

The immediate goals are:

1. create `report-template/references.bib`,
2. cite PPO, GAE, SAC, TD3, GRPO, and ObjectRL,
3. learn a workflow reusable for the master thesis,
4. avoid manual reference formatting.

---

## 1. Recommended LaTeX setup

In `main.tex`:

```latex
\usepackage[numbers]{natbib}
```

or, if the NeurIPS template already loads natbib, do not load it twice.

At the end of `main.tex`:

```latex
\bibliographystyle{plainnat}
\bibliography{references}
```

Then cite with:

```latex
\citet{schulman2017proximal} introduced PPO as ...
```

or:

```latex
PPO stabilizes policy-gradient updates using a clipped surrogate objective \citep{schulman2017proximal}.
```

---

## 2. Minimum references for midway

The midway report should cite at least:

1. PPO
2. GAE
3. SAC
4. TD3
5. GRPO source
6. ObjectRL
7. Gymnasium/Farama or DMC suite if used directly

---

## 3. Initial `references.bib`

Create:

```text
report-template/references.bib
```

with the following starting entries.

```bibtex
@article{schulman2017proximal,
  title = {Proximal Policy Optimization Algorithms},
  author = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year = {2017}
}

@article{schulman2015gae,
  title = {High-Dimensional Continuous Control Using Generalized Advantage Estimation},
  author = {Schulman, John and Moritz, Philipp and Levine, Sergey and Jordan, Michael and Abbeel, Pieter},
  journal = {arXiv preprint arXiv:1506.02438},
  year = {2015}
}

@inproceedings{haarnoja2018sac,
  title = {Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor},
  author = {Haarnoja, Tuomas and Zhou, Aurick and Abbeel, Pieter and Levine, Sergey},
  booktitle = {International Conference on Machine Learning},
  year = {2018}
}

@inproceedings{fujimoto2018td3,
  title = {Addressing Function Approximation Error in Actor-Critic Methods},
  author = {Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle = {International Conference on Machine Learning},
  year = {2018}
}

@article{baykal2025objectrl,
  title = {ObjectRL: An Object-Oriented Reinforcement Learning Codebase},
  author = {Baykal, Gulcin and Akg{"u}l, Abdullah and Haussmann, Manuel and Tasdighi, Bahareh and Werge, Nicklas and Wu, Yi-Shan and Kandemir, Melih},
  year = {2025},
  journal = {arXiv preprint arXiv:2507.03487}
}

@article{brockman2016openai,
  title = {OpenAI Gym},
  author = {Brockman, Greg and Cheung, Vicki and Pettersson, Ludwig and Schneider, Jonas and Schulman, John and Tang, Jie and Zaremba, Wojciech},
  journal = {arXiv preprint arXiv:1606.01540},
  year = {2016}
}

@article{tunyasuvunakool2020dmcontrol,
  title = {dm_control: Software and Tasks for Continuous Control},
  author = {Tunyasuvunakool, Saran and Muldal, Alistair and Doron, Yotam and Liu, Siqi and Bohez, Steven and Merel, Josh and Erez, Tom and Lillicrap, Timothy and Heess, Nicolas and Tassa, Yuval},
  journal = {Software Impacts},
  volume = {6},
  pages = {100022},
  year = {2020}
}
```

For GRPO, add the exact BibTeX entry after verifying the source you decide to cite. A placeholder can be added temporarily:

```bibtex
@article{shao2024deepseekmath,
  title = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  journal = {arXiv preprint arXiv:2402.03300},
  year = {2024}
}
```

Verify this entry before final submission.

---

## 4. Citation style rules

Use citations to support claims, not every sentence.

Good:

```latex
PPO is used as the on-policy baseline because it stabilizes policy-gradient updates using a clipped surrogate objective \citep{schulman2017proximal}.
```

Bad:

```latex
PPO is good \citep{schulman2017proximal}.
```

Good:

```latex
SAC is included as an off-policy maximum-entropy actor-critic baseline for continuous control \citep{haarnoja2018sac}.
```

Bad:

```latex
SAC is another RL algorithm.
```

---

## 5. BibTeX workflow

### Add a new paper

1. Find official BibTeX from the paper page, arXiv, OpenReview, publisher, or author website.
2. Paste into `references.bib`.
3. Normalize key names:

```text
authorYYYYshorttitle
```

Examples:

```text
schulman2017proximal
haarnoja2018sac
fujimoto2018td3
baykal2025objectrl
```

4. Cite in LaTeX.
5. Compile with BibTeX.

---

## 6. Useful tools

Suggested tools:

- Google Scholar → Cite → BibTeX
- arXiv export citation
- OpenReview citation
- publisher citation export
- doi2bib.org
- Zotero or another reference manager

Always verify:

- title,
- authors,
- year,
- venue,
- arXiv identifier or DOI,
- spelling and capitalization.

---

## 7. How to cite ObjectRL in the report

Suggested sentence:

```latex
The interim baseline experiments use ObjectRL implementations of PPO, SAC, and TD3 \citep{baykal2025objectrl}.
```

Suggested longer version:

```latex
ObjectRL is used as the implementation framework for the interim PPO, SAC, and TD3 baselines, following the course clarification that these standard algorithms do not need to be reimplemented from scratch \citep{baykal2025objectrl}.
```

---

## 8. How to cite AI use

Do not cite ChatGPT/Copilot/Claude like scientific sources unless the course explicitly requires it.

Instead include an explicit statement:

```latex
\section*{Use of AI Tools}
AI tools were used to assist with planning, code scaffolding, LaTeX structuring, and drafting of preliminary explanatory text. The final technical choices, verification of claims, implementation decisions, experiment interpretation, and submitted text remain the responsibility of the author.
```

---

## 9. Bibliography checklist

Before submission:

- [ ] Every citation key compiles.
- [ ] No `?` appears in the PDF citations.
- [ ] No unused placeholder references remain unless acceptable.
- [ ] GRPO reference is verified.
- [ ] ObjectRL reference is included if ObjectRL is used.
- [ ] PPO, GAE, SAC, and TD3 are cited in related work.
- [ ] Gymnasium/DMC are cited if their environments are described.
