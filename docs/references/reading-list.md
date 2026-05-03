# Reading List for DM887 GRPO Midway Report

This file lists the core papers and references for the DM887 GRPO for Control midway report.

The actual PDF files should be stored locally in:

```text
papers/
```

The `papers/` folder is intentionally Git-ignored because it may contain copyrighted papers or course-provided material.

This file is safe to commit. It documents:

1. what each paper is used for,
2. the recommended local filename,
3. the recommended BibTeX key,
4. the official or preferred link.

---

## 1. Minimum papers for the midway report

These are the most important papers for the midway report.

| Priority | Topic | Paper | Use in report | Local filename | BibTeX key | Link |
|---:|---|---|---|---|---|---|
| 1 | PPO | Schulman et al., *Proximal Policy Optimization Algorithms* | Main PPO baseline; clipped surrogate objective; on-policy baseline | `ppo_schulman_2017.pdf` | `schulman2017proximal` | https://arxiv.org/abs/1707.06347 |
| 2 | GAE | Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* | Advantage estimation for PPO; bias-variance tradeoff | `gae_schulman_2015.pdf` | `schulman2015gae` | https://arxiv.org/abs/1506.02438 |
| 3 | TRPO | Schulman et al., *Trust Region Policy Optimization* | Background for PPO; monotonic improvement/trust-region policy updates | `trpo_schulman_2015.pdf` | `schulman2015trpo` | https://arxiv.org/abs/1502.05477 |
| 4 | Conservative policy iteration | Kakade and Langford, *Approximately Optimal Approximate Reinforcement Learning* | Theoretical predecessor for conservative policy improvement | `kakade_langford_2002.pdf` | `kakade2002approximately` | https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf |
| 5 | SAC | Haarnoja et al., *Soft Actor-Critic Algorithms and Applications* | Off-policy maximum-entropy continuous-control baseline | `sac_haarnoja_2018.pdf` | `haarnoja2018sacapps` | https://arxiv.org/abs/1812.05905 |
| 6 | TD3 | Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods* | TD3 baseline; clipped double Q-learning, delayed policy updates, target smoothing | `td3_fujimoto_2018.pdf` | `fujimoto2018td3` | https://arxiv.org/abs/1802.09477 |
| 7 | DDPG | Lillicrap et al., *Continuous Control with Deep Reinforcement Learning* | Background for deterministic actor-critic methods and continuous control | `ddpg_lillicrap_2015.pdf` | `lillicrap2015continuous` | https://arxiv.org/abs/1509.02971 |
| 8 | DPG | Silver et al., *Deterministic Policy Gradient Algorithms* | Theoretical background for deterministic policy gradients and TD3/DDPG family | `dpg_silver_2014.pdf` | `silver2014deterministic` | https://proceedings.mlr.press/v32/silver14.html |
| 9 | GRPO source | Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* | Original GRPO source paper; GRPO as PPO variant | `deepseekmath_grpo_2024.pdf` | `shao2024deepseekmath` | https://arxiv.org/abs/2402.03300 |
| 10 | GRPO/RL reasoning context | DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* | Course-provided DeepSeek paper; GRPO in reasoning RL context | `deepseek_r1_2025.pdf` | `deepseekai2025deepseekr1` | https://arxiv.org/abs/2501.12948 |
| 11 | RLHF context | Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* | Context for PPO/RLHF and why GRPO emerged in LLM training | `instructgpt_rlhf_2022.pdf` | `ouyang2022training` | https://arxiv.org/abs/2203.02155 |
| 12 | ObjectRL | Baykal et al., *ObjectRL: An Object-Oriented Reinforcement Learning Codebase* | ObjectRL citation; implementation source for PPO/SAC/TD3 baselines | `objectrl_baykal_2025.pdf` | `baykal2025objectrl` | https://arxiv.org/abs/2507.03487 |
| 13 | DMC | Tunyasuvunakool et al., *dm_control: Software and Tasks for Continuous Control* | Citation for DeepMind Control Suite environments | `dm_control_2020.pdf` | `tunyasuvunakool2020dmcontrol` | https://arxiv.org/abs/2006.12983 |
| 14 | Gymnasium | Towers et al., *Gymnasium: A Standard Interface for Reinforcement Learning Environments* | Citation for Gymnasium/Farama environments and API | `gymnasium_towers_2024.pdf` | `towers2024gymnasium` | https://arxiv.org/abs/2407.17032 |

---

## 2. How to use these in the midway report

### Related work

Use these papers:

- PPO: `schulman2017proximal`
- GAE: `schulman2015gae`
- SAC: `haarnoja2018sacapps`
- TD3: `fujimoto2018td3`
- GRPO: `shao2024deepseekmath`
- RLHF context: `ouyang2022training`
- DeepSeek-R1 context: `deepseekai2025deepseekr1`

### Methodology

Use these papers:

- MDP / RL definitions: course slides and standard RL notation
- PPO/GAE advantage notation: `schulman2015gae`, `schulman2017proximal`
- TRPO/CPI connection: `schulman2015trpo`, `kakade2002approximately`

### Experiments

Use these papers/software references:

- ObjectRL: `baykal2025objectrl`
- Gymnasium: `towers2024gymnasium`
- DMC: `tunyasuvunakool2020dmcontrol`

---

## 3. Recommended download filenames

Use these filenames in `papers/`:

```text
papers/
├── ppo_schulman_2017.pdf
├── gae_schulman_2015.pdf
├── trpo_schulman_2015.pdf
├── kakade_langford_2002.pdf
├── sac_haarnoja_2018.pdf
├── td3_fujimoto_2018.pdf
├── ddpg_lillicrap_2015.pdf
├── dpg_silver_2014.pdf
├── deepseekmath_grpo_2024.pdf
├── deepseek_r1_2025.pdf
├── instructgpt_rlhf_2022.pdf
├── objectrl_baykal_2025.pdf
├── dm_control_2020.pdf
└── gymnasium_towers_2024.pdf
```

---

## 4. Notes on DeepSeek / GRPO references

There are two relevant DeepSeek papers:

1. `DeepSeekMath` (`arXiv:2402.03300`)  
   This is the key paper to cite when introducing GRPO as a PPO variant.

2. `DeepSeek-R1` (`arXiv:2501.12948`)  
   This is the course-provided DeepSeek paper and is useful for explaining the later LLM reasoning context in which GRPO became prominent.

For the midway report, cite `DeepSeekMath` for the GRPO method and cite `DeepSeek-R1` only if discussing the broader LLM/RL context.

---

## 5. Suggested `references.bib` entries

Add these entries to:

```text
report/midway/references.bib
```

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

@article{schulman2015trpo,
  title = {Trust Region Policy Optimization},
  author = {Schulman, John and Levine, Sergey and Moritz, Philipp and Jordan, Michael I. and Abbeel, Pieter},
  journal = {arXiv preprint arXiv:1502.05477},
  year = {2015}
}

@inproceedings{kakade2002approximately,
  title = {Approximately Optimal Approximate Reinforcement Learning},
  author = {Kakade, Sham M. and Langford, John},
  booktitle = {Proceedings of the Nineteenth International Conference on Machine Learning},
  year = {2002}
}

@article{haarnoja2018sacapps,
  title = {Soft Actor-Critic Algorithms and Applications},
  author = {Haarnoja, Tuomas and Zhou, Aurick and Hartikainen, Kristian and Tucker, George and Ha, Sehoon and Tan, Jie and Kumar, Vikash and Zhu, Henry and Gupta, Abhishek and Abbeel, Pieter and Levine, Sergey},
  journal = {arXiv preprint arXiv:1812.05905},
  year = {2018}
}

@inproceedings{fujimoto2018td3,
  title = {Addressing Function Approximation Error in Actor-Critic Methods},
  author = {Fujimoto, Scott and Hoof, Herke van and Meger, David},
  booktitle = {International Conference on Machine Learning},
  year = {2018}
}

@article{lillicrap2015continuous,
  title = {Continuous Control with Deep Reinforcement Learning},
  author = {Lillicrap, Timothy P. and Hunt, Jonathan J. and Pritzel, Alexander and Heess, Nicolas and Erez, Tom and Tassa, Yuval and Silver, David and Wierstra, Daan},
  journal = {arXiv preprint arXiv:1509.02971},
  year = {2015}
}

@inproceedings{silver2014deterministic,
  title = {Deterministic Policy Gradient Algorithms},
  author = {Silver, David and Lever, Guy and Heess, Nicolas and Degris, Thomas and Wierstra, Daan and Riedmiller, Martin},
  booktitle = {International Conference on Machine Learning},
  pages = {387--395},
  year = {2014}
}

@article{shao2024deepseekmath,
  title = {DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models},
  author = {Shao, Zhihong and Wang, Peiyi and Zhu, Qihao and Xu, Runxin and Song, Junxiao and Bi, Xiao and Zhang, Haowei and Zhang, Mingchuan and Li, Y. K. and Wu, Y. and Guo, Daya},
  journal = {arXiv preprint arXiv:2402.03300},
  year = {2024}
}

@article{deepseekai2025deepseekr1,
  title = {DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author = {{DeepSeek-AI}},
  journal = {arXiv preprint arXiv:2501.12948},
  year = {2025}
}

@article{ouyang2022training,
  title = {Training Language Models to Follow Instructions with Human Feedback},
  author = {Ouyang, Long and Wu, Jeff and Jiang, Xu and Almeida, Diogo and Wainwright, Carroll L. and Mishkin, Pamela and Zhang, Chong and Agarwal, Sandhini and Slama, Katarina and Ray, Alex and Schulman, John and Hilton, Jacob and Kelton, Fraser and Miller, Luke and Simens, Maddie and Askell, Amanda and Welinder, Peter and Christiano, Paul and Leike, Jan and Lowe, Ryan},
  journal = {Advances in Neural Information Processing Systems},
  year = {2022}
}

@article{baykal2025objectrl,
  title = {ObjectRL: An Object-Oriented Reinforcement Learning Codebase},
  author = {Baykal, Gulcin and Akg{"u}l, Abdullah and Haussmann, Manuel and Tasdighi, Bahareh and Werge, Nicklas and Wu, Yi-Shan and Kandemir, Melih},
  journal = {arXiv preprint arXiv:2507.03487},
  year = {2025}
}

@article{tunyasuvunakool2020dmcontrol,
  title = {dm_control: Software and Tasks for Continuous Control},
  author = {Tunyasuvunakool, Saran and Muldal, Alistair and Doron, Yotam and Liu, Siqi and Bohez, Steven and Merel, Josh and Erez, Tom and Lillicrap, Timothy and Heess, Nicolas and Tassa, Yuval},
  journal = {Software Impacts},
  volume = {6},
  pages = {100022},
  year = {2020}
}

@article{towers2024gymnasium,
  title = {Gymnasium: A Standard Interface for Reinforcement Learning Environments},
  author = {Towers, Mark and Kwiatkowski, Ariel and Terry, Jordan K. and Balis, John U. and De Cola, Gianluca and Deleu, Tristan and Goul{\~a}o, Manuel and Kallinteris, Andreas and KG, Arjun and Krimmel, Markus and others},
  journal = {arXiv preprint arXiv:2407.17032},
  year = {2024}
}
```

---

## 6. Agent guidance

Agents may inspect this file and the local `papers/` folder if present.

However:

- Do not copy long passages from papers.
- Do not rely on papers without adding citations to `references.bib`.
- Do not commit PDFs.
- For the midway report, prioritize PPO, GAE, SAC, TD3, GRPO, ObjectRL, Gymnasium, and DMC.
