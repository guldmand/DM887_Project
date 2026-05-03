"""CarRacing CNN baseline (project-side).

A clearly-labeled fallback path for the DM887 midway baselines. ObjectRL
asserts 1-D Box observations in ``objectrl/agents/base_agent.py`` and its
default actor/critic ``arch`` callables expect ``dim_state`` as an ``int``.
Adding CNN support would require modifying ``external/objectrl/``, which is
out of scope. This module therefore implements a small SAC agent with a
shared CNN feature extractor for the continuous Car Racing environment.

Components
----------
- ``CarRacingCNNWrapper`` :  HWC uint8 -> CHW float32 in [0, 1] (channels-first,
  no spatial resize, no grayscale). Output observation space:
  ``Box(0.0, 1.0, (3, 96, 96), dtype=float32)``.
- ``ImageReplayBuffer`` :  stores observations as uint8 (HWC) to save memory;
  performs the /255 + transpose conversion on sample.
- ``CNNFeatureExtractor`` :  Atari-style 3-conv stack -> linear projection.
- ``SACActorCNN`` / ``SACCriticCNN`` / ``SACAgentCNN`` :  minimal continuous SAC
  with automatic temperature tuning. Each network owns its own CNN trunk
  (no parameter sharing across actor/critic, matching most SAC-image
  implementations).

This module does **not** depend on ObjectRL; it depends only on PyTorch,
NumPy, and Gymnasium. PPO and TD3 are not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------


class CarRacingCNNWrapper(gym.ObservationWrapper):
    """Convert CarRacing HWC uint8 obs to CHW float32 in [0, 1]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        h, w, c = env.observation_space.shape  # (96, 96, 3)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(c, h, w), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) / 255.0).transpose(2, 0, 1)


def make_carracing_cnn_env(seed: int) -> gym.Env:
    """Create the CarRacing CNN env (continuous, image obs preprocessed)."""
    env = gym.make("CarRacing-v3", continuous=True)
    env = CarRacingCNNWrapper(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# ---------------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------------


class CNNFeatureExtractor(nn.Module):
    """Atari-style 3-conv CNN -> linear projection.

    Input  : Tensor of shape (B, 3, 96, 96), float32 in [0, 1].
    Output : Tensor of shape (B, feature_dim).
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # 96 -> 23 -> 10 -> 8  ;  64*8*8 = 4096
        self.linear = nn.Sequential(nn.Linear(64 * 8 * 8, feature_dim), nn.ReLU(inplace=True))
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.conv(x))


# ---------------------------------------------------------------------------
# SAC actor / critic with CNN trunk
# ---------------------------------------------------------------------------


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SACActorCNN(nn.Module):
    def __init__(self, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 feature_dim: int = 256) -> None:
        super().__init__()
        self.encoder = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.mean = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Linear(feature_dim, action_dim)
        # action rescaling from tanh output [-1, 1] to env range
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        # squashed log-prob correction
        log_prob = normal.log_prob(x) - torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        # rescale action
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        action = y * scale + bias
        mean_action = torch.tanh(mean) * scale + bias
        return action, log_prob, mean_action


class SACCriticCNN(nn.Module):
    """Twin Q-networks, each with its own CNN trunk."""

    def __init__(self, action_dim: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.encoder1 = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
        self.encoder2 = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = self.encoder1(obs)
        h2 = self.encoder2(obs)
        q1 = self.q1(torch.cat([h1, act], dim=-1))
        q2 = self.q2(torch.cat([h2, act], dim=-1))
        return q1, q2


# ---------------------------------------------------------------------------
# Replay buffer (uint8 storage)
# ---------------------------------------------------------------------------


class ImageReplayBuffer:
    """Stores obs as uint8 (C, H, W) to save memory; converts on sample."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int], action_dim: int,
                 device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        # Store as uint8 in [0, 255] (we re-quantize float32 obs).
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float,
            next_obs: np.ndarray, done: bool) -> None:
        i = self.idx
        # obs is float32 in [0, 1], CHW. Re-quantize to uint8.
        self.obs[i] = np.clip(obs * 255.0, 0, 255).astype(np.uint8)
        self.next_obs[i] = np.clip(next_obs * 255.0, 0, 255).astype(np.uint8)
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        obs = torch.from_numpy(self.obs[idx]).float().div_(255.0).to(self.device)
        next_obs = torch.from_numpy(self.next_obs[idx]).float().div_(255.0).to(self.device)
        actions = torch.from_numpy(self.actions[idx]).to(self.device)
        rewards = torch.from_numpy(self.rewards[idx]).to(self.device)
        dones = torch.from_numpy(self.dones[idx]).to(self.device)
        return {"obs": obs, "actions": actions, "rewards": rewards,
                "next_obs": next_obs, "dones": dones}


# ---------------------------------------------------------------------------
# SAC agent
# ---------------------------------------------------------------------------


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 64
    feature_dim: int = 256
    target_entropy: float | None = None
    init_alpha: float = 0.2


class SACAgentCNN:
    def __init__(self, env: gym.Env, device: torch.device, cfg: SACConfig) -> None:
        assert isinstance(env.action_space, spaces.Box), "continuous only"
        self.device = device
        self.cfg = cfg
        action_dim = env.action_space.shape[0]
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.actor = SACActorCNN(action_dim, self.action_low, self.action_high,
                                 feature_dim=cfg.feature_dim).to(device)
        self.critic = SACCriticCNN(action_dim, feature_dim=cfg.feature_dim).to(device)
        self.critic_target = SACCriticCNN(action_dim, feature_dim=cfg.feature_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.target_entropy = (
            -float(action_dim) if cfg.target_entropy is None else cfg.target_entropy
        )
        self.log_alpha = torch.tensor(np.log(cfg.init_alpha), dtype=torch.float32,
                                      device=device, requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, _, mean_action = self.actor.sample(obs_t)
        out = mean_action if deterministic else action
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, act, rew, next_obs, done = (batch["obs"], batch["actions"], batch["rewards"],
                                         batch["next_obs"], batch["dones"])
        # critic update
        with torch.no_grad():
            next_action, next_logp, _ = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_action)
            tq = torch.min(tq1, tq2) - self.alpha * next_logp
            target = rew.unsqueeze(-1) + self.cfg.gamma * (1.0 - done.unsqueeze(-1)) * tq
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # actor update
        new_action, logp, _ = self.actor.sample(obs)
        q1_pi, q2_pi = self.critic(obs, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * logp - q_pi).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # alpha update
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # polyak target update
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {"critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "alpha": float(self.alpha.detach().cpu())}


# ---------------------------------------------------------------------------
# Training / evaluation loop helpers
# ---------------------------------------------------------------------------


def evaluate(agent: SACAgentCNN, env: gym.Env, n_episodes: int) -> list[float]:
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = bool(terminated or truncated)
        returns.append(ep_ret)
    return returns


def train_sac_cnn(
    *,
    seed: int,
    max_steps: int,
    warmup_steps: int,
    eval_frequency: int,
    eval_episodes: int,
    device: torch.device,
    buffer_capacity: int = 10_000,
    learn_frequency: int = 1,
    on_eval: Any = None,
    should_stop: Any = None,
) -> dict[str, Any]:
    """Run a single SAC-CNN training loop. Returns a dict of summary info.

    Calls ``on_eval(eval_step, returns)`` after every evaluation block.
    Calls ``should_stop()`` between steps; stops cleanly if it returns True.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_carracing_cnn_env(seed)
    eval_env = make_carracing_cnn_env(seed + 10_000)

    cfg = SACConfig()
    agent = SACAgentCNN(env, device, cfg)
    obs_shape = env.observation_space.shape  # (3, 96, 96)
    action_dim = env.action_space.shape[0]
    buffer = ImageReplayBuffer(min(buffer_capacity, max(max_steps, cfg.batch_size)),
                               obs_shape, action_dim, device)

    obs, _ = env.reset()
    n_eval_blocks = 0
    train_step = 0
    stopped_early = False
    while train_step < max_steps:
        if should_stop is not None and should_stop():
            stopped_early = True
            break
        if train_step < warmup_steps or len(buffer) < cfg.batch_size:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buffer.add(obs, action, float(reward), next_obs, done and not truncated)
        obs = next_obs
        if done:
            obs, _ = env.reset()
        train_step += 1

        if (train_step >= warmup_steps and len(buffer) >= cfg.batch_size
                and train_step % learn_frequency == 0):
            agent.update(buffer.sample(cfg.batch_size))

        if eval_frequency > 0 and train_step % eval_frequency == 0:
            returns = evaluate(agent, eval_env, eval_episodes)
            n_eval_blocks += 1
            if on_eval is not None:
                on_eval(train_step, returns)

    env.close()
    eval_env.close()
    return {"train_steps_completed": train_step,
            "n_eval_blocks": n_eval_blocks,
            "stopped_early": stopped_early}


# ---------------------------------------------------------------------------
# TD3-CNN
# ---------------------------------------------------------------------------
#
# Reuses ``CarRacingCNNWrapper``, ``CNNFeatureExtractor`` and
# ``ImageReplayBuffer``. Each network owns its own CNN trunk (no parameter
# sharing across actor / critic1 / critic2), matching the SAC-CNN baseline.


class TD3ActorCNN(nn.Module):
    """Deterministic actor: CNN -> MLP -> tanh -> rescale to env range."""

    def __init__(self, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 feature_dim: int = 256) -> None:
        super().__init__()
        self.encoder = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, action_dim), nn.Tanh(),
        )
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        y = self.head(self.encoder(obs))  # in [-1, 1]
        scale = (self.action_high - self.action_low) / 2.0
        bias = (self.action_high + self.action_low) / 2.0
        return y * scale + bias


class TD3SingleCriticCNN(nn.Module):
    """One Q-network with its own CNN trunk."""

    def __init__(self, action_dim: int, feature_dim: int = 256) -> None:
        super().__init__()
        self.encoder = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.q = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([self.encoder(obs), act], dim=-1))


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 64
    feature_dim: int = 256
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2


class TD3AgentCNN:
    def __init__(self, env: gym.Env, device: torch.device, cfg: TD3Config) -> None:
        assert isinstance(env.action_space, spaces.Box), "continuous only"
        self.device = device
        self.cfg = cfg
        action_dim = env.action_space.shape[0]
        self.action_dim = action_dim
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self._action_low_t = torch.as_tensor(self.action_low, dtype=torch.float32, device=device)
        self._action_high_t = torch.as_tensor(self.action_high, dtype=torch.float32, device=device)
        self._action_range_half_t = (self._action_high_t - self._action_low_t) / 2.0

        self.actor = TD3ActorCNN(action_dim, self.action_low, self.action_high,
                                 feature_dim=cfg.feature_dim).to(device)
        self.actor_target = TD3ActorCNN(action_dim, self.action_low, self.action_high,
                                        feature_dim=cfg.feature_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic1 = TD3SingleCriticCNN(action_dim, feature_dim=cfg.feature_dim).to(device)
        self.critic2 = TD3SingleCriticCNN(action_dim, feature_dim=cfg.feature_dim).to(device)
        self.critic1_target = TD3SingleCriticCNN(action_dim, feature_dim=cfg.feature_dim).to(device)
        self.critic2_target = TD3SingleCriticCNN(action_dim, feature_dim=cfg.feature_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        for p in list(self.critic1_target.parameters()) + list(self.critic2_target.parameters()):
            p.requires_grad = False

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )
        self._update_count = 0

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action = self.actor(obs_t).squeeze(0)
        if not deterministic:
            noise = torch.randn_like(action) * (self.cfg.exploration_noise * self._action_range_half_t)
            action = action + noise
        action = torch.clamp(action, self._action_low_t, self._action_high_t)
        return action.cpu().numpy().astype(np.float32)

    def _soft_update(self, online: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for p, tp in zip(online.parameters(), target.parameters()):
                tp.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs, act, rew, next_obs, done = (batch["obs"], batch["actions"], batch["rewards"],
                                         batch["next_obs"], batch["dones"])

        # ---- target policy smoothing ----
        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            noise_std = self.cfg.target_policy_noise * self._action_range_half_t
            noise_clip = self.cfg.target_noise_clip * self._action_range_half_t
            noise = torch.randn_like(next_action) * noise_std
            noise = torch.clamp(noise, -noise_clip, noise_clip)
            next_action = torch.clamp(next_action + noise, self._action_low_t, self._action_high_t)

            tq1 = self.critic1_target(next_obs, next_action)
            tq2 = self.critic2_target(next_obs, next_action)
            tq = torch.min(tq1, tq2)
            target = rew.unsqueeze(-1) + self.cfg.gamma * (1.0 - done.unsqueeze(-1)) * tq

        q1 = self.critic1(obs, act)
        q2 = self.critic2(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        info = {"critic_loss": float(critic_loss.detach().cpu()), "actor_loss": float("nan")}
        self._update_count += 1
        if self._update_count % self.cfg.policy_delay == 0:
            actor_action = self.actor(obs)
            actor_loss = -self.critic1(obs, actor_action).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            info["actor_loss"] = float(actor_loss.detach().cpu())
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)
        return info


def train_td3_cnn(
    *,
    seed: int,
    max_steps: int,
    warmup_steps: int,
    eval_frequency: int,
    eval_episodes: int,
    device: torch.device,
    buffer_capacity: int = 10_000,
    learn_frequency: int = 1,
    on_eval: Any = None,
    should_stop: Any = None,
) -> dict[str, Any]:
    """Run a single TD3-CNN training loop on CarRacing-v3."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_carracing_cnn_env(seed)
    eval_env = make_carracing_cnn_env(seed + 10_000)

    cfg = TD3Config()
    agent = TD3AgentCNN(env, device, cfg)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    buffer = ImageReplayBuffer(min(buffer_capacity, max(max_steps, cfg.batch_size)),
                               obs_shape, action_dim, device)

    obs, _ = env.reset()
    n_eval_blocks = 0
    train_step = 0
    stopped_early = False
    while train_step < max_steps:
        if should_stop is not None and should_stop():
            stopped_early = True
            break
        if train_step < warmup_steps or len(buffer) < cfg.batch_size:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        buffer.add(obs, action, float(reward), next_obs, done and not truncated)
        obs = next_obs
        if done:
            obs, _ = env.reset()
        train_step += 1

        if (train_step >= warmup_steps and len(buffer) >= cfg.batch_size
                and train_step % learn_frequency == 0):
            agent.update(buffer.sample(cfg.batch_size))

        if eval_frequency > 0 and train_step % eval_frequency == 0:
            returns = evaluate(agent, eval_env, eval_episodes)
            n_eval_blocks += 1
            if on_eval is not None:
                on_eval(train_step, returns)

    env.close()
    eval_env.close()
    return {"train_steps_completed": train_step,
            "n_eval_blocks": n_eval_blocks,
            "stopped_early": stopped_early}


# ---------------------------------------------------------------------------
# PPO-CNN
# ---------------------------------------------------------------------------
#
# Architecture choice: a *shared* CNN trunk feeds both the actor head
# (Gaussian mean + state-independent log_std parameter) and the critic head
# (scalar V(s)). This is the standard PPO-image pattern and saves compute
# on what is by far the most expensive part of the network. The actor/critic
# MLP heads on top of the shared features are separate.
#
# Action mapping: a Gaussian policy over the *unsquashed* action followed by
# clipping to the environment's action bounds before stepping. Log-prob is
# computed on the unclipped Gaussian sample (a common, well-documented PPO
# convention -- see e.g. Stable-Baselines3 PPO with Box action spaces). This
# avoids the tanh-squash log-prob correction needed by SAC and keeps the
# clipped policy ratio interpretable.
#
# The PPO update implements: GAE-lambda advantages, advantage normalisation,
# clipped surrogate policy loss, MSE value loss, entropy bonus, and gradient
# clipping. Updates run for ``update_epochs`` epochs over the rollout buffer,
# split into minibatches of size ``minibatch_size``.


class PPOActorCriticCNN(nn.Module):
    """Shared-trunk CNN actor-critic for continuous CarRacing."""

    def __init__(self, action_dim: int, action_low: np.ndarray, action_high: np.ndarray,
                 feature_dim: int = 256, log_std_init: float = -0.5) -> None:
        super().__init__()
        self.encoder = CNNFeatureExtractor(in_channels=3, feature_dim=feature_dim)
        self.actor_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256, action_dim),
        )
        # State-independent log_std (standard PPO-continuous parameterisation).
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        self.critic_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256, 1),
        )
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = self.encoder(obs)
        mean = self.actor_head(feats)
        value = self.critic_head(feats).squeeze(-1)
        log_std = self.log_std.expand_as(mean).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, value

    def dist(self, obs: torch.Tensor) -> tuple[torch.distributions.Normal, torch.Tensor]:
        mean, log_std, value = self.forward(obs)
        std = log_std.exp()
        return torch.distributions.Normal(mean, std), value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (clipped_action_for_env, log_prob_of_unclipped, value)."""
        dist, value = self.dist(obs)
        if deterministic:
            raw = dist.mean
        else:
            raw = dist.sample()
        log_prob = dist.log_prob(raw).sum(-1)
        clipped = torch.clamp(raw, self.action_low, self.action_high)
        return clipped, log_prob, value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    rollout_steps: int = 256
    update_epochs: int = 4
    minibatch_size: int = 64
    feature_dim: int = 256


class PPORolloutBuffer:
    """Fixed-size on-policy buffer storing observations as uint8 (HWC)."""

    def __init__(self, capacity: int, obs_shape: tuple[int, int, int],
                 action_dim: int, device: torch.device) -> None:
        c, h, w = obs_shape  # CHW
        self.capacity = capacity
        self.device = device
        self.obs = np.zeros((capacity, h, w, c), dtype=np.uint8)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.values = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0

    def add(self, obs_chw: np.ndarray, action: np.ndarray, log_prob: float,
            reward: float, done: bool, value: float) -> None:
        i = self.ptr
        # obs_chw is float32 in [0,1] CHW; convert back to uint8 HWC for storage.
        hwc_uint8 = np.clip(obs_chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
        self.obs[i] = hwc_uint8
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.values[i] = value
        self.ptr += 1

    def reset(self) -> None:
        self.ptr = 0

    def is_full(self) -> bool:
        return self.ptr >= self.capacity

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float
                    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.ptr
        advantages = np.zeros((n,), dtype=np.float32)
        gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + self.values[:n]
        return advantages, returns

    def to_tensors(self, advantages: np.ndarray, returns: np.ndarray
                   ) -> dict[str, torch.Tensor]:
        n = self.ptr
        # uint8 HWC -> float32 CHW in [0,1]
        obs_f = (self.obs[:n].astype(np.float32) / 255.0).transpose(0, 3, 1, 2)
        return {
            "obs": torch.from_numpy(obs_f).to(self.device),
            "actions": torch.from_numpy(self.actions[:n]).to(self.device),
            "log_probs": torch.from_numpy(self.log_probs[:n]).to(self.device),
            "advantages": torch.from_numpy(advantages).to(self.device),
            "returns": torch.from_numpy(returns).to(self.device),
            "values": torch.from_numpy(self.values[:n]).to(self.device),
        }


class PPOAgentCNN:
    def __init__(self, env: gym.Env, device: torch.device, cfg: PPOConfig) -> None:
        assert isinstance(env.action_space, spaces.Box), "continuous only"
        self.device = device
        self.cfg = cfg
        action_dim = env.action_space.shape[0]
        self.action_dim = action_dim
        self.net = PPOActorCriticCNN(
            action_dim, env.action_space.low, env.action_space.high,
            feature_dim=cfg.feature_dim,
        ).to(device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.learning_rate)

    @torch.no_grad()
    def select_action(self, obs: np.ndarray, deterministic: bool = False
                      ) -> tuple[np.ndarray, float, float]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        action, log_prob, value = self.net.act(obs_t, deterministic=deterministic)
        return (action.squeeze(0).cpu().numpy().astype(np.float32),
                float(log_prob.item()), float(value.item()))

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        _, _, v = self.net(obs_t)
        return float(v.item())

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        obs = batch["obs"]
        actions = batch["actions"]
        old_log_probs = batch["log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Normalise advantages across the full rollout.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        mb_size = max(1, min(cfg.minibatch_size, n))
        last_info = {"policy_loss": 0.0, "value_loss": 0.0,
                     "entropy": 0.0, "approx_kl": 0.0}
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start:start + mb_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)
                mb_obs = obs.index_select(0, mb_t)
                mb_actions = actions.index_select(0, mb_t)
                mb_old_lp = old_log_probs.index_select(0, mb_t)
                mb_adv = advantages.index_select(0, mb_t)
                mb_ret = returns.index_select(0, mb_t)

                dist, value = self.net.dist(mb_obs)
                new_log_probs = dist.log_prob(mb_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, mb_ret)
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.optim.step()

                with torch.no_grad():
                    approx_kl = (mb_old_lp - new_log_probs).mean().item()
                last_info = {
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "value_loss": float(value_loss.detach().cpu()),
                    "entropy": float(entropy.detach().cpu()),
                    "approx_kl": float(approx_kl),
                }
        return last_info


def _ppo_evaluate(agent: PPOAgentCNN, env: gym.Env, n_episodes: int) -> list[float]:
    returns: list[float] = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += float(reward)
            done = bool(terminated or truncated)
        returns.append(ep_ret)
    return returns


def train_ppo_cnn(
    *,
    seed: int,
    max_steps: int,
    warmup_steps: int = 0,           # ignored: PPO is on-policy, no warmup
    eval_frequency: int,
    eval_episodes: int,
    device: torch.device,
    buffer_capacity: int = 0,        # ignored: PPO uses an on-policy rollout buffer
    learn_frequency: int = 1,        # ignored: PPO updates after each rollout
    on_eval: Any = None,
    should_stop: Any = None,
) -> dict[str, Any]:
    """Run a single PPO-CNN training loop on CarRacing-v3."""
    del warmup_steps, buffer_capacity, learn_frequency  # explicit: not used

    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_carracing_cnn_env(seed)
    eval_env = make_carracing_cnn_env(seed + 10_000)

    cfg = PPOConfig()
    # Auto-shrink rollout / minibatch for very small smoke runs so that
    # eval_frequency can still trigger evaluations and minibatch_size
    # never exceeds the rollout buffer.
    cfg.rollout_steps = max(1, min(cfg.rollout_steps, max_steps))
    if eval_frequency > 0:
        cfg.rollout_steps = max(1, min(cfg.rollout_steps, eval_frequency))
    cfg.minibatch_size = max(1, min(cfg.minibatch_size, cfg.rollout_steps))

    agent = PPOAgentCNN(env, device, cfg)
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    buffer = PPORolloutBuffer(cfg.rollout_steps, obs_shape, action_dim, device)

    obs, _ = env.reset()
    n_eval_blocks = 0
    train_step = 0
    stopped_early = False
    while train_step < max_steps:
        if should_stop is not None and should_stop():
            stopped_early = True
            break

        action, log_prob, value = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        # ``done`` here is used as a non-terminal mask in GAE; treat truncation
        # like SAC/TD3 do (still terminal for episode-return purposes).
        buffer.add(obs, action, log_prob, float(reward), done, value)
        obs = next_obs
        if done:
            obs, _ = env.reset()
        train_step += 1

        if buffer.is_full():
            last_value = 0.0 if done else agent.value(obs)
            advantages, returns = buffer.compute_gae(last_value, cfg.gamma, cfg.gae_lambda)
            agent.update(buffer.to_tensors(advantages, returns))
            buffer.reset()

        if eval_frequency > 0 and train_step % eval_frequency == 0:
            returns_eval = _ppo_evaluate(agent, eval_env, eval_episodes)
            n_eval_blocks += 1
            if on_eval is not None:
                on_eval(train_step, returns_eval)

    env.close()
    eval_env.close()
    return {"train_steps_completed": train_step,
            "n_eval_blocks": n_eval_blocks,
            "stopped_early": stopped_early}
