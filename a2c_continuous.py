import os
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# Environment factory: grayscale + resize + frame stack
# ============================================================
def make_env():
    env = gym.make("CarRacing-v3", continuous=True)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)  # (H,W)
    env = gym.wrappers.ResizeObservation(env, (84, 84))           # (84,84)
    env = gym.wrappers.FrameStackObservation(env, 4)              # (4,84,84)
    return env


def obs_to_tensor(obs) -> torch.Tensor:
    # obs: (4,84,84) -> float32 in [0,1]
    x = np.array(obs, copy=False).astype(np.float32) / 255.0
    return torch.from_numpy(x)


# ============================================================
# Actor–Critic Network (Continuous)
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, in_channels=4, action_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            n_flat = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU())

        # Actor
        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        z = self.conv(x)
        z = self.fc(z)
        mean = self.actor_mean(z)
        logstd = self.actor_logstd.expand_as(mean)
        value = self.critic(z).squeeze(-1)
        return mean, logstd, value


def squash_action(raw_action: torch.Tensor) -> np.ndarray:
    """
    raw_action: unconstrained R^3
    returns valid CarRacing action:
    steer in [-1,1], gas/brake in [0,1]
    """
    steer = torch.tanh(raw_action[0])
    gas = torch.sigmoid(raw_action[1])
    brake = torch.sigmoid(raw_action[2])
    return torch.stack([steer, gas, brake]).detach().cpu().numpy().astype(np.float32)


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    total_episodes: int = 200        # start small
    max_steps: int = 1000
    gamma: float = 0.99
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logdir: str = "runs/a2c_cont"
    save_path: str = "models/a2c_cont.pt"


def compute_returns(rewards, gamma):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


# ============================================================
# Training
# ============================================================
def main():
    cfg = Config()
    os.makedirs(cfg.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    env = make_env()
    model = ActorCritic().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(cfg.logdir)

    reward_window = deque(maxlen=20)
    global_step = 0

    print(f"Device: {cfg.device}")
    print(f"Logging to: {cfg.logdir}")

    for ep in range(1, cfg.total_episodes + 1):
        obs, info = env.reset(seed=cfg.seed + ep)

        logps, values, entropies, rewards = [], [], [], []
        ep_reward = 0.0

        for t in range(cfg.max_steps):
            global_step += 1

            x = obs_to_tensor(obs).unsqueeze(0).to(cfg.device)  # (1,4,84,84)
            mean, logstd, value = model(x)

            std = torch.exp(logstd)
            dist = torch.distributions.Normal(mean, std)

            raw_action = dist.sample().squeeze(0)
            logp = dist.log_prob(raw_action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            action = squash_action(raw_action)
            obs, reward, terminated, truncated, info = env.step(action)

            logps.append(logp)
            values.append(value.squeeze(0))
            entropies.append(entropy)
            rewards.append(float(reward))
            ep_reward += float(reward)

            if terminated or truncated:
                break

        returns = compute_returns(rewards, cfg.gamma).to(cfg.device)
        values_t = torch.stack(values)
        logp_t = torch.stack(logps)
        ent_t = torch.stack(entropies)

        advantages = returns - values_t.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(logp_t * advantages).mean()
        critic_loss = (returns - values_t).pow(2).mean()
        entropy = ent_t.mean()

        loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        reward_window.append(ep_reward)

        writer.add_scalar("episode/reward", ep_reward, ep)
        writer.add_scalar("episode/length", len(rewards), ep)
        writer.add_scalar("train/actor_loss", actor_loss.item(), ep)
        writer.add_scalar("train/critic_loss", critic_loss.item(), ep)
        writer.add_scalar("train/entropy", entropy.item(), ep)
        writer.add_scalar("train/reward_20ep_mean", float(np.mean(reward_window)), ep)

        if ep % 10 == 0:
            print(
                f"ep={ep:4d} | steps={global_step:7d} | "
                f"ep_rew={ep_reward:8.2f} | mean20={np.mean(reward_window):8.2f}"
            )

    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, cfg.save_path)
    print(f"Saved model to: {cfg.save_path}")

    env.close()
    writer.close()
    print("Training done.")


if __name__ == "__main__":
    main()
