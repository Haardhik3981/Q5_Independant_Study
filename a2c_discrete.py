import os
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


# 9 discrete actions -> continuous [steer, gas, brake]
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),    # 0 coast
    np.array([0.0, 1.0, 0.0], dtype=np.float32),    # 1 gas
    np.array([0.0, 0.0, 0.8], dtype=np.float32),    # 2 brake
    np.array([-0.8, 0.0, 0.0], dtype=np.float32),   # 3 hard left
    np.array([+0.8, 0.0, 0.0], dtype=np.float32),   # 4 hard right
    np.array([-0.4, 1.0, 0.0], dtype=np.float32),   # 5 left + gas
    np.array([+0.4, 1.0, 0.0], dtype=np.float32),   # 6 right + gas
    np.array([-0.4, 0.0, 0.0], dtype=np.float32),   # 7 gentle left
    np.array([+0.4, 0.0, 0.0], dtype=np.float32),   # 8 gentle right
]

def rgb_to_gray(obs: np.ndarray) -> np.ndarray:
    # obs: (96,96,3) uint8 -> (96,96) float32 in [0,1]
    gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
    return (gray / 255.0).astype(np.float32)

class FrameStack4:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, frame2d: np.ndarray) -> np.ndarray:
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(frame2d)
        return self._stack()

    def step(self, frame2d: np.ndarray) -> np.ndarray:
        self.frames.append(frame2d)
        return self._stack()

    def _stack(self) -> np.ndarray:
        # (k,96,96)
        return np.stack(self.frames, axis=0)

class ActorCriticCNN(nn.Module):
    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 96, 96)
            n_flat = self.backbone(dummy).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU())
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        z = self.backbone(x)
        z = self.fc(z)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

@dataclass
class Config:
    env_id: str = "CarRacing-v3"
    total_episodes: int = 400
    max_steps: int = 1000
    gamma: float = 0.99
    lr: float = 1e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    logdir: str = "runs/a2c_discrete_framestack"
    save_path: str = "models/a2c_discrete_framestack.pt"
    skip_zoom_steps: int = 50

def compute_returns(rewards, gamma):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

def main():
    cfg = Config()
    os.makedirs(cfg.logdir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    env = gym.make(cfg.env_id, continuous=True)
    model = ActorCriticCNN(n_actions=len(DISCRETE_ACTIONS), in_channels=4).to(cfg.device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    writer = SummaryWriter(cfg.logdir)

    reward_window = deque(maxlen=20)
    global_step = 0

    print(f"Device: {cfg.device}")
    print(f"Logging to: {cfg.logdir}")

    for ep in range(1, cfg.total_episodes + 1):
        obs, info = env.reset(seed=cfg.seed + ep)

        # skip zoom-in frames by taking no-op actions
        for _ in range(cfg.skip_zoom_steps):
            obs, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            if terminated or truncated:
                obs, info = env.reset(seed=cfg.seed + ep)

        fs = FrameStack4(4)
        state = fs.reset(rgb_to_gray(obs))

        logps, values, entropies, rewards = [], [], [], []
        ep_reward = 0.0

        for t in range(cfg.max_steps):
            global_step += 1

            x = torch.from_numpy(state).unsqueeze(0).to(cfg.device)  # (1,4,96,96)
            logits, value = model(x)
            dist = torch.distributions.Categorical(logits=logits)
            a_idx = dist.sample()

            logps.append(dist.log_prob(a_idx).squeeze(0))
            values.append(value.squeeze(0))
            entropies.append(dist.entropy().squeeze(0))

            action = DISCRETE_ACTIONS[int(a_idx.item())]
            obs, reward, terminated, truncated, info = env.step(action)

            state = fs.step(rgb_to_gray(obs))

            rewards.append(float(reward))
            ep_reward += float(reward)

            if terminated or truncated:
                break

        returns = compute_returns(rewards, cfg.gamma).to(cfg.device)
        values_t = torch.stack(values)
        logp_t = torch.stack(logps)
        ent_t = torch.stack(entropies)

        adv = returns - values_t.detach()
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_loss = -(logp_t * adv).mean()
        critic_loss = (returns - values_t).pow(2).mean()
        entropy = ent_t.mean()

        loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        reward_window.append(ep_reward)

        writer.add_scalar("episode/reward", ep_reward, ep)
        writer.add_scalar("episode/length", len(rewards), ep)
        writer.add_scalar("train/actor_loss", actor_loss.item(), ep)
        writer.add_scalar("train/critic_loss", critic_loss.item(), ep)
        writer.add_scalar("train/entropy", entropy.item(), ep)
        writer.add_scalar("train/reward_20ep_mean", float(np.mean(reward_window)), ep)

        if ep % 10 == 0:
            print(f"ep={ep:4d} | steps={global_step:7d} | ep_rew={ep_reward:8.2f} | mean20={np.mean(reward_window):8.2f}")

    torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, cfg.save_path)
    print(f"Saved model to: {cfg.save_path}")

    env.close()
    writer.close()
    print("Training done.")

if __name__ == "__main__":
    main()
