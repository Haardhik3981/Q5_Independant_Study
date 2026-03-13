import os
import time
import argparse
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# Utilities
# ============================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def to_u8(o):
    return np.asarray(o, dtype=np.uint8)


def obs84_to_t(obs_u8: np.ndarray, device: str) -> torch.Tensor:
    # (B,4,84,84) uint8 -> float [0,1]
    x = obs_u8.astype(np.float32) / 255.0
    return torch.from_numpy(x).to(device)


# ============================================================
# Common CarRacing env (84x84 grayscale, 4-stack)
# Used by: PPO, SAC, A2C_continuous
# ============================================================
def make_env_84(seed: int, record_dir: Optional[str] = None):
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.reset(seed=seed)

    if record_dir is not None:
        ensure_dir(record_dir)
        # Record ALL episodes (episode_trigger always True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=record_dir,
            episode_trigger=lambda ep: True,
            name_prefix="eval",
        )
    return env


# ============================================================
# A2C DISCRETE env processing (96x96 custom grayscale + custom stack)
# ============================================================
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


def rgb_to_gray_96(obs: np.ndarray) -> np.ndarray:
    # obs: (96,96,3) uint8 -> (96,96) float32 in [0,1]
    gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
    return (gray / 255.0).astype(np.float32)


class FrameStack4_96:
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
        # (4,96,96) float32 [0,1]
        return np.stack(self.frames, axis=0)


def make_env_a2c_discrete(seed: int, record_dir: Optional[str] = None):
    # Keep native frames (96x96) like your a2c_discrete.py
    env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
    env.reset(seed=seed)
    if record_dir is not None:
        ensure_dir(record_dir)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=record_dir,
            episode_trigger=lambda ep: True,
            name_prefix="eval",
        )
    return env


# ============================================================
# Model definitions
# ============================================================

# SAC OLD
class SAC_CNNEncoder_Old(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, 4, 84, 84)).shape[1]
        self.out_dim = n

    def forward(self, x):
        return self.net(x)


class SAC_Actor_Old(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.enc = SAC_CNNEncoder_Old()
        self.fc = nn.Sequential(nn.Linear(self.enc.out_dim, 512), nn.ReLU())
        self.mu = nn.Linear(512, action_dim)
        self.logstd = nn.Linear(512, action_dim)

    def forward(self, x):
        z = self.fc(self.enc(x))
        mu = self.mu(z)
        logstd = self.logstd(z)
        logstd = torch.clamp(logstd, -5, 2)
        return mu, logstd


# SAC REFINED
class SAC_CNNEncoder_Refined(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 1, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, 4, 84, 84)).shape[1]
        self.out_dim = n
        self.norm = nn.LayerNorm(n)

    def forward(self, x):
        return self.norm(self.net(x))


class SAC_Actor_Refined(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.enc = SAC_CNNEncoder_Refined()
        self.fc = nn.Sequential(nn.Linear(self.enc.out_dim, 512), nn.ReLU())
        self.mu = nn.Linear(512, action_dim)
        self.logstd = nn.Linear(512, action_dim)

    def forward(self, x):
        z = self.fc(self.enc(x))
        mu = self.mu(z)
        logstd = self.logstd(z)
        logstd = torch.clamp(logstd, -5, 2)
        return mu, logstd


def squash_to_env_sac(a_tanh: torch.Tensor) -> torch.Tensor:
    steer = a_tanh[..., 0]                 # [-1,1]
    gas = (a_tanh[..., 1] + 1) * 0.5       # [0,1]
    brake = (a_tanh[..., 2] + 1) * 0.5     # [0,1]
    return torch.stack([steer, gas, brake], dim=-1)


@torch.no_grad()
def sac_actor_deterministic_action(actor: nn.Module, x: torch.Tensor) -> torch.Tensor:
    mu, _ = actor(x)
    a_tanh = torch.tanh(mu)
    return squash_to_env_sac(a_tanh)


# PPO
class PPO_AC(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, 4, 84, 84)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n, 512), nn.ReLU())
        self.mu = nn.Linear(512, 3)
        self.logstd = nn.Parameter(torch.zeros(3))
        self.v = nn.Linear(512, 1)

    def forward(self, x):
        z = self.fc(self.cnn(x))
        return self.mu(z), self.logstd.expand(x.size(0), 3), self.v(z).squeeze(-1)


@torch.no_grad()
def ppo_deterministic_action(net: PPO_AC, x: torch.Tensor) -> torch.Tensor:
    mu, _, _ = net(x)
    # deterministic == use mean as "u"
    u = mu
    steer = torch.tanh(u[..., 0])
    gas = torch.sigmoid(u[..., 1])
    brake = torch.sigmoid(u[..., 2])
    return torch.stack([steer, gas, brake], dim=-1)


# A2C continuous
class A2C_Continuous(nn.Module):
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
        self.actor_mean = nn.Linear(512, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        z = self.fc(self.conv(x))
        mean = self.actor_mean(z)
        logstd = self.actor_logstd.expand_as(mean)
        value = self.critic(z).squeeze(-1)
        return mean, logstd, value


@torch.no_grad()
def a2c_cont_deterministic_action(net: A2C_Continuous, x: torch.Tensor) -> torch.Tensor:
    mean, _, _ = net(x)
    raw = mean.squeeze(0)
    steer = torch.tanh(raw[0])
    gas = torch.sigmoid(raw[1])
    brake = torch.sigmoid(raw[2])
    return torch.stack([steer, gas, brake], dim=-1).unsqueeze(0)


# A2C discrete 
class A2C_Discrete_CNN(nn.Module):
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
        z = self.fc(self.backbone(x))
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value


@torch.no_grad()
def a2c_discrete_deterministic_action(net: A2C_Discrete_CNN, state_4x96: np.ndarray, device: str) -> np.ndarray:
    # state_4x96: (4,96,96) float32 [0,1]
    x = torch.from_numpy(state_4x96).unsqueeze(0).to(device)
    logits, _ = net(x)
    a_idx = torch.argmax(logits, dim=-1).item()
    return DISCRETE_ACTIONS[int(a_idx)]


# ============================================================
# Checkpoint auto-detection
# ============================================================
def detect_algo_and_arch(ckpt: dict, ckpt_path: str) -> Tuple[str, Optional[str]]:
    """
    Returns:
      algo_id in {"ppo", "a2c_discrete", "a2c_cont", "sac"}
      arch in {None, "old", "refined"}  (only meaningful for SAC)
    """
    # If you stored metadata, use it
    algo_id = ckpt.get("algo_id", None)
    arch = ckpt.get("arch", None)

    # Infer if missing
    if algo_id is None:
        # SAC-like checkpoints: have "actor"
        if "actor" in ckpt:
            algo_id = "sac"
        # PPO/A2C commonly: "state_dict"
        elif "state_dict" in ckpt:
            # Heuristic by filename / directory
            p = ckpt_path.lower()
            if "ppo" in p:
                algo_id = "ppo"
            elif "a2c_discrete" in p or "discrete" in p:
                algo_id = "a2c_discrete"
            elif "a2c_cont" in p or "continuous" in p:
                algo_id = "a2c_cont"
            else:
                # fallback: assume PPO-style network if 84x84 CNN weights exist
                algo_id = "ppo"
        else:
            raise ValueError("Unknown checkpoint format (expected keys like 'actor' or 'state_dict').")

    # SAC arch detection (prevents your old mismatch errors)
    if algo_id == "sac" and arch is None:
        sd = ckpt["actor"]
        # refined adds LayerNorm AND an extra conv (64->64 1x1)
        # these show up as keys like: enc.norm.weight, enc.norm.bias, enc.net.6.weight, ...
        if any(k.startswith("enc.norm.") for k in sd.keys()):
            arch = "refined"
        else:
            arch = "old"

    return algo_id, arch


# ============================================================
# Main eval
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--run_dir", type=str, default="Task9_results", help="Base dir to store results")
    ap.add_argument("--video", action="store_true", help="Record videos for ALL episodes")
    ap.add_argument("--device", type=str, default=None, help="cuda / cpu (auto if not set)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path, map_location=device)

    algo_id, arch = detect_algo_and_arch(ckpt, ckpt_path)

    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{algo_id}_{arch or 'na'}_{ckpt_stem}_{ts}"
    out_dir = ensure_dir(os.path.join(args.run_dir, run_name))
    video_dir = ensure_dir(os.path.join(out_dir, "videos")) if args.video else None
    tb_dir = ensure_dir(os.path.join(out_dir, "runs"))

    print("======================================")
    print("Unified Eval Runner")
    print(f"ckpt:      {ckpt_path}")
    print(f"algo_id:   {algo_id}")
    print(f"arch:      {arch}")
    print(f"device:    {device}")
    print(f"out_dir:   {out_dir}")
    print(f"video_dir: {video_dir}")
    print(f"tb_dir:    {tb_dir}")
    print("======================================")

    writer = SummaryWriter(tb_dir)

    rewards = []
    lengths = []

    # SAC 
    if algo_id == "sac":
        if arch == "refined":
            actor = SAC_Actor_Refined().to(device)
        else:
            actor = SAC_Actor_Old().to(device)

        actor.load_state_dict(ckpt["actor"])
        actor.eval()

        env = make_env_84(args.seed, record_dir=video_dir)

        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            ep_rew = 0.0
            ep_len = 0

            while not done:
                x = obs84_to_t(to_u8(obs)[None, ...], device)
                a = sac_actor_deterministic_action(actor, x).squeeze(0).cpu().numpy().astype(np.float32)
                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_rew += float(r)
                ep_len += 1

            rewards.append(ep_rew)
            lengths.append(ep_len)
            print(f"ep={ep:02d} reward={ep_rew:8.2f} len={ep_len}")
            writer.add_scalar("eval/episode_reward", ep_rew, ep)
            writer.add_scalar("eval/episode_length", ep_len, ep)

        env.close()

    # PPO 
    elif algo_id == "ppo":
        net = PPO_AC().to(device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        net.load_state_dict(sd)
        net.eval()

        env = make_env_84(args.seed, record_dir=video_dir)

        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            ep_rew = 0.0
            ep_len = 0

            while not done:
                x = obs84_to_t(to_u8(obs)[None, ...], device)
                a = ppo_deterministic_action(net, x).squeeze(0).cpu().numpy().astype(np.float32)
                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_rew += float(r)
                ep_len += 1

            rewards.append(ep_rew)
            lengths.append(ep_len)
            print(f"ep={ep:02d} reward={ep_rew:8.2f} len={ep_len}")
            writer.add_scalar("eval/episode_reward", ep_rew, ep)
            writer.add_scalar("eval/episode_length", ep_len, ep)

        env.close()

    # A2C continuous 
    elif algo_id == "a2c_cont":
        net = A2C_Continuous().to(device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        net.load_state_dict(sd)
        net.eval()

        env = make_env_84(args.seed, record_dir=video_dir)

        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            done = False
            ep_rew = 0.0
            ep_len = 0

            while not done:
                x = obs84_to_t(to_u8(obs)[None, ...], device)
                a = a2c_cont_deterministic_action(net, x).squeeze(0).cpu().numpy().astype(np.float32)
                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_rew += float(r)
                ep_len += 1

            rewards.append(ep_rew)
            lengths.append(ep_len)
            print(f"ep={ep:02d} reward={ep_rew:8.2f} len={ep_len}")
            writer.add_scalar("eval/episode_reward", ep_rew, ep)
            writer.add_scalar("eval/episode_length", ep_len, ep)

        env.close()

    # A2C discrete
    elif algo_id == "a2c_discrete":
        net = A2C_Discrete_CNN(n_actions=len(DISCRETE_ACTIONS), in_channels=4).to(device)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        net.load_state_dict(sd)
        net.eval()

        env = make_env_a2c_discrete(args.seed, record_dir=video_dir)

        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)

            fs = FrameStack4_96(4)
            state = fs.reset(rgb_to_gray_96(obs))

            done = False
            ep_rew = 0.0
            ep_len = 0

            while not done:
                a = a2c_discrete_deterministic_action(net, state, device)
                obs, r, term, trunc, _ = env.step(a)
                done = term or trunc

                state = fs.step(rgb_to_gray_96(obs))
                ep_rew += float(r)
                ep_len += 1
                if ep_len >= 1000:
                    break

            rewards.append(ep_rew)
            lengths.append(ep_len)
            print(f"ep={ep:02d} reward={ep_rew:8.2f} len={ep_len}")
            writer.add_scalar("eval/episode_reward", ep_rew, ep)
            writer.add_scalar("eval/episode_length", ep_len, ep)

        env.close()

    else:
        raise ValueError(f"Unsupported algo_id: {algo_id}")

    rewards_np = np.array(rewards, dtype=np.float32)
    lengths_np = np.array(lengths, dtype=np.int32)

    mean_r = float(rewards_np.mean()) if len(rewards_np) else 0.0
    std_r = float(rewards_np.std()) if len(rewards_np) else 0.0
    mean_l = float(lengths_np.mean()) if len(lengths_np) else 0.0
    std_l = float(lengths_np.std()) if len(lengths_np) else 0.0

    print("\n==== EVAL SUMMARY ====")
    print(f"algo_id:      {algo_id}")
    print(f"arch:         {arch}")
    print(f"episodes:     {args.episodes}")
    print(f"mean_reward:  {mean_r:.2f}  std: {std_r:.2f}")
    print(f"mean_length:  {mean_l:.1f} std: {std_l:.1f}")
    print(f"saved_to:     {out_dir}")

    writer.add_scalar("eval/mean_reward", mean_r, 0)
    writer.add_scalar("eval/std_reward", std_r, 0)
    writer.add_scalar("eval/mean_length", mean_l, 0)
    writer.add_scalar("eval/std_length", std_l, 0)
    writer.add_text("meta/algo_id", str(algo_id))
    writer.add_text("meta/arch", str(arch))
    writer.add_text("meta/ckpt", str(ckpt_path))
    writer.close()


if __name__ == "__main__":
    main()