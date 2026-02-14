import os, time, random
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


# -------------------------
# Config
# -------------------------
@dataclass
class Cfg:
    env_id: str = "CarRacing-v3"

    total_steps: int = 3_000_000
    start_steps: int = 10_000          # random actions to fill replay
    replay_size: int = 500_000
    batch_size: int = 256

    gamma: float = 0.99
    tau: float = 0.005                # target smoothing
    lr: float = 3e-4

    policy_delay: int = 1             # update policy every N Q updates (1 is fine)
    updates_per_step: int = 1         # can raise to 2 if training slow

    logdir: str = "runs/sac_minimal"
    model_dir: str = "models"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    save_every: int = 200_000


# -------------------------
# Env
# -------------------------
def make_env(seed: int):
    env = gym.make("CarRacing-v3", continuous=True)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.FrameStackObservation(env, 4)
    env.reset(seed=seed)
    return env


# -------------------------
# Replay Buffer (stores uint8 obs)
# -------------------------
class ReplayBuffer:
    def __init__(self, size: int, obs_shape=(4, 84, 84), action_dim=3):
        self.size = size
        self.obs = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.act = np.zeros((size, action_dim), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, o, a, r, no, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = no
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int):
        max_i = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_i, size=batch_size)
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )

    def __len__(self):
        return self.size if self.full else self.ptr


# -------------------------
# Networks
# -------------------------
def obs_to_t(obs_u8: np.ndarray, device: str) -> torch.Tensor:
    # obs_u8: (B,4,84,84) uint8 -> float [0,1]
    x = obs_u8.astype(np.float32) / 255.0
    return torch.from_numpy(x).to(device)


class CNNEncoder(nn.Module):
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


class Actor(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.enc = CNNEncoder()
        self.fc = nn.Sequential(nn.Linear(self.enc.out_dim, 512), nn.ReLU())
        self.mu = nn.Linear(512, action_dim)
        self.logstd = nn.Linear(512, action_dim)

    def forward(self, x):
        z = self.fc(self.enc(x))
        mu = self.mu(z)
        logstd = self.logstd(z)
        logstd = torch.clamp(logstd, -5, 2)  # standard SAC clamp
        return mu, logstd


class CriticQ(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()
        self.enc = CNNEncoder()
        self.fc = nn.Sequential(
            nn.Linear(self.enc.out_dim + action_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x, a):
        z = self.enc(x)
        za = torch.cat([z, a], dim=1)
        return self.fc(za).squeeze(-1)


def squash_to_env(raw_a: torch.Tensor) -> torch.Tensor:
    # raw_a is already squashed via tanh to [-1,1] per dim;
    # map for CarRacing: steer [-1,1], gas/brake [0,1]
    steer = raw_a[..., 0]                # [-1,1]
    gas = (raw_a[..., 1] + 1) * 0.5      # [-1,1] -> [0,1]
    brake = (raw_a[..., 2] + 1) * 0.5
    return torch.stack([steer, gas, brake], dim=-1)


def sample_action_and_logp(actor: Actor, x: torch.Tensor):
    """
    SAC uses tanh-squashed Gaussian:
      u ~ N(mu, std), a = tanh(u)
      logpi(a) = logpi(u) - sum log(1 - tanh(u)^2)
    We'll return:
      a_env (steer,gas,brake), logp, and raw tanh(a) in [-1,1]^3 for critics
    """
    mu, logstd = actor(x)
    std = torch.exp(logstd)
    dist = Normal(mu, std)
    u = dist.rsample()  # reparameterization
    a_tanh = torch.tanh(u)  # in [-1,1]
    # log prob correction
    logp_u = dist.log_prob(u).sum(-1)
    log_det = torch.log(1 - a_tanh * a_tanh + 1e-6).sum(-1)
    logp = logp_u - log_det
    a_env = squash_to_env(a_tanh)
    return a_env, a_tanh, logp


@torch.no_grad()
def actor_deterministic_action(actor: Actor, x: torch.Tensor):
    mu, _ = actor(x)
    a_tanh = torch.tanh(mu)
    return squash_to_env(a_tanh), a_tanh


def soft_update(targ: nn.Module, src: nn.Module, tau: float):
    for tp, sp in zip(targ.parameters(), src.parameters()):
        tp.data.mul_(1 - tau).add_(tau * sp.data)


# -------------------------
# Train
# -------------------------
def main():
    cfg = Cfg()
    os.makedirs(cfg.logdir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    writer = SummaryWriter(cfg.logdir)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = make_env(cfg.seed)
    obs, _ = env.reset(seed=cfg.seed)

    # Convert initial obs (LazyFrames/stack) to uint8 array with shape (4,84,84)
    def to_u8(o):
        return np.asarray(o, dtype=np.uint8)

    rb = ReplayBuffer(cfg.replay_size)

    actor = Actor().to(cfg.device)
    q1 = CriticQ().to(cfg.device)
    q2 = CriticQ().to(cfg.device)
    q1_t = CriticQ().to(cfg.device)
    q2_t = CriticQ().to(cfg.device)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())

    act_opt = optim.Adam(actor.parameters(), lr=cfg.lr)
    q1_opt = optim.Adam(q1.parameters(), lr=cfg.lr)
    q2_opt = optim.Adam(q2.parameters(), lr=cfg.lr)

    # Auto alpha tuning
    log_alpha = torch.tensor(0.0, requires_grad=True, device=cfg.device)
    alpha_opt = optim.Adam([log_alpha], lr=cfg.lr)
    target_entropy = -3.0  # -action_dim is a standard choice

    episode_reward = 0.0
    ep_len = 0
    ep = 0
    last_log = time.time()

    global_step = 0

    while global_step < cfg.total_steps:
        global_step += 1
        ep_len += 1

        # action selection
        if global_step < cfg.start_steps:
            # random action in env space
            a_env = np.array([
                np.random.uniform(-1, 1),   # steer
                np.random.uniform(0, 1),    # gas
                np.random.uniform(0, 1),    # brake
            ], dtype=np.float32)
        else:
            x = obs_to_t(to_u8(obs)[None, ...], cfg.device)
            with torch.no_grad():
                a_env_t, _ = actor_deterministic_action(actor, x)
            a_env = a_env_t.squeeze(0).cpu().numpy().astype(np.float32)

        next_obs, r, term, trunc, _ = env.step(a_env)
        done = float(term or trunc)

        rb.add(to_u8(obs), a_env, float(r), to_u8(next_obs), done)

        obs = next_obs
        episode_reward += float(r)

        if done:
            ep += 1
            writer.add_scalar("episode/reward", episode_reward, ep)
            writer.add_scalar("episode/length", ep_len, ep)
            obs, _ = env.reset(seed=cfg.seed + ep + 1)
            episode_reward = 0.0
            ep_len = 0

        # updates
        if len(rb) >= cfg.batch_size:
            for _ in range(cfg.updates_per_step):
                o, a_env_b, r_b, no, d_b = rb.sample(cfg.batch_size)
                o_t = obs_to_t(o, cfg.device)
                no_t = obs_to_t(no, cfg.device)

                a_env_t = torch.from_numpy(a_env_b).to(cfg.device)
                r_t = torch.from_numpy(r_b).to(cfg.device)
                d_t = torch.from_numpy(d_b).to(cfg.device)

                # Convert env action back to tanh action space for critics:
                # steer already [-1,1]
                # gas/brake [0,1] -> [-1,1]
                a_tanh = torch.stack([
                    a_env_t[:, 0],
                    a_env_t[:, 1] * 2 - 1,
                    a_env_t[:, 2] * 2 - 1,
                ], dim=1)

                with torch.no_grad():
                    # target actions from actor on next state
                    next_a_env, next_a_tanh, next_logp = sample_action_and_logp(actor, no_t)
                    alpha = log_alpha.exp()
                    q1_next = q1_t(no_t, next_a_tanh)
                    q2_next = q2_t(no_t, next_a_tanh)
                    q_next = torch.min(q1_next, q2_next) - alpha * next_logp
                    y = r_t + (1.0 - d_t) * cfg.gamma * q_next

                # Q losses
                q1_pred = q1(o_t, a_tanh)
                q2_pred = q2(o_t, a_tanh)
                q1_loss = ((q1_pred - y) ** 2).mean()
                q2_loss = ((q2_pred - y) ** 2).mean()

                q1_opt.zero_grad()
                q1_loss.backward()
                q1_opt.step()

                q2_opt.zero_grad()
                q2_loss.backward()
                q2_opt.step()

                # Policy + alpha update
                if global_step % cfg.policy_delay == 0:
                    a_env_new, a_tanh_new, logp_new = sample_action_and_logp(actor, o_t)
                    alpha = log_alpha.exp()

                    q1_new = q1(o_t, a_tanh_new)
                    q2_new = q2(o_t, a_tanh_new)
                    q_new = torch.min(q1_new, q2_new)

                    # maximize Q - alpha logpi  => minimize (alpha logpi - Q)
                    act_loss = (alpha * logp_new - q_new).mean()

                    act_opt.zero_grad()
                    act_loss.backward()
                    act_opt.step()

                    # alpha loss
                    alpha_loss = -(log_alpha * (logp_new + target_entropy).detach()).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()

                    # target updates
                    soft_update(q1_t, q1, cfg.tau)
                    soft_update(q2_t, q2, cfg.tau)

                    # logging
                    if global_step % 2048 == 0:
                        writer.add_scalar("train/q1_loss", q1_loss.item(), global_step)
                        writer.add_scalar("train/q2_loss", q2_loss.item(), global_step)
                        writer.add_scalar("train/actor_loss", act_loss.item(), global_step)
                        writer.add_scalar("train/alpha", log_alpha.exp().item(), global_step)
                        writer.add_scalar("train/logp", logp_new.mean().item(), global_step)

        # checkpoint
        if global_step % cfg.save_every == 0:
            ckpt = os.path.join(cfg.model_dir, f"sac_minimal_{global_step}.pt")
            torch.save({
                "actor": actor.state_dict(),
                "q1": q1.state_dict(),
                "q2": q2.state_dict(),
                "log_alpha": log_alpha.detach().cpu().item(),
                "config": cfg.__dict__
            }, ckpt)
            print("[SAVE]", ckpt)

        # periodic console output
        if global_step % 10_000 == 0:
            dt = time.time() - last_log
            last_log = time.time()
            print(f"step={global_step} | replay={len(rb)} | fps~{int(10_000/dt)}")

    final = os.path.join(cfg.model_dir, "sac_minimal_final.pt")
    torch.save({
        "actor": actor.state_dict(),
        "q1": q1.state_dict(),
        "q2": q2.state_dict(),
        "log_alpha": log_alpha.detach().cpu().item(),
        "config": cfg.__dict__
    }, final)
    print("[DONE]", final)
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
