import os, time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Cfg:
    total_steps: int = 3_000_000   # bump to 5M if needed
    n_envs: int = 8                # can drop to 4
    rollout_len: int = 1024
    update_epochs: int = 4
    minibatch: int = 2048

    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.10

    lr: float = 1e-4
    ent: float = 0.003
    vf: float = 0.5
    grad: float = 0.5

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    logdir: str = "runs/ppo_minimal"
    model_dir: str = "models"


def make_env(seed, idx):
    def thunk():
        env = gym.make("CarRacing-v3", continuous=True)
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.FrameStackObservation(env, 4)
        env.reset(seed=seed + idx)
        return env
    return thunk


class AC(nn.Module):
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


def obs_t(obs, device):
    return torch.from_numpy(np.asarray(obs, np.float32) / 255.0).to(device)


def logp_tanh(mu, logstd, u):
    # Gaussian logprob in pre-tanh space + tanh correction
    dist = Normal(mu, torch.exp(logstd))
    logp_u = dist.log_prob(u).sum(-1)
    a = torch.tanh(u)
    log_det = torch.log(1 - a * a + 1e-6).sum(-1)
    return logp_u - log_det, dist.entropy().sum(-1)


def action_env(u):
    # map to env: steer [-1,1], gas/brake [0,1]
    steer = torch.tanh(u[..., 0])
    gas = torch.sigmoid(u[..., 1])
    brake = torch.sigmoid(u[..., 2])
    return torch.stack([steer, gas, brake], dim=-1)


def main():
    cfg = Cfg()
    os.makedirs(cfg.logdir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)
    w = SummaryWriter(cfg.logdir)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    envs = gym.vector.SyncVectorEnv([make_env(cfg.seed, i) for i in range(cfg.n_envs)])
    obs, _ = envs.reset(seed=cfg.seed)

    net = AC().to(cfg.device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.rollout_len, cfg.n_envs
    B = T * N

    obs_buf = np.zeros((T, N, 4, 84, 84), np.uint8)
    u_buf = np.zeros((T, N, 3), np.float32)
    logp_buf = np.zeros((T, N), np.float32)
    rew_buf = np.zeros((T, N), np.float32)
    done_buf = np.zeros((T, N), np.float32)
    val_buf = np.zeros((T, N), np.float32)

    step = 0
    it = 0
    start = time.time()

    while step < cfg.total_steps:
        it += 1

        # rollout
        for t in range(T):
            obs_buf[t] = obs
            x = obs_t(obs, cfg.device)
            with torch.no_grad():
                mu, logstd, v = net(x)
                u = Normal(mu, torch.exp(logstd)).sample()
                logp, ent = logp_tanh(mu, logstd, u)

            u_buf[t] = u.cpu().numpy()
            logp_buf[t] = logp.cpu().numpy()
            val_buf[t] = v.cpu().numpy()

            a = action_env(u).cpu().numpy().astype(np.float32)
            obs, r, term, trunc, _ = envs.step(a)
            done = (term | trunc).astype(np.float32)

            rew_buf[t] = r.astype(np.float32)
            done_buf[t] = done
            step += N

        # bootstrap
        with torch.no_grad():
            _, _, v_last = net(obs_t(obs, cfg.device))
        v_last = v_last.cpu().numpy()

        # GAE
        adv = np.zeros((T, N), np.float32)
        last = np.zeros((N,), np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - done_buf[t]
            next_v = v_last if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_v * nonterm - val_buf[t]
            last = delta + cfg.gamma * cfg.lam * nonterm * last
            adv[t] = last
        ret = adv + val_buf

        # flatten
        b_obs = obs_buf.reshape(B, 4, 84, 84)
        b_u = u_buf.reshape(B, 3)
        b_logp = logp_buf.reshape(B)
        b_adv = adv.reshape(B)
        b_ret = ret.reshape(B)

        # advantage normalization (keep!)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # PPO update
        idx = np.arange(B)
        kls = []; clips = []; ents = []; pls = []; vls = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, B, cfg.minibatch):
                mb = idx[s:s + cfg.minibatch]
                x = obs_t(b_obs[mb], cfg.device)
                mu, logstd, v = net(x)

                u = torch.from_numpy(b_u[mb]).to(cfg.device)
                logp, ent = logp_tanh(mu, logstd, u)

                old = torch.from_numpy(b_logp[mb]).to(cfg.device)
                adv_t = torch.from_numpy(b_adv[mb]).to(cfg.device)
                ret_t = torch.from_numpy(b_ret[mb]).to(cfg.device)

                ratio = torch.exp(logp - old)
                s1 = ratio * adv_t
                s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv_t
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * (ret_t - v).pow(2).mean()
                entropy = ent.mean()

                loss = pol_loss + cfg.vf * v_loss - cfg.ent * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad)
                opt.step()

                with torch.no_grad():
                    kl = (old - logp).mean().item()
                    clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean().item()
                kls.append(kl); clips.append(clipfrac)
                ents.append(entropy.item()); pls.append(pol_loss.item()); vls.append(v_loss.item())

        if it % 10 == 0:
            fps = int(step / (time.time() - start))
            print(f"steps={step:8d} | it={it:4d} | kl={np.mean(kls):.4f} | clip={np.mean(clips):.3f} | ent={np.mean(ents):.3f} | pol={np.mean(pls):.3f} | v={np.mean(vls):.3f} | fps={fps}")
            w.add_scalar("train/kl", float(np.mean(kls)), it)
            w.add_scalar("train/clipfrac", float(np.mean(clips)), it)
            w.add_scalar("train/entropy", float(np.mean(ents)), it)

        if step % 200_000 < N:
            p = os.path.join(cfg.model_dir, f"ppo_minimal_{step}.pt")
            torch.save({"state_dict": net.state_dict(), "config": cfg.__dict__}, p)
            print("[SAVE]", p)

    final = os.path.join(cfg.model_dir, "ppo_minimal_final.pt")
    torch.save({"state_dict": net.state_dict(), "config": cfg.__dict__}, final)
    print("[DONE]", final)
    envs.close()
    w.close()


if __name__ == "__main__":
    main()
