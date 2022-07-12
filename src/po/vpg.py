"""
Many conventions, styles, and method from 
https://github.com/openai/spinningup.
"""

from typing import Dict, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from basemodels import MLPActorCritic

# Implements GAE & Reward-to-go of Vanila Policy Gradient


OBSERVATION: str = "_literal_observation"
ACTION: str = "_literal_action"
REWARD: str = "_literal_reward"
RETURN: str = "_literal_return"
VALUE: str = "_literal_value"
POL_WEIGHT: str = "_literal_pol_weight"
LOG_P: str = "_literal_log_p"


def discount_cumsum(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Slow, recusive implementation... But can it get better than this?
    """
    for i in range(len(x) - 1, 0, -1):
        x[i - 1] += gamma * x[i]

    return x


class VanilaPolicyGradient:
    def __init__(
        self,
        env,
        actor_critic: nn.Module,
        steps_per_epoch: int = 1000,
        epochs: int = 250,
        gamma: float = 0.99,
        pi_lr: float = 3e-4,
        vf_lr: float = 1e-3,
        train_v_iters: int = 100,
        lam: float = 0.95,
        max_ep_len: int = 3000,
        pg_weight: str = "reward-to-go",
        device: str = "cpu",
    ):
        self.env = env
        self.ac = actor_critic
        self.ac = self.ac.to(device)

        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_v_iters = train_v_iters
        self.lam = lam
        self.max_ep_len = max_ep_len

        self.pi_opt = optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_opt = optim.Adam(self.ac.v.parameters(), lr=vf_lr)

        self.pg_weight = pg_weight
        self.device = device

        _MAX_STEP = max_ep_len

        self.empty_buffs = {
            OBSERVATION: torch.zeros(_MAX_STEP, *self.env.observation_space.shape),
            ACTION: torch.zeros(_MAX_STEP, *self.env.action_space.shape),
            REWARD: torch.zeros(_MAX_STEP, 1),
            RETURN: torch.zeros(_MAX_STEP, 1),
            VALUE: torch.zeros(_MAX_STEP, 1),
            POL_WEIGHT: torch.zeros(_MAX_STEP, 1),
            LOG_P: torch.zeros(_MAX_STEP, 1),
        }

        print("BUFFER SIZES")
        for k, v in self.empty_buffs.items():
            print(f"{k}: {v.shape}")

    @torch.no_grad()
    def collect(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, Union[int, float]],
    ]:
        obs, ep_ret, ep_len = self.env.reset(), 0, 0
        ptr = 0

        bufs = {k: v.clone() for k, v in self.empty_buffs.items()}

        b_obs, b_acts, b_weights, b_rets, b_logp = [], [], [], [], []

        for t in range(self.steps_per_epoch):
            action, value, logp = self.ac.step(torch.tensor(obs).to(self.device))
            # print(action, value, logp)
            next_obs, reward, is_done, _ = self.env.step(action)

            ep_ret += reward
            ep_len += 1

            bufs[OBSERVATION][ptr] = torch.tensor(obs)
            bufs[ACTION][ptr] = torch.tensor(action)
            bufs[REWARD][ptr] = torch.tensor(reward)
            bufs[VALUE][ptr] = torch.tensor(value)
            bufs[LOG_P][ptr] = torch.tensor(logp)
            ptr += 1

            # print(bufs[ACTION])

            obs = next_obs

            timeout = ep_len == self.max_ep_len
            terminal = is_done or timeout
            epoch_ended = t == self.steps_per_epoch - 1

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, _ = self.ac.step(torch.tensor(obs).to(self.device))
                else:
                    v = 0

                bufs[REWARD][ptr] = torch.tensor(v)
                _totlen = ptr

                rwds = bufs[REWARD][: _totlen + 1]
                bufs[RETURN] = discount_cumsum(rwds, self.gamma)[:-1]

                if self.pg_weight == "discounted-returns":
                    bufs[POL_WEIGHT] = rwds[0].repeat(len(bufs[RETURN]))

                elif self.pg_weight == "reward-to-go":
                    bufs[POL_WEIGHT] = discount_cumsum(rwds, self.gamma)[:-1]

                elif self.pg_weight == "reward-to-go-baseline":
                    bufs[POL_WEIGHT] = bufs[RETURN]

                elif self.pg_weight == "discounted-td-residual":
                    bufs[VALUE][ptr] = torch.tensor(v)
                    vals = bufs[VALUE][: _totlen + 1]
                    deltas = rwds[:-1] + self.gamma * vals[1:] - vals[:-1]
                    bufs[POL_WEIGHT] = deltas

                elif self.pg_weight == "gae":
                    bufs[VALUE][ptr] = torch.tensor(v)
                    vals = bufs[VALUE][: _totlen + 1]
                    deltas = rwds[:-1] + self.gamma * vals[1:] - vals[:-1]
                    bufs[POL_WEIGHT] = discount_cumsum(deltas, self.gamma * self.lam)

                if terminal:

                    wandb.log(
                        {
                            "ep_ret": ep_ret,
                            "ep_len": ep_len,
                        }
                    )
                    print("Episode {} finished after {} steps".format(ep_len, ep_len))

                obs, ep_ret, ep_len = self.env.reset(), 0, 0

                b_obs.append(bufs[OBSERVATION][:_totlen])
                b_acts.append(bufs[ACTION][:_totlen])
                b_weights.append(bufs[POL_WEIGHT][:_totlen])
                b_rets.append(bufs[RETURN][:_totlen])
                b_logp.append(bufs[LOG_P][:_totlen])

                # clear
                bufs = {k: v.clone() for k, v in self.empty_buffs.items()}
                ptr = 0

        wghts = torch.cat(b_weights, dim=0)
        # normalized
        wghts = (wghts - wghts.mean()) / (wghts.std())

        return (
            torch.cat(b_obs, dim=0),
            torch.cat(b_acts, dim=0),
            wghts,
            torch.cat(b_rets, dim=0),
            torch.cat(b_logp, dim=0),
            {},
        )

    def optimize(self, obs, act, weight, ret, logp) -> None:

        obs, act, weight, ret, logp = (
            obs.to(self.device),
            act.to(self.device),
            weight.to(self.device),
            ret.to(self.device),
            logp.to(self.device),
        )

        self.pi_opt.zero_grad()
        # compute pi
        pi, logp = self.ac.pi(obs, act)
        loss_pi = -(logp * weight).mean()
        loss_pi.backward()
        self.pi_opt.step()

        for i in range(self.train_v_iters):
            self.vf_opt.zero_grad()
            # compute v
            loss_v = (self.ac.v(obs) - ret).pow(2).mean()
            loss_v.backward()
            self.vf_opt.step()

    def train(self, debug: bool = False) -> None:

        if not debug:
            wandb.init(
                project="policy-optimization-torch",
                name=f"{self.env.spec.id}-{self.__class__.__name__}-{self.pg_weight}",
                entity="simoryu",
                config={
                    "gamma": self.gamma,
                    "pi_lr": self.pi_lr,
                    "vf_lr": self.vf_lr,
                    "train_v_iters": self.train_v_iters,
                    "lam": self.lam,
                    "max_ep_len": self.max_ep_len,
                    "pg_weight": self.pg_weight,
                    "steps_per_epoch": self.steps_per_epoch,
                    "epochs": self.epochs,
                    "env_name": self.env.__class__.__name__,
                    "actor_critic": self.ac.__class__.__name__,
                    "env": self.env.__class__.__name__,
                },
                reinit=True,
            )

        # wandb.watch(self.ac)

        for epoch in range(self.epochs):

            print(f"Epoch {epoch}")

            obs, act, weight, ret, logp, meta = self.collect()
            self.optimize(obs, act, weight, ret, logp)

            if debug:
                break

        wandb.finish()


if __name__ == "__main__":

    # test_vec = torch.randn(10)
    # print(test_vec)
    # print(discount_cumsum(test_vec, 0.9))

    env = gym.make("BipedalWalker-v3")

    ac = MLPActorCritic(env.observation_space, env.action_space, (64, 64))
    vpg = VanilaPolicyGradient(
        env=env, actor_critic=ac, pg_weight="reward-to-go", device="cuda"
    )

    vpg.train(debug=False)
