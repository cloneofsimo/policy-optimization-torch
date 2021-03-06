"""
Many conventions, styles, and method from 
https://github.com/openai/spinningup.
"""

import gym
import torch
import torch.nn as nn

from basemodels import MLPActorCritic
from vpg import VanilaPolicyGradient


class PPO(VanilaPolicyGradient):
    def __init__(
        self,
        env,
        actor_critic: nn.Module,
        steps_per_epoch: int = 6400,
        epochs: int = 500,
        gamma: float = 0.99,
        pi_lr: float = 1e-4,
        vf_lr: float = 1e-4,
        train_v_iters: int = 200,
        train_pi_iters: int = 50,
        lam: float = 0.95,
        max_ep_len: int = 1600,
        pg_weight: str = "reward-to-go",
        target_kl=0.01,
        clip_ratio=0.2,
        device: str = "cpu",
    ):
        super().__init__(
            env,
            actor_critic,
            steps_per_epoch,
            epochs=epochs,
            gamma=gamma,
            pi_lr=pi_lr,
            vf_lr=vf_lr,
            train_v_iters=train_v_iters,
            lam=lam,
            max_ep_len=max_ep_len,
            pg_weight=pg_weight,
            device = device,
        )

        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio

    def optimize(self, obs, act, weight, ret, logp) -> None:

        self.ac.train()

        obs, act, weight, ret, logp = (
            obs.to(self.device),
            act.to(self.device),
            weight.to(self.device),
            ret.to(self.device),
            logp.to(self.device),
        )
        logp_old = logp

        # Optimzie pi
        for i in range(self.train_pi_iters):
            self.pi_opt.zero_grad()

            pi, logp = self.ac.pi(obs, act)
            ratio = torch.exp(logp - logp_old)
            cliped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

            loss_pi = -(torch.min(ratio * weight, cliped * weight)).mean()
            kl = (logp_old - logp).mean().item()

            if kl > 1.5 * self.target_kl:
                print("# Not updating policy")
                break

            loss_pi.backward()
            self.pi_opt.step()

        for i in range(self.train_v_iters):
            self.vf_opt.zero_grad()
            # compute v
            loss_v = (self.ac.v(obs) - ret).pow(2).mean()
            loss_v.backward()
            self.vf_opt.step()

        self.ac.eval()


if __name__ == "__main__":

    # compares ppo and vpg, with different reward weights.
    env = gym.make("CartPole-v0")
    env = gym.make("BipedalWalker-v3")
    ac = MLPActorCritic(env.observation_space, env.action_space, (128, 128, 128))
    ppo = PPO(env=env, actor_critic=ac, pg_weight="gae", device = "cpu")
    ppo.train(debug=False)

    # for pg_weight in [
    #     "discounted-returns",
    #     "reward-to-go",
    #     "reward-to-go-baseline",
    #     "discounted-td-residual",
    #     "gae",
    # ]:

    #     ac = MLPActorCritic(env.observation_space, env.action_space, (64, 64))
    #     ppo = PPO(env=env, actor_critic=ac, pg_weight=pg_weight, device = "cuda:1")
    #     ppo.train(debug=False)

    #     ac = MLPActorCritic(env.observation_space, env.action_space, (128, 256, 128))
    #     vga = VanilaPolicyGradient(env=env, actor_critic=ac, pg_weight=pg_weight, device = "cuda:1")
    #     vga.train(debug=False)
