# CoDesignPPO — PPO variant with discount regularization.
#
# Reference: Chen et al., "Structure-control Co-design of Quadruped
# Robots Based on Pre-training-Fine-tuning Framework", ROBOT, 2025.
#
# Eq (3): A_t^{gamma_reg} = sum_{l=0}^{inf} (gamma_reg * lambda)^l * delta_{t+l}^{gamma_reg}
# Eq (4): delta_t^{gamma_reg} = r_t + gamma_reg * V(s_{t+1}) - V(s_t)
#
# The discount-regularized GAE uses gamma_reg = 0.98 instead of the
# standard gamma = 0.99. This reduces reliance on distant future rewards,
# improving policy generalization across different morphologies.

import torch

from rsl_rl.algorithms.ppo import PPO


class CoDesignPPO(PPO):
    """PPO algorithm with discount regularization for co-design pre-training.

    Accepts both ``gamma`` (standard discount factor) and ``gamma_reg``
    (regularized discount factor). During pre-training, ``gamma_reg`` is
    used in the GAE advantage/return computation (Eq 3-4), reducing
    variance and improving generalization across morphologies.
    """

    def __init__(self, *args, gamma_reg=0.98, **kwargs):
        """Initialize CoDesignPPO.

        Parameters
        ----------
        gamma_reg : float
            Regularized discount factor for GAE computation (paper: 0.98).
            Must satisfy 0 < gamma_reg < gamma <= 1.
        """
        super().__init__(*args, **kwargs)
        self.gamma_reg = gamma_reg

    def compute_returns(self, last_critic_obs):
        """Override to use gamma_reg for GAE computation.

        Eq (3): A_t^{gamma_reg} = sum_l (gamma_reg * lambda)^l * delta_{t+l}^{gamma_reg}
        Eq (4): delta_t^{gamma_reg} = r_t + gamma_reg * V(s_{t+1}) - V(s_t)

        The standard PPO uses self.gamma (default 0.99); here we use
        self.gamma_reg (default 0.98) for the GAE computation.
        """
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma_reg, self.lam)
