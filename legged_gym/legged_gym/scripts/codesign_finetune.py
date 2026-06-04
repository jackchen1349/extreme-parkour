#!/usr/bin/env python3
"""Fine-tuning + Bayesian Optimization script for the co-design framework.

Phase 2 of the paper's pre-training-fine-tuning framework:
- Loads a pre-trained policy (from Phase 1 pre-training)
- Uses Bayesian optimization to search for optimal leg-length scaling factors
- For each candidate xi, fine-tunes the pre-trained policy for ~400 steps
- Fitness = non-discounted cumulative reward (Eq 7)

Usage:
    python codesign_finetune.py --exptid 002-00-FINETUNE --resume \
        --resumeid 001-00-PRETRAIN --checkpoint 6000 --device cuda:0
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(LEGGED_GYM_DIR))

import isaacgym  # noqa: F401

import torch
import numpy as np
from copy import deepcopy
import argparse

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.codesign.codesign_config import CoDesignCfg, CoDesignCfgPPO
from legged_gym.envs.codesign.codesign_robot import CoDesignLeggedRobot
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import get_load_path, class_to_dict, update_cfg_from_args
from legged_gym.utils.bayesian_optimizer import BayesianOptimizer


def get_finetune_args():
    """Parse arguments for fine-tuning + BO."""
    parser = argparse.ArgumentParser(
        description="Co-Design Fine-tuning with Bayesian Optimization"
    )
    parser.add_argument("--task", type=str, default="codesign", help="Task name")
    parser.add_argument("--exptid", type=str, required=True,
                        help="Experiment ID for this fine-tuning run")
    parser.add_argument("--resumeid", type=str, required=True,
                        help="Pre-trained experiment ID to load from")
    parser.add_argument("--checkpoint", type=int, default=-1,
                        help="Checkpoint iteration to load (-1 = latest)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Number of environments for fine-tuning")
    parser.add_argument("--n_init", type=int, default=5,
                        help="BO initial random samples")
    parser.add_argument("--n_iter", type=int, default=15,
                        help="BO optimization iterations")
    parser.add_argument("--finetune_steps", type=int, default=400,
                        help="Fine-tuning steps per BO iteration")
    parser.add_argument("--eval_episodes", type=int, default=2,
                        help="Number of evaluation episodes for fitness")
    parser.add_argument("--task_type", type=str, default="both",
                        choices=["jump", "high_jump", "both"],
                        help="Task type for fitness evaluation")
    parser.add_argument("--debug", action="store_true", default=False,
                        help="Debug mode: fewer envs")
    parser.add_argument("--web", action="store_true", default=False)
    return parser.parse_args()


def evaluate_fitness(env, runner, num_episodes: int = 2,
                     episode_length_s: float = 20.0) -> float:
    """Evaluate policy fitness (Eq 7): non-discounted cumulative reward.

    f = (1/N) * sum_i sum_t r(s_t^i, a_t^i)

    Parameters
    ----------
    env : CoDesignLeggedRobot
        Environment.
    runner : OnPolicyRunner
        Policy runner.
    num_episodes : int
        Number of evaluation episodes per environment.
    episode_length_s : float
        Episode length in seconds.

    Returns
    -------
    float
        Mean non-discounted cumulative reward.
    """
    runner.alg.actor_critic.eval()
    total_reward = 0.0
    total_steps = 0

    max_steps = int(episode_length_s / (env.dt))
    eval_steps_per_env = num_episodes * max_steps // env.cfg.env.num_envs + 1

    obs = env.get_observations()
    privileged_obs = env.get_privileged_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs

    for _ in range(eval_steps_per_env):
        with torch.no_grad():
            actions = runner.alg.actor_critic.act_inference(obs, hist_encoding=False)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        total_reward += rewards.sum().item()
        total_steps += 1

    runner.alg.actor_critic.train()
    return total_reward / (env.num_envs * num_episodes)


def run_finetune():
    """Main fine-tuning + BO loop."""
    args = get_finetune_args()

    # ---- 1. Load pre-trained checkpoint ----
    pretrain_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "parkour_new")
    resume_path = get_load_path(
        pretrain_log_root,
        load_run=args.resumeid,
        checkpoint=args.checkpoint,
    )
    print(f"Loading pre-trained model from: {resume_path}")
    pretrain_state = torch.load(resume_path, map_location=args.device)

    # ---- 2. Create base configs ----
    env_cfg = CoDesignCfg()
    train_cfg = CoDesignCfgPPO()

    # Fine-tuning: disable spatial randomization, use standard gamma
    env_cfg.spatial_rand.enable = False
    train_cfg.algorithm.gamma = 0.99
    train_cfg.runner.algorithm_class_name = 'PPO'  # Standard PPO for fine-tuning
    train_cfg.runner.experiment_name = 'codesign_finetune'
    train_cfg.runner.max_iterations = args.finetune_steps + 1

    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs
    elif args.debug:
        env_cfg.env.num_envs = 64

    # ---- 3. Set up BO ----
    # Parameter bounds: [front_thigh, front_calf, rear_thigh, rear_calf]
    xi_bounds = [(0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (0.6, 1.4)]
    bo = BayesianOptimizer(
        bounds=xi_bounds,
        n_init=args.n_init,
        n_iter=args.n_iter,
        length_scale=0.3,
        noise=1e-3,
        seed=42,
    )

    print(f"\n{'='*60}")
    print(f"Bayesian Optimization: {bo.n_init} init + {bo.n_iter} iter = {bo.n_total} total")
    print(f"Fine-tuning steps per candidate: {args.finetune_steps}")
    print(f"Parameter bounds: {xi_bounds}")
    print(f"Task type: {args.task_type}")
    print(f"{'='*60}\n")

    best_fitness_so_far = -float('inf')
    best_xi_so_far = None

    for bo_iter in range(bo.n_total):
        xi = bo.suggest()
        print(f"\n--- BO iteration {bo_iter + 1}/{bo.n_total} ---")
        print(f"    Candidate xi: [{xi[0]:.4f}, {xi[1]:.4f}, {xi[2]:.4f}, {xi[3]:.4f}]")

        # ---- 4. Create environment with fixed xi ----
        env_cfg_fresh = deepcopy(env_cfg)
        train_cfg_fresh = deepcopy(train_cfg)

        # Override xi sampling to use fixed values
        env_cfg_fresh.spatial_rand.enable = False

        # Register and create env
        task_registry.register("codesign_finetune", CoDesignLeggedRobot,
                               env_cfg_fresh, train_cfg_fresh)
        # Set fixed xi before env creation (hack: override _xi_values after init)
        # We need to create the env with the fixed xi. Since spatial_rand is
        # disabled, all envs will get xi=1.0. We override this by modifying
        # the xi_values post-creation.
        env, _ = task_registry.make_env(
            name="codesign_finetune",
            args=args,
            env_cfg=env_cfg_fresh,
        )
        env._xi_values[:] = torch.tensor(xi, dtype=torch.float, device=env.device)

        # Recompute PD gains for this xi
        pd_coeffs = env_cfg_fresh.spatial_rand.pd_correction_coeffs
        from legged_gym.envs.codesign.urdf_utils import compute_pd_correction
        for j in range(env.num_dofs):
            name = env.dof_names[j]
            leg_prefix = name[:2]
            if "calf" in name:
                xi_idx = {"FR": 1, "FL": 1, "RR": 3, "RL": 3}.get(leg_prefix, 0)
            else:
                xi_idx = {"FR": 0, "FL": 0, "RR": 2, "RL": 2}.get(leg_prefix, 0)
            eta = compute_pd_correction(xi[xi_idx], pd_coeffs)
            env.p_gains_env[:, j] = eta * env.p_gains[j]
            env.d_gains_env[:, j] = eta * env.d_gains[j]

        # ---- 5. Create runner and load pre-trained weights ----
        runner, train_cfg_runner = task_registry.make_alg_runner(
            env=env,
            name="codesign_finetune",
            args=args,
            train_cfg=train_cfg_fresh,
            init_wandb=False,
        )
        runner.alg.actor_critic.load_state_dict(pretrain_state['model_state_dict'])
        runner.alg.estimator.load_state_dict(pretrain_state['estimator_state_dict'])

        # ---- 6. Fine-tune ----
        print(f"    Fine-tuning for {args.finetune_steps} steps...")
        runner.learn(num_learning_iterations=args.finetune_steps)

        # ---- 7. Evaluate fitness ----
        fitness = evaluate_fitness(env, runner, num_episodes=args.eval_episodes)
        print(f"    Fitness: {fitness:.4f}")

        bo.update(xi, float(fitness))

        if fitness > best_fitness_so_far:
            best_fitness_so_far = fitness
            best_xi_so_far = xi.copy()
            # Save best model
            best_path = os.path.join(
                LEGGED_GYM_ROOT_DIR, "logs", "parkour_new",
                f"{args.exptid}_best_xi_{bo_iter}.pt"
            )
            os.makedirs(os.path.dirname(best_path), exist_ok=True)
            torch.save({
                'model_state_dict': runner.alg.actor_critic.state_dict(),
                'estimator_state_dict': runner.alg.estimator.state_dict(),
                'xi': xi,
                'fitness': fitness,
            }, best_path)

        print(f"    Best so far: xi={best_xi_so_far}, fitness={best_fitness_so_far:.4f}")

        # Cleanup env to free GPU memory
        del env, runner

    # ---- 8. Report results ----
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best xi: {best_xi_so_far}")
    print(f"Best fitness: {best_fitness_so_far:.4f}")
    print(f"All history:")
    X_hist, y_hist = bo.history
    for i in range(len(y_hist)):
        print(f"  [{i}] xi={X_hist[i].round(4)}, fitness={y_hist[i]:.4f}")


if __name__ == "__main__":
    run_finetune()
