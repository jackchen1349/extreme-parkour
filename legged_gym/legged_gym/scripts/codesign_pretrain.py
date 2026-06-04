#!/usr/bin/env python3
"""Pre-training script for the co-design framework.

Phase 1 of the paper's pre-training-fine-tuning framework:
- Uses spatial domain randomization (multiple robot morphologies)
- Uses discount regularization (gamma_reg = 0.98)
- Trains a generalizable parkour policy across diverse leg lengths

Usage (must run from legged_gym/legged_gym/scripts/):
    python codesign_pretrain.py --exptid 001-00-PRETRAIN --device cuda:0

    # Debug mode (small scale):
    python codesign_pretrain.py --exptid 001-00-PRETRAIN --device cuda:0 --debug
"""

import os
import sys

# Ensure the parent directory is on the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(LEGGED_GYM_DIR))

import isaacgym  # noqa: F401 (MUST be before torch)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.codesign.codesign_config import CoDesignCfg, CoDesignCfgPPO
from legged_gym.envs.codesign.codesign_robot import CoDesignLeggedRobot
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict


def train(args):
    """Pre-training entry point."""
    # Override default task name
    if args.task == "a1":
        args.task = "codesign"

    # Register the codesign task (idempotent if already registered)
    env_cfg = CoDesignCfg()
    train_cfg = CoDesignCfgPPO()

    # Register task
    task_registry.register("codesign", CoDesignLeggedRobot, env_cfg, train_cfg)

    # Build log path
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name)
    log_dir = os.path.join(log_root, args.exptid) if args.exptid else None

    # Apply CLI overrides
    update_cfg_from_args(env_cfg, train_cfg, args)
    env_cfg_dict = class_to_dict(env_cfg)
    train_cfg_dict = class_to_dict(train_cfg)

    # Create environment
    env, env_cfg = task_registry.make_env(name="codesign", args=args, env_cfg=env_cfg)
    print(f"Created environment with {env.num_envs} envs")
    print(f"Spatial domain randomization: {env_cfg.spatial_rand.enable}")
    print(f"xi range: {env_cfg.spatial_rand.xi_range}")
    print(f"num unique URDF groups: {env_cfg.spatial_rand.num_groups}")

    # Create algorithm runner
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name="codesign", args=args, train_cfg=train_cfg,
        log_root=log_dir
    )
    print(f"Algorithm: {train_cfg.runner.algorithm_class_name}")
    print(f"gamma: {train_cfg.algorithm.gamma}")
    print(f"gamma_reg: {train_cfg.algorithm.gamma_reg}")
    print(f"max_iterations: {train_cfg.runner.max_iterations}")

    # Start pre-training
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True
    )


if __name__ == "__main__":
    args = get_args()
    train(args)
