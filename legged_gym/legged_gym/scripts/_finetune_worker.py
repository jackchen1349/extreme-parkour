#!/usr/bin/env python3
"""Worker script for co-design fine-tuning + fitness evaluation.

Evaluates one candidate xi vector by:
1. Creating an environment with fixed leg-length scaling xi
2. Loading a pre-trained policy checkpoint
3. Fine-tuning for N PPO iterations
4. Evaluating the non-discounted cumulative reward (Eq 7)

Runs as a subprocess spawned by codesign_finetune.py so each evaluation
gets a fresh Isaac Gym / PhysX instance that is fully reclaimed on exit.

Usage:
    python _finetune_worker.py --xi "0.8,1.2,0.9,1.1" \
        --resume_path /path/to/model_6000.pt \
        --finetune_steps 400 --eval_episodes 2 \
        --task_type jump \
        --tag finetune_test_000 --out /tmp/result.json
"""

import os
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(SCRIPT_DIR)))

import isaacgym  # noqa: F401 — MUST come before torch
import torch
import numpy as np

from legged_gym.envs.codesign.codesign_config import CoDesignCfg, CoDesignCfgPPO
from legged_gym.envs.codesign.codesign_robot import CoDesignLeggedRobot
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import get_args
from legged_gym import LEGGED_GYM_ROOT_DIR

# ---------------------------------------------------------------------------
# Terrain mapping: --task_type -> terrain_dict
# ---------------------------------------------------------------------------
_ALL_TERRAIN_KEYS = [
    "smooth slope", "rough slope up", "rough slope down",
    "rough stairs up", "rough stairs down", "discrete",
    "stepping stones", "gaps", "smooth flat",
    "pit", "wall", "platform",
    "large stairs up", "large stairs down",
    "parkour", "parkour_hurdle", "parkour_flat",
    "parkour_step", "parkour_gap", "demo",
]

TASK_TERRAIN_MAP = {
    "jump": {**{k: 0. for k in _ALL_TERRAIN_KEYS}, "parkour_gap": 1.0},
    "high_jump": {**{k: 0. for k in _ALL_TERRAIN_KEYS}, "parkour_hurdle": 1.0},
    "both": {
        **{k: 0. for k in _ALL_TERRAIN_KEYS},
        "parkour": 0.2, "parkour_hurdle": 0.2, "parkour_flat": 0.2,
        "parkour_step": 0.2, "parkour_gap": 0.2,
    },
}


def evaluate_fitness(env, runner, num_episodes: int = 2) -> float:
    """Evaluate non-discounted cumulative reward (Eq 7).

    Wrapped in inference_mode so env tensor inplace ops work correctly
    after runner.learn() tainted them during PPO rollout.
    """
    runner.alg.actor_critic.eval()
    total_reward = 0.0

    max_steps = int(env.max_episode_length)
    eval_steps_per_env = num_episodes * max_steps // env.cfg.env.num_envs + 1

    with torch.inference_mode():
        obs = env.get_observations()
        privileged_obs = env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs

        for _ in range(eval_steps_per_env):
            actions = runner.alg.actor_critic.act_inference(obs, hist_encoding=False)
            obs, privileged_obs, rewards, dones, infos = env.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            total_reward += rewards.sum().item()

    runner.alg.actor_critic.train()
    return total_reward / (env.num_envs * num_episodes)


def run(args, xi_list):
    """Fine-tune and evaluate one candidate xi."""

    # wandb: disabled in debug, otherwise online (same pattern as codesign_pretrain.py)
    import wandb
    if args.debug:
        mode = "disabled"
    else:
        mode = "online"
    if args.no_wandb:
        mode = "disabled"
    wandb.init(project="codesign_finetune", name=f"ft_{args.tag}", mode=mode, dir="../../logs")

    # ---- 1. Create configs ----
    env_cfg = CoDesignCfg()
    train_cfg = CoDesignCfgPPO()
    env_cfg.seed = train_cfg.seed

    # Fixed xi (no spatial randomization)
    env_cfg.spatial_rand.enable = False
    env_cfg.spatial_rand.target_xi = xi_list

    # Lock terrain at max difficulty
    env_cfg.terrain.curriculum = False

    # Set terrain based on task_type
    task_type = getattr(args, 'task_type', 'both')
    terrain_map = TASK_TERRAIN_MAP.get(task_type, TASK_TERRAIN_MAP["both"])
    env_cfg.terrain.terrain_dict = dict(terrain_map)
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())

    # Fine-tuning uses standard PPO (not CoDesignPPO with gamma_reg)
    train_cfg.runner.algorithm_class_name = 'PPO'
    train_cfg.runner.experiment_name = 'codesign_finetune'
    train_cfg.runner.max_iterations = args.finetune_steps + 1
    # train_cfg.algorithm.gamma = 0.99

    if args.num_envs is not None:
        env_cfg.env.num_envs = args.num_envs

    # ---- 2. Create log directory (fixes Issue 1) ----
    log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", "codesign_finetune",
                           f"{args.tag}")
    os.makedirs(log_dir, exist_ok=True)

    # ---- 3. Register task and create env ----
    task_name = f"finetune_{args.tag}"
    task_registry.register(task_name, CoDesignLeggedRobot, env_cfg, train_cfg)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)

    # Lock all envs at max terrain difficulty
    env.terrain_levels[:] = env.max_terrain_level - 1

    # ---- 4. Create runner and load pre-trained weights ----
    runner, _ = task_registry.make_alg_runner(
        env=env, name=task_name, args=args, train_cfg=train_cfg,
        log_root=log_dir,
    )

    pretrain_state = torch.load(args.resume_path, map_location=args.device)
    runner.alg.actor_critic.load_state_dict(pretrain_state['model_state_dict'])
    runner.alg.estimator.load_state_dict(pretrain_state['estimator_state_dict'])

    # ---- 5. Fine-tune ----
    print(f"[worker:{args.tag}] Fine-tuning {args.finetune_steps} steps on "
          f"{task_type} terrain, xi={[round(v,4) for v in xi_list]}")
    runner.learn(num_learning_iterations=args.finetune_steps, init_at_random_ep_len=True)

    # ---- 6. Evaluate fitness ----
    fitness = evaluate_fitness(env, runner, num_episodes=args.eval_episodes)
    print(f"[worker:{args.tag}] Fitness: {fitness:.4f}")

    # ---- 7. Write result ----
    result = {
        "xi": xi_list,
        "fitness": float(fitness),
        "task_type": task_type,
        "tag": args.tag,
        "finetune_steps": args.finetune_steps,
    }
    with open(args.out, "w") as f:
        json.dump(result, f)

    # ---- 8. Cleanup ----
    del env, runner


if __name__ == "__main__":
    worker_params = [
        {"name": "--xi", "type": str, "required": True,
         "help": "Comma-separated xi: '0.8,1.2,0.9,1.1'"},
        {"name": "--resume_path", "type": str, "required": True,
         "help": "Path to pre-trained model checkpoint (.pt)"},
        {"name": "--finetune_steps", "type": int, "default": 400,
         "help": "Fine-tuning PPO iterations"},
        {"name": "--eval_episodes", "type": int, "default": 2,
         "help": "Evaluation episodes for fitness"},
        {"name": "--tag", "type": str, "required": True,
         "help": "Unique tag for this evaluation"},
        {"name": "--out", "type": str, "required": True,
         "help": "Path for JSON result file"},
        {"name": "--task_type", "type": str, "default": "both",
         "choices": ["jump", "high_jump", "both"]},
    ]
    args = get_args(extra_parameters=worker_params)

    xi = [float(x) for x in args.xi.split(",")]
    assert len(xi) == 4, f"Expected 4 xi values, got {len(xi)}"

    run(args, xi)
