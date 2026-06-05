# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause
#
# Pre-training script for structure-control co-design.
# Uses spatial domain randomization + discount regularization (gamma_reg=0.98).
#
# Usage:
#   python codesign_pretrain.py --exptid 001-00-PRETRAIN --device cuda:0
#   python codesign_pretrain.py --exptid 001-00-PRETRAIN --device cuda:0 --debug

import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
import torch
import wandb


def train(args):
    args.headless = True
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass

    if args.debug:
        mode = "disabled"
        args.headless = False
        args.rows = 5
        args.cols = 8
        args.num_envs = 64
    else:
        mode = "online"

    if args.no_wandb:
        mode = "disabled"

    wandb.init(project=args.proj_name, name=args.exptid,
               entity="jackchen1349-shenzhen", group=args.exptid[:3],
               mode=mode, dir="../../logs")

    # Create env using pre-registered "codesign" task (envs/__init__.py)
    env, env_cfg = task_registry.make_env(name="codesign", args=args)

    # Log full structured config to wandb
    wandb.config.update({"env_cfg": class_to_dict(env_cfg)})

    # Create runner (uses CoDesignPPO with gamma_reg=0.98 from config)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root=log_pth, env=env, name="codesign", args=args)

    wandb.config.update({"train_cfg": class_to_dict(train_cfg)})

    # Upload source config files
    wandb.save(LEGGED_GYM_ENVS_DIR + "/codesign/codesign_config.py", policy="now")
    wandb.save(LEGGED_GYM_ENVS_DIR + "/codesign/codesign_robot.py", policy="now")

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,
                     init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    train(args)
