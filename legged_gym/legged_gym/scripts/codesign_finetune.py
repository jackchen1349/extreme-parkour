#!/usr/bin/env python3
"""Fine-tuning + Bayesian Optimization script for the co-design framework.

Phase 2 of the paper's pre-training-fine-tuning framework:
- Loads a pre-trained policy (from Phase 1 pre-training)
- Uses Bayesian optimization to search for optimal leg-length scaling factors
- For each candidate xi, spawns a subprocess that fine-tunes and evaluates
- Fitness = non-discounted cumulative reward (Eq 7)
- --task_type controls which terrain is used for evaluation:
    jump      -> parkour_gap only
    high_jump -> parkour_hurdle only
    both      -> all 5 parkour types

Usage:
    python codesign_finetune.py --exptid 011-00-FINETUNE --resume \
        --resumeid 010-00 --checkpoint 6000 --device cuda:0 --task_type jump
"""

import os
import sys
import json
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEGGED_GYM_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(LEGGED_GYM_DIR))

# Isaac Gym MUST be imported before torch (required by helpers import chain)
import isaacgym  # noqa: F401

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR

# Import from legged_gym.envs first to resolve circular import:
#   legged_gym.utils -> task_registry -> legged_gym.envs.base.legged_robot_config
#   -> legged_gym.envs.__init__ -> task_registry (CIRCULAR)
# By importing from legged_gym.envs first, task_registry initializes cleanly.
from legged_gym.envs.codesign.codesign_config import CoDesignCfg, CoDesignCfgPPO  # noqa: F401

from legged_gym.utils.helpers import get_load_path, get_args
from legged_gym.utils.bayesian_optimizer import BayesianOptimizer

# Path to worker script
FINETUNE_WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "_finetune_worker.py")


def get_finetune_args():
    """Parse arguments for fine-tuning + BO, reusing get_args as base."""
    finetune_params = [
        {"name": "--n_init", "type": int, "default": 5,
         "help": "BO initial random samples"},
        {"name": "--n_iter", "type": int, "default": 15,
         "help": "BO optimization iterations"},
        {"name": "--finetune_steps", "type": int, "default": 800,
         "help": "Fine-tuning steps per BO iteration"},
        {"name": "--eval_episodes", "type": int, "default": 2,
         "help": "Evaluation episodes for fitness"},
        {"name": "--task_type", "type": str, "default": "both",
         "choices": ["jump", "high_jump", "both"],
         "help": "Terrain type for fitness evaluation"},
    ]
    args = get_args(extra_parameters=finetune_params)
    if args.task == "a1":  # override get_args default for finetune
        args.task = "codesign"
    args.proj_name = "codesign_finetune"
    return args


def evaluate_candidate(xi, checkpoint_path, args, bo_iter):
    """Run one fine-tuning + evaluation in a subprocess.

    Parameters
    ----------
    xi : ndarray (4,)
        Candidate [front_thigh, front_calf, rear_thigh, rear_calf].
    checkpoint_path : str
        Absolute path to pre-trained model checkpoint.
    args : argparse.Namespace
        Experiment configuration.
    bo_iter : int
        Index of this BO iteration.

    Returns
    -------
    tuple (ndarray, float)
        The evaluated xi and fitness.
    """
    xi_str = ",".join(f"{v:.6f}" for v in xi)
    tag = f"{args.exptid}_{bo_iter:03d}"
    result_file = f"/tmp/finetune_result_{tag}.json"

    # Resume check: skip if result already exists
    if os.path.exists(result_file):
        print(f"    [resume] Found cached result at {result_file}")
        with open(result_file) as f:
            d = json.load(f)
        return np.array(d["xi"]), d["fitness"]

    cmd = [
        sys.executable, FINETUNE_WORKER_SCRIPT,
        f"--xi={xi_str}",
        "--resume_path", checkpoint_path,
        "--finetune_steps", str(args.finetune_steps),
        "--eval_episodes", str(args.eval_episodes),
        "--tag", tag,
        "--out", result_file,
        "--device", args.device,
        "--task_type", args.task_type,
    ]
    if args.num_envs is not None:
        cmd.extend(["--num_envs", str(args.num_envs)])
    if getattr(args, 'rows', None) is not None:
        cmd.extend(["--rows", str(args.rows)])
    if getattr(args, 'cols', None) is not None:
        cmd.extend(["--cols", str(args.cols)])
    # Pass --headless unless debug mode (where we want the viewer)
    if not args.debug:
        cmd.append("--headless")
    # Pass debug/wandb flags to worker
    if args.debug:
        cmd.append("--debug")
    if args.no_wandb:
        cmd.append("--no_wandb")

    # Set LD_LIBRARY_PATH so subprocess finds libpython3.8.so
    env = os.environ.copy()
    conda_lib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    timeout_seconds = max(3600, args.finetune_steps * 10)
    subprocess.run(cmd, check=True, timeout=timeout_seconds, env=env)

    with open(result_file) as f:
        d = json.load(f)
    return np.array(d["xi"]), d["fitness"]


def run_finetune():
    """Main fine-tuning + BO loop."""
    args = get_finetune_args()

    # Debug mode: smaller env, disable wandb, show viewer (same pattern as pretrain)
    if args.debug:
        if args.num_envs is None:
            args.num_envs = 64
        args.no_wandb = True
        args.headless = False
        args.rows = 5
        args.cols = 8
        print(f"[DEBUG] num_envs={args.num_envs}, rows=5, cols=8, headless=False, no_wandb=True")

    # ---- 1. Resolve pre-trained checkpoint path ----
    pretrain_log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs",
                                     "parkour_new", args.resumeid)
    resume_path = get_load_path(pretrain_log_root, checkpoint=args.checkpoint)
    print(f"Pre-trained checkpoint: {resume_path}")

    # ---- 2. Create log directory (fixes Issue 1) ----
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)
    os.makedirs(log_root, exist_ok=True)

    # ---- 3. Set up BO ----
    xi_bounds = [(0.6, 1.4), (0.6, 1.4), (0.6, 1.4), (0.6, 1.4)]
    bo = BayesianOptimizer(
        bounds=xi_bounds,
        n_init=args.n_init if not args.debug else 3,
        n_iter=args.n_iter if not args.debug else 2,
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

    # ---- 4. Wandb init ----
    import wandb
    if args.no_wandb or args.debug:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"
    wandb.init(project=args.proj_name, name=args.exptid,
               entity="jackchen1349-shenzhen", group=args.exptid[:3],
               mode=wandb_mode, dir="../../logs", config={
                   "xi_bounds": xi_bounds,
                   "n_init": args.n_init, "n_iter": args.n_iter,
                   "finetune_steps": args.finetune_steps,
                   "task_type": args.task_type,
                   "resumeid": args.resumeid,
               })

    best_fitness_so_far = -float('inf')
    best_xi_so_far = None

    for bo_iter in range(bo.n_total):
        try:
            xi = bo.suggest()
            print(f"\n--- BO iteration {bo_iter + 1}/{bo.n_total} ---")
            print(f"    Candidate xi: [{xi[0]:.4f}, {xi[1]:.4f}, {xi[2]:.4f}, {xi[3]:.4f}]")

            t0 = time.time()
            try:
                _, fitness = evaluate_candidate(xi, resume_path, args, bo_iter)
            except subprocess.TimeoutExpired:
                print(f"    Worker timed out, assigning low fitness")
                fitness = -1000.0
            except subprocess.CalledProcessError as e:
                print(f"    Worker failed (exit code {e.returncode}), assigning low fitness")
                fitness = -1000.0

            elapsed = time.time() - t0
            bo.update(xi, float(fitness))
            print(f"    Fitness: {fitness:.4f}  [{elapsed:.0f}s]")

            if fitness > best_fitness_so_far:
                best_fitness_so_far = fitness
                best_xi_so_far = xi.copy()
                # Save best result metadata
                best_info = {
                    "xi": xi.tolist(),
                    "fitness": float(fitness),
                    "bo_iter": bo_iter,
                    "task_type": args.task_type,
                }
                best_path = os.path.join(log_root, f"best_{bo_iter:03d}.json")
                with open(best_path, "w") as f:
                    json.dump(best_info, f)
                print(f"    -> New best! xi={best_xi_so_far}, fitness={best_fitness_so_far:.4f}")

            wandb.log({
                "bo_iter": bo_iter,
                "xi_0": xi[0], "xi_1": xi[1], "xi_2": xi[2], "xi_3": xi[3],
                "fitness": float(fitness),
                "best_fitness": best_fitness_so_far,
            }, step=bo_iter)

        except Exception as e:
            print(f"    BO iteration {bo_iter} failed: {e}")
            import traceback
            traceback.print_exc()

    # ---- 5. Report results ----
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Task type: {args.task_type}")
    print(f"Best xi: {best_xi_so_far}")
    print(f"Best fitness: {best_fitness_so_far:.4f}")
    print(f"All history:")
    X_hist, y_hist = bo.history
    for i in range(len(y_hist)):
        print(f"  [{i}] xi={X_hist[i].round(4)}, fitness={y_hist[i]:.4f}")

    # Save final results
    final_output = {
        "task_type": args.task_type,
        "best_xi": best_xi_so_far.tolist() if best_xi_so_far is not None else None,
        "best_fitness": float(best_fitness_so_far),
        "history": [
            {"xi": X_hist[i].tolist(), "fitness": float(y_hist[i])}
            for i in range(len(y_hist))
        ],
    }
    final_path = os.path.join(log_root, "final_results.json")
    with open(final_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nResults saved to {log_root}")

    wandb.finish()


if __name__ == "__main__":
    run_finetune()
