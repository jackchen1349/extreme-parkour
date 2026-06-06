# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Extreme Parkour with Legged Robots — a CMU research project that trains legged robots (Unitree A1, Go1) to perform parkour maneuvers using reinforcement learning in NVIDIA Isaac Gym. Published at arXiv:2309.14341.

Additionally, this repository includes a **Co-Design framework** (based on Chen et al., ROBOT 2025, `j.cnki.robot.240239.pdf`) for joint optimization of robot leg length and control policy. See the paper for the pre-training-fine-tuning framework, spatial domain randomization, and discount regularization methodology.

**Active environment**: conda env `parkour_1` (Python 3.8, PyTorch 2.3.1, CUDA 12.1). Isaac Gym at `/root/isaacgym/`.

## Commands

All scripts run from `legged_gym/legged_gym/scripts/`. Use absolute paths with `conda run`:

```bash
cd /root/extreme-parkour/legged_gym/legged_gym/scripts

# === Original Extreme Parkour ===
python train.py --exptid 001-00-WHATEVER --device cuda:0
python train.py --exptid 002-00-WHATEVER --device cuda:0 --resume --resumeid 001-00 --delay --use_camera
python play.py --exptid 001-00

# === Co-Design Pre-training (Phase 1) ===
python codesign_pretrain.py --task codesign --exptid 010-00-PRETRAIN --device cuda:0
python codesign_pretrain.py --task codesign --exptid 010-00-PRETRAIN --device cuda:0 --debug

# === Co-Design Fine-tuning + Bayesian Optimization (Phase 2) ===
python codesign_finetune.py --exptid 011-00-FINETUNE --resumeid 010-00 --checkpoint 6000 --device cuda:0
python codesign_finetune.py --exptid 011-00-FINETUNE --resumeid 010-00 --checkpoint 6000 --debug --no_wandb

# === PD Coefficient Optimization (BO search) ===
python codesign_optimize_pd.py --exptid PDOPT-001 --n_init 10 --n_iter 20 --device cuda:0  # full search, ~5-10h
python codesign_optimize_pd.py --exptid PDOPT-001 --debug --device cuda:0                    # debug: 3+5 evals

# === Grid Search PD (legacy, linear-only) ===
python pd_search.py  # 63 experiments, ~1.5h

# === Verification ===
python verify_codesign.py
python /root/extreme-parkour/legged_gym/legged_gym/tests/test_codesign_integration.py

# === Export ===
python save_jit.py --exptid 001-00
```

`--exptid` uses prefix matching on the first 6 characters. Make each prefix unique.

## Architecture

### Original Framework (`rsl_rl/` + `legged_gym/`)

- **`rsl_rl/`** — RL library: PPO (`algorithms/ppo.py`), CoDesignPPO (`algorithms/codesign_ppo.py`), ActorCriticRMA (`modules/actor_critic.py`), RolloutStorage (`storage/rollout_storage.py`), OnPolicyRunner (`runners/on_policy_runner.py`)
- **`legged_gym/legged_gym/envs/base/`** — `BaseTask` → `LeggedRobot`, `LeggedRobotCfg`/`LeggedRobotCfgPPO`
- **`legged_gym/legged_gym/utils/`** — `task_registry.py` (task factory), `terrain.py` (procedural parkour terrain), `helpers.py` (CLI args, config deserialization)
- **`legged_gym/legged_gym/scripts/`** — train.py, play.py, save_jit.py

### Co-Design Extension (`legged_gym/legged_gym/envs/codesign/`)

| Class | Inherits From | Purpose |
|-------|--------------|---------|
| `CoDesignCfg` | `LeggedRobotCfg` | URDF path (v3), `spatial_rand` config, `n_priv_latent=33`, `damping=1.0`, BO-optimized `pd_correction_coeffs` |
| `CoDesignCfgPPO` | `LeggedRobotCfgPPO` | `gamma=0.98`, `gamma_reg=0.98`, `algorithm_class_name='CoDesignPPO'` |
| `CoDesignLeggedRobot` | `LeggedRobot` | Multi-URDF env creation; per-env PD gains (Eq 1-2); xi in privileged observations |
| `CoDesignPPO` | `PPO` | Overrides `compute_returns` to use `gamma_reg=0.98` for GAE (paper Eq 3-4) |

Supporting files:
- `urdf_utils.py` — URDF geometry/mass/inertia/origin scaling per paper Table 2; `URDFCache`; `compute_pd_correction` (Eq 1)
- `bayesian_optimizer.py` — Self-contained GP (Matérn 2.5 kernel) + EI acquisition for black-box optimization
- `codesign_optimize_pd.py` + `_pd_optimize_worker.py` — BO-based PD coefficient search over [a,b,c,d]
- `pd_search.py` + `_pd_worker.py` — Legacy grid search over (xi, eta_ratio) pairs

### PD Coefficient Optimization Pipeline

Two complementary approaches exist for finding optimal `[a,b,c,d]`:

1. **BO search** (`codesign_optimize_pd.py`): Bayesian Optimization over 4D coefficient space. Each candidate evaluated by training a policy from scratch (500 iters, 64 envs) with spatial randomization (random xi per env). Fitness = mean cumulative episode reward over the last 50 episodes (paper Eq 7). Full search: 10 LHS init + 20 EI iterations.

2. **Anchored interpolation** (proposed improvement): Search optimal scalar η at a few anchor xi points via 1D optimization, then fit cubic through the (ξ, η*) pairs.

### Observation Space (Co-Design)

```
[proprio(53) | scandots(132) | priv_explicit(9) | priv_latent(33) | history(530)] = 757
```

Privileged latent (33 dims):
```
mass(1) | COM_offsets(3) | xi_values(4) | friction(1) | Kp_multipliers(12) | Kd_multipliers(12) = 33
```

The 4 xi values: `[ξ_front_thigh, ξ_front_calf, ξ_rear_thigh, ξ_rear_calf]`.

### Environment Registration

Tasks in `legged_gym/envs/__init__.py`:
- `"a1"` → `LeggedRobot` + `A1ParkourCfg`
- `"go1"` → `LeggedRobot` + `Go1RoughCfg`
- `"codesign"` → `CoDesignLeggedRobot` + `CoDesignCfg` + `CoDesignCfgPPO`

Scripts use `args.task` soft-encoding (not hardcoded task names). `codesign_pretrain.py` defaults task to `"codesign"` when `args.task == 'a1'` (the `get_args()` default). `codesign_finetune.py` uses `args.task + "_finetune"` for dynamic registration.

### URDF

**Active model**: `parkour_quadruped_a1_style_v3.urdf` — A1 joints + A1 physics params (mass 4.713) + rectangular box visual/collision geometry. Thigh = 0.20m, Calf = 0.21012m.

**v3 key changes from v2**: Visual: cylinders→boxes with explicit dimensions. Collision: boxes with `rpy=(0,90°,0)` rotation (after rotation, raw X dimension maps to leg-length Z). Foot joint z: -0.21012→-0.2.

**URDF scaling per paper Table 2** (implemented in `build_scaled_urdf`):

| Parameter | Formula | Implementation |
|-----------|---------|---------------|
| Visual/collision origin z | `−lᵢ × ξᵢ / 2` | `set_origin_z()` |
| Visual/collision geometry (length) | `lᵢ × ξᵢ` | `_set_box_dim()` on appropriate axis |
| Mass | `mᵢ × ξᵢ` | Direct set |
| Inertia | `I × ξᵢ` (mass factor only) | `_scale_inertia(ixx, iyy, izz, xi)` |
| Joint origins (knee, ankle) | `z × ξᵢ` | `set_origin_z()` |

**Important**: Inertia scales linearly with ξ (I∝ξ, not I∝ξ³). The paper uses the original link dimensions in the inertia formula, so only the mass factor changes.

PD parameters: `kp=40`, `kd=1.0` (Eq 2, damping changed from 0.7). BO-optimized `pd_correction_coeffs = [-0.4545, 0.3459, 0.9346, -0.2786]` (fitness 0.26 vs baseline 0.08). Per-env gains: `kp_i = η(ξᵢ) × 40`, `kd_i = η(ξᵢ) × 1.0`.

## Gotchas

- **Isaac Gym MUST be imported before torch**. Always `import isaacgym` first.
- **`num_envs` must be divisible by `terrain.num_cols`**.
- **Dots in URDF filenames** confuse Isaac Gym. Use integers: `robot_xi_6000_8000.urdf`.
- **`CoDesignLeggedRobot._parse_cfg` runs before `self.device` is set**. Store tensors on CPU, move to device in `_init_buffers`.
- **Subprocess workers need `LD_LIBRARY_PATH`** including conda lib path (`/root/miniconda3/envs/parkour_1/lib`) for `libpython3.8.so`.
- **`physics_engine` must be `gymapi.SIM_PHYSX` enum**, not integer 0.
- **Negative coefficients in `--coeffs` CLI args**: Use `--coeffs=-0.3,...` (with `=`) to prevent argparse from misinterpreting negative numbers as flags.
- **BO orchestrator imports `bayesian_optimizer` via `importlib`** to avoid `legged_gym.utils.__init__` circular import.
- **`class_to_dict`** converts nested config objects to dicts for wandb logging.
- **`OnPolicyRunner` final model** saved with `tot_iter` naming (e.g., `model_50000.pt`), not `current_learning_iteration`.
