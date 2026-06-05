# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Extreme Parkour with Legged Robots — a CMU research project that trains legged robots (Unitree A1, Go1) to perform parkour maneuvers using reinforcement learning in NVIDIA Isaac Gym. Published at arXiv:2309.14341.

Additionally, this repository now includes a **Co-Design framework** (based on Chen et al., ROBOT 2025) for joint optimization of robot leg length and control policy.

**Active environment**: conda env `parkour_1` (Python 3.8, PyTorch 2.3.1, CUDA 12.1). Isaac Gym at `/root/isaacgym/`.

## Commands

All scripts run from `legged_gym/legged_gym/scripts/`:

```bash
# === Original Extreme Parkour ===
conda activate parkour_1
python train.py --exptid 001-00-WHATEVER --device cuda:0
python train.py --exptid 002-00-WHATEVER --device cuda:0 --resume --resumeid 001-00 --delay --use_camera
python play.py --exptid 001-00

# === Co-Design Pre-training (Phase 1) ===
python codesign_pretrain.py --exptid 010-00-PRETRAIN --device cuda:0
python codesign_pretrain.py --exptid 010-00-PRETRAIN --device cuda:0 --debug  # 64 envs

# === Co-Design Fine-tuning + Bayesian Optimization (Phase 2) ===
python codesign_finetune.py --exptid 011-00-FINETUNE --resumeid 010-00 --checkpoint 6000 --device cuda:0

# === PD Coefficient Search ===
python pd_search.py  # 63 experiments, ~1.5h

# === Verification ===
python verify_codesign.py  # 256-env headless training with monitoring
python /root/extreme-parkour/legged_gym/legged_gym/tests/test_codesign_integration.py

# === Export ===
python save_jit.py --exptid 001-00
```

`--exptid` uses prefix matching on the first 6 characters. Make each prefix unique.

## Architecture

### Original Framework (`rsl_rl/` + `legged_gym/`)

- **`rsl_rl/`** — RL library: PPO (`algorithms/ppo.py`), ActorCriticRMA (`modules/actor_critic.py`), RolloutStorage (`storage/rollout_storage.py`), OnPolicyRunner (`runners/on_policy_runner.py`)
- **`legged_gym/legged_gym/envs/base/`** — `BaseTask` → `LeggedRobot`, `LeggedRobotCfg`/`LeggedRobotCfgPPO`
- **`legged_gym/legged_gym/utils/`** — `task_registry.py` (task factory), `terrain.py` (procedural terrain), `helpers.py` (CLI args)
- **`legged_gym/legged_gym/scripts/`** — train.py, play.py, save_jit.py

### Co-Design Extension (`legged_gym/legged_gym/envs/codesign/`)

| Class | Inherits From | Purpose |
|-------|--------------|---------|
| `CoDesignCfg` | `LeggedRobotCfg` | Adds `spatial_rand` config; uses `parkour_quadruped_a1_style_v2.urdf`; `n_priv_latent=33` |
| `CoDesignCfgPPO` | `LeggedRobotCfgPPO` | Sets `gamma=0.98`, `gamma_reg=0.98`, `algorithm_class_name='CoDesignPPO'` |
| `CoDesignLeggedRobot` | `LeggedRobot` | Multi-URDF env creation; per-env PD gains (Eq 1-2); xi in observations |
| `CoDesignPPO` | `PPO` | Overrides `compute_returns` to use `gamma_reg` for GAE |

Supporting files:
- `urdf_utils.py` — XML-based URDF scaler (Table 2), `URDFCache`, `compute_pd_correction`
- `bayesian_optimizer.py` — Self-contained GP (Matern 2.5) + EI acquisition for ξ search
- `pd_search.py` + `_pd_worker.py` — Multi-process PD coefficient grid search

### Observation Space (Co-Design)

```
[proprio(53) | scandots(132) | priv_explicit(9) | priv_latent(33) | history(530)] = 757
```

Privileged latent (33 dims) per paper specification:
```
mass(1) | COM_offsets(3) | xi_values(4) | friction(1) | Kp_multipliers(12) | Kd_multipliers(12) = 33
```

The 4 xi values correspond to: `[ξ_front_thigh, ξ_front_calf, ξ_rear_thigh, ξ_rear_calf]`.

### Environment Registration

Tasks registered in `legged_gym/envs/__init__.py`:
- `"a1"` → `LeggedRobot` + `A1ParkourCfg`
- `"go1"` → `LeggedRobot` + `Go1RoughCfg`
- `"codesign"` → `CoDesignLeggedRobot` + `CoDesignCfg` + `CoDesignCfgPPO`

Registration calls `task_registry.register()`. Use `make_env(name=..., args=args)` without passing `env_cfg` — it auto-retrieves the registered config and handles `seed` propagation via `get_cfgs()`.

### URDF

Base model: `resources/robots/parkour_quadruped/urdf/parkour_quadruped_a1_style_v2.urdf` — A1 joint names/order + A1 physics params (mass 4.713, inertia) + Lite3 visual/collision geometry. Thigh = 0.20m, Calf = 0.21012m.

PD parameters: `kp=40`, `kd=0.7` (paper Eq 2). Per-env gains stored as `p_gains_env`/`d_gains_env` shape `(num_envs, num_dofs)`.

## Gotchas

- **Isaac Gym MUST be imported before torch**. Always `import isaacgym` first.
- **`num_envs` must be divisible by `terrain.num_cols`**, otherwise CUDA index out-of-bounds.
- **Dots in URDF filenames** confuse Isaac Gym's format detection. Use integers: `robot_xi_6000_8000.urdf` not `xi_0.6_0.8.urdf`.
- **`CoDesignLeggedRobot._parse_cfg` runs before `self.device` is set** (parent calls `super().__init__()` after). Store tensors on CPU, move to device in `_init_buffers`.
- **`class_to_dict`** in `helpers.py` converts nested config objects to dicts for wandb logging.
- **`update_cfg_from_args`** requires args to have `use_camera`, `delay`, `resume`, `headless` attributes.
