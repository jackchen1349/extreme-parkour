# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Extreme Parkour with Legged Robots — a CMU research project that trains legged robots (Unitree A1, Go1) to perform parkour maneuvers using reinforcement learning in NVIDIA Isaac Gym. Published at arXiv:2309.14341.

Additionally, this repository includes a **Co-Design framework** (based on Chen et al., ROBOT 2025, `j.cnki.robot.240239.pdf`) for joint optimization of robot leg length and control policy. See the paper for the pre-training-fine-tuning framework, spatial domain randomization, and discount regularization methodology.

**Active environment**: conda env `parkour_1` (Python 3.8, PyTorch 2.3.1, CUDA 12.1). Isaac Gym at `/root/isaacgym/`. The `install.sh` script is **legacy** — it targets torch 1.10+cu113; the active environment was built with newer dependencies.

## Commands

All scripts run from `legged_gym/legged_gym/scripts/`. Use absolute paths with `conda run`:

```bash
cd /root/extreme-parkour/legged_gym/legged_gym/scripts

# === Original Extreme Parkour ===
python train.py --exptid 001-00-WHATEVER --device cuda:0
python train.py --exptid 002-00-WHATEVER --device cuda:0 --resume --resumeid 001-00 --delay --use_camera
python play.py --exptid 001-00
python play.py --task codesign --exptid 010-00 --xi "0.8,1.2,0.9,1.1"  # play with custom leg length
python play.py --task codesign --exptid 010-00                          # play with random leg length

# === Co-Design Pre-training (Phase 1) ===
python codesign_pretrain.py --task codesign --exptid 010-00-PRETRAIN --device cuda:0
python codesign_pretrain.py --task codesign --exptid 010-00-PRETRAIN --device cuda:0 --debug

# === Co-Design Fine-tuning + Bayesian Optimization (Phase 2) ===
# --task_type controls terrain: jump (gap only) | high_jump (hurdle only) | both (all 5 types)
python codesign_finetune.py --exptid 011-00-FINETUNE --resumeid 010-01 --checkpoint 9500 --device cuda:0 --task_type jump
python codesign_finetune.py --exptid 011-00-FINETUNE --resumeid 010-01 --checkpoint 9500 --debug --no_wandb

# === Worker standalone test (single xi evaluation) ===
python _finetune_worker.py --xi "0.8,1.2,0.9,1.1" \
    --resume_path /path/to/model_9500.pt --finetune_steps 5 --eval_episodes 1 \
    --task_type jump --tag test_001 --out /tmp/result.json --device cuda:0

# === Evaluate with custom leg length ===
python evaluate.py --task codesign --exptid 011-JUMP --checkpoint 400 \
    --xi "0.8,1.2,0.9,1.1" --task_type jump --device cuda:0

# === PD Coefficient Optimization (BO search) ===
python codesign_optimize_pd.py --exptid PDOPT-001 --n_init 10 --n_iter 20 --device cuda:0  # full search, ~5-10h
python codesign_optimize_pd.py --exptid PDOPT-001 --debug --device cuda:0                    # debug: 3+5 evals

# === Grid Search PD (legacy, linear-only) ===
python pd_search.py  # 63 experiments, ~1.5h

# === Verification ===
python verify_codesign.py
python /root/extreme-parkour/legged_gym/legged_gym/tests/test_codesign_integration.py
python /root/extreme-parkour/legged_gym/legged_gym/tests/test_codesign_phase1.py
python /root/extreme-parkour/legged_gym/legged_gym/tests/test_codesign_phase2.py
python /root/extreme-parkour/legged_gym/legged_gym/tests/test_codesign_phase3.py

# === Anchor-Point PD Search (Phase 1) ===
python anchor_search.py --exptid ANCHOR-001 --device cuda:0
python anchor_search.py --exptid ANCHOR-001 --debug --device cuda:0  # 2+3 per anchor, ~30 evals

# === Coefficient Comparison ===
python _compare_worker.py --tag BASELINE --out /tmp/baseline.json --device cuda:0
python _compare_worker.py --tag ANCHOR --out /tmp/anchor.json --use_separate --device cuda:0
python compare_coeffs.py --device cuda:0  # Compare anchor-derived coefficients vs BO baseline (500 iters)

# === Export ===
python save_jit.py --exptid 001-00

# === PD Analysis Tools ===
python pd_calibrate.py  # PD coefficient calibration experiment for xi->eta mapping
python pd_compare.py    # Compare PD correction coefficient sets

# === Visualization & Evaluation ===
python visualize.py --exptid 001-00  # t-SNE + state coverage analysis
python evaluate.py --task codesign --exptid 010-00 --xi "0.8,1.2,0.9,1.1" --task_type jump --device cuda:0

# === Remote Fetch (from CMU cluster) ===
python fetch.py  # SSH-based log/model fetching
```

`--exptid` uses prefix matching on the first 6 characters. Make each prefix unique.

## Architecture

### Original Framework (`rsl_rl/` + `legged_gym/`)

- **`rsl_rl/`** — RL library: PPO (`algorithms/ppo.py`), CoDesignPPO (`algorithms/codesign_ppo.py`), ActorCriticRMA (`modules/actor_critic.py`), RolloutStorage (`storage/rollout_storage.py`), OnPolicyRunner (`runners/on_policy_runner.py`)
- **`rsl_rl/modules/`** — Also includes `Estimator`, `Discriminator*` variants (LSD, ContDIAYN) for vision-based supervision, and depth backbone variants: `DepthOnlyFCBackbone58x87`, `RecurrentDepthBackbone`, `StackDepthEncoder`. There's also an unused `actor_critic_recurrent.py` (LSTM variant, commented out).
- **`legged_gym/legged_gym/envs/base/`** — `BaseConfig` → `BaseTask` → `LeggedRobot`, `LeggedRobotCfg`/`LeggedRobotCfgPPO`
- **`legged_gym/legged_gym/utils/`** — `task_registry.py` (task factory), `terrain.py` (procedural parkour terrain), `helpers.py` (CLI args, config deserialization)
- **`legged_gym/legged_gym/scripts/`** — train.py, play.py, save_jit.py
- **`legged_gym/legged_gym/scripts/legged_gym/envs/`** — Mirror directory with stripped-down copies of `a1/a1_config.py` and `base/legged_robot_config.py`. Used as a path-resolution hack by `save_jit.py` (which does `sys.path.append("../../../rsl_rl")`).

**`n_proprio` discrepancy**: The base config uses `n_proprio = 53` (3+2+3+4+36+5), but `save_jit.py` uses `n_proprio = 49` (3+2+3+4+36+4+1) for the hardware-deployment JIT model. This is intentional — the vision/deployment model uses a slightly different proprioceptive input set.

### Co-Design Extension (`legged_gym/legged_gym/envs/codesign/`)

| Class | Inherits From | Purpose |
|-------|--------------|---------|
| `CoDesignCfg` | `LeggedRobotCfg` | URDF path (v3), `spatial_rand` config, `n_priv_latent=49`, `damping=1.0`, front/rear-separated `pd_correction_coeffs` |
| `CoDesignCfgPPO` | `LeggedRobotCfgPPO` | `gamma=0.98`, `gamma_reg=0.98`, `algorithm_class_name='CoDesignPPO'` |
| `CoDesignLeggedRobot` | `LeggedRobot` | Multi-URDF env creation; per-env PD gains (Eq 1-2) for HipY+Knee only (HipX excluded); xi in privileged observations |
| `CoDesignPPO` | `PPO` | Overrides `compute_returns` to use `gamma_reg=0.98` for GAE (paper Eq 3-4) |

Supporting files:
- `urdf_utils.py` — URDF geometry/mass/inertia/origin scaling per paper Table 2; `URDFCache`; `compute_pd_correction` (Eq 1)
- `bayesian_optimizer.py` — Self-contained GP (Matérn 2.5 kernel) + EI acquisition for black-box optimization (in `utils/`)
- `codesign_optimize_pd.py` + `_pd_optimize_worker.py` — BO-based PD coefficient search over [a,b,c,d]
- `anchor_search.py` + `_anchor_worker.py` — Anchor-point 2D BO search with front/rear separation
- `_compare_worker.py` — Single-coefficient-set evaluator (full spatial rand, 500 iters)
- `pd_search.py` + `_pd_worker.py` — Legacy grid search over (xi, eta_ratio) pairs

**Body mass/COM extraction** (in `codesign_robot.py` `_create_envs`):
- Builds `_body_idx` name→index map from Isaac Gym `get_asset_rigid_body_names`
- Reads raw masses/COM from `body_props` before parent's `_process_rigid_body_props`
- Applies domain randomization separately: trunk → parent's additive rand, legs → multiplicative mass + additive COM
- Stores in `body_mass_tensor` (5 dims): trunk uses `mass_params[0]` (offset), legs use perturbed actual values
- Stores in `body_com_tensor` (15 dims): trunk uses `mass_params[1:4]` (COM offset), legs use perturbed actual values
- Helper: `_average_body_com(props, indices)` averages COM across left/right body indices
- Hip mass is NOT stored separately (only thigh_f, thigh_r, calf_f, calf_r leg masses)

### PD Coefficient Optimization Pipeline

Three approaches exist for finding optimal PD correction coefficients:

1. **BO search** (`codesign_optimize_pd.py`): 4D Bayesian Optimization over `[a,b,c,d]` space. Each candidate evaluated by training a policy from scratch (500 iters, 64 envs) with spatial randomization. Full search: 10 LHS init + 20 EI iterations.

2. **Anchor-point interpolation** (`anchor_search.py`): 2D BO at 6 representative ξ anchors [0.60, 0.76, 0.92, 1.08, 1.24, 1.40]. Per anchor: 4 LHS + 8 BO iterations searching optimal (η_front, η_rear). Total: 72 evaluations ≈ 12h. Fits separate cubic polynomials for front and rear legs. Debug mode: 2 LHS + 3 BO per anchor. Worker: `_anchor_worker.py`.

3. **Coefficient comparison** (`_compare_worker.py`): Evaluates a single coefficient set with full spatial randomization at 500 iters. Supports front/rear separation via `--use_separate` flag. Used for A/B testing coefficient sets.

### Co-Design Fine-tuning (Phase 2) — Subprocess Architecture

`codesign_finetune.py` (orchestrator) + `_finetune_worker.py` (worker) use a **subprocess pattern** to work around Isaac Gym's single-simulation-per-process limit:

- **Orchestrator**: runs BO logic only (no Isaac Gym envs). Spawns one subprocess per candidate ξ via `subprocess.run()`.
- **Worker**: creates Isaac Gym, registers CoDesignLeggedRobot with `target_xi`, fine-tunes, evaluates fitness, writes result JSON, exits → OS reclaims all GPU/PhysX resources.
- Same pattern as `codesign_optimize_pd.py` + `_pd_optimize_worker.py`.
- Worker uses `get_args(extra_parameters=[...])` (NOT `SimpleNamespace`) for standard Isaac Gym CLI args. Custom params must avoid name conflicts with `get_args()` defaults — notably `--checkpoint` is `int` in `get_args()`, so the worker uses `--resume_path` instead.
- `--task_type` controls terrain via `TASK_TERRAIN_MAP` (defined in worker, same structure as `_finetune_worker.py` lines 51-58): `jump` → parkour_gap only, `high_jump` → parkour_hurdle only, `both` → all 5 types.
- Debug mode follows `codesign_pretrain.py` pattern: `num_envs=64, rows=5, cols=8, headless=False, no_wandb=True`.

### Observation Space (Co-Design)

```
[proprio(53) | scandots(132) | priv_explicit(9) | priv_latent(49) | history(530)] = 773
```

Privileged latent (49 dims) — **body physics parameters** with domain randomization:

```
body_mass(5):      [Trunk_offset | Thigh_front | Thigh_rear | Calf_front | Calf_rear]
body_COM(15):      [Trunk_COM_offset | Thigh_front_xyz | Thigh_rear_xyz | Calf_front_xyz | Calf_rear_xyz]
friction(1) | Kp(12) | Kd(12) | xi(4) = 49
```

**Grouping**: left-right symmetric (FR=FL, RR=RL). Front/rear separated. Trunk mass/COM uses the **offset** from domain randomization (`mass_params = [rand_mass, rand_com_x, rand_com_y, rand_com_z]`), not the post-randomization actual value — matching the parent class `LeggedRobot` behavior. Leg masses are multiplicative-perturbed (±5%); leg COM values are additive-perturbed (±1cm). Hip mass is not included in priv_latent.

The 4 xi values: `[ξ_front_thigh, ξ_front_calf, ξ_rear_thigh, ξ_rear_calf]`.

Body names from Isaac Gym (17 rigid bodies after fixed-joint merging): `base` (merged base+trunk+camera+imu), `FL_hip`, `FL_thigh`, `FL_calf`, `FL_foot`, `FR_hip`, `FR_thigh`, `FR_calf`, `FR_foot`, `RL_hip`, `RL_thigh`, `RL_calf`, `RL_foot`, `RR_hip`, `RR_thigh`, `RR_calf`, `RR_foot`.

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

PD parameters: `kp=40`, `kd=1.0`. Per-env gains: `kp_i = η(ξᵢ) × 40`, `kd_i = η(ξᵢ) × 1.0`.

**PD correction applies to HipY + Knee only (8 DOFs)**. HipX (`*_hip_joint`, hip abduction/adduction, 4 DOFs) keeps base PD gains (kp=40, kd=1.0) without any xi-based correction. The mapping:

- `*_hip_joint` → no correction (base gains)
- `*_thigh_joint` → xi[THIGH_XI_IDX] → eta × base
- `*_calf_joint` → xi[CALF_XI_IDX] → eta × base

**Single polynomial mode** (default): BO-optimized `pd_correction_coeffs = [-0.4545, 0.3459, 0.9346, -0.2786]` applied to all DOFs.

**Front/rear separated mode** (`use_separate_front_rear = True`): Anchor-optimized values:
- Front (FR/FL): `[1.1434, -3.4730, 4.2724, -0.9780]`
- Rear (RR/RL): `[4.6244, -13.8671, 14.0724, -3.8996]`

**Body mass/COM domain randomization** (configured in `domain_rand`):
- Trunk: additive mass `added_mass_range [0, 3] kg`, additive COM `added_com_range [-0.2, 0.2] m` (inherited from parent). Only the **offset** is stored in priv_latent.
- Legs (Thigh/Calf): multiplicative mass `leg_mass_range [0.95, 1.05]`, additive COM `leg_com_range [-0.01, 0.01] m`. Actual perturbed values stored in priv_latent.
- Hip mass is randomized in simulation but NOT included in priv_latent.

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
- **Isaac Gym merges links connected by fixed joints**. The trunk, camera_box, imu_link are merged into a single rigid body named `"base"`. Use `body_props[body_idx["base"]]` to access trunk mass/COM, not `"trunk"`. Only 17 rigid bodies exist (not 24 as in the URDF).
- **`friction_coeffs_tensor` is 1D** `(num_envs,)` — unsqueeze to `(num_envs, 1)` before concatenating into priv_latent.
- **`body_mass_tensor` and `body_com_tensor`**: Trunk entries store the domain-randomization **offset** (`mass_params[0]` and `mass_params[1:4]`), NOT the post-randomization actual values. Leg entries store perturbed actual values. `body_mass_tensor` is 5 dims (no hip). Read raw values from `body_props` *before* `_process_rigid_body_props` to avoid double-randomization with the parent class.
- **Circular import**: `legged_gym.utils.__init__` → `task_registry` → `legged_gym.envs.base.legged_robot_config` → `legged_gym.envs.__init__` → `task_registry`. Fix by importing from `legged_gym.envs` BEFORE `legged_gym.utils.helpers`.
- **`wandb.init()` required before `runner.learn()`**: `OnPolicyRunner.learn()` calls `wandb.log()` unconditionally. Worker scripts must call `wandb.init()` first (use `mode="disabled"` if no logging needed). The `init_wandb` parameter in `OnPolicyRunner.__init__` is a **dead parameter** — accepted but never used.
- **`torch.inference_mode()` for eval after `runner.learn()`**: PPO rollout uses `torch.inference_mode()`, tainting env buffer tensors. Post-training evaluation MUST also use `torch.inference_mode()` when calling `env.step()` or in-place ops on `obs_history_buf`/`contact_buf` will crash with "Inplace update to inference tensor outside InferenceMode".
- **`--checkpoint` name conflict**: `get_args()` defines `--checkpoint` as `int` (default -1). Worker scripts that need a model path must use a different name like `--resume_path` to avoid type conflicts.
- **Worker args via `get_args()`**: Prefer `get_args(extra_parameters=[...])` over `types.SimpleNamespace` for worker CLI — it provides all standard Isaac Gym defaults (`--headless`, `--sim_device`, `--physics_engine`, etc.) for free.
