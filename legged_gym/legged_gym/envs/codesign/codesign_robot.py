# CoDesignLeggedRobot — LeggedRobot subclass implementing
# spatial domain randomization for structure-control co-design.
#
# Reference: Chen et al., "Structure-control Co-design of Quadruped
# Robots Based on Pre-training-Fine-tuning Framework", ROBOT, 2025.

import os
import numpy as np
import torch

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot import LeggedRobot, euler_from_quaternion
from legged_gym.envs.codesign.codesign_config import CoDesignCfg
from legged_gym.envs.codesign.urdf_utils import (
    URDFCache,
    compute_pd_correction,
    LEG_PREFIXES,
    THIGH_XI_IDX,
    CALF_XI_IDX,
)
from legged_gym.utils.helpers import class_to_dict
import copy
from tqdm import tqdm


class CoDesignLeggedRobot(LeggedRobot):
    """LeggedRobot with spatial domain randomization over leg lengths.

    Each environment receives a robot with independently scaled leg
    segments (front thigh, front calf, rear thigh, rear calf), following
    the paper's spatial domain randomization approach.

    Inherits all reward functions, observation computation, termination
    checks, and terrain management from LeggedRobot.
    """

    def __init__(self, cfg: CoDesignCfg, sim_params, physics_engine, sim_device, headless):
        self._xi_values = None          # (num_envs, 4) — set in _parse_cfg
        self._xi_unique = None          # unique xi rows
        self._xi_to_asset = {}          # xi_tuple -> gym asset handle
        self._urdf_cache = None
        self.p_gains_env = None         # (num_envs, num_dofs)
        self.d_gains_env = None         # (num_envs, num_dofs)

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------

    def _parse_cfg(self, cfg):
        super()._parse_cfg(cfg)

        # Parse spatial randomization config.
        # NOTE: self.device is NOT available yet (set later in BaseTask.__init__),
        # so we store xi_values on CPU and move to device in _init_buffers.
        sr = self.cfg.spatial_rand
        if sr.enable:
            xi_range = sr.xi_range
            num_envs = self.cfg.env.num_envs
            num_groups = int(sr.num_groups)

            if num_groups >= num_envs:
                self._xi_values = torch.zeros(num_envs, 4, dtype=torch.float, device='cpu')
                for i in range(4):
                    self._xi_values[:, i] = torch_rand_float(
                        xi_range[0], xi_range[1], (num_envs, 1), device='cpu'
                    ).squeeze(1)
            else:
                group_size = num_envs // num_groups
                xi_unique = torch.zeros(num_groups, 4, dtype=torch.float, device='cpu')
                for i in range(4):
                    xi_unique[:, i] = torch_rand_float(
                        xi_range[0], xi_range[1], (num_groups, 1), device='cpu'
                    ).squeeze(1)
                repeats = num_envs // num_groups
                remainder = num_envs % num_groups
                self._xi_values = torch.cat([
                    xi_unique.repeat_interleave(repeats, dim=0),
                    xi_unique[:remainder]
                ], dim=0)  # stays on CPU
        else:
            self._xi_values = torch.ones(self.cfg.env.num_envs, 4,
                                         dtype=torch.float, device='cpu')

    # ------------------------------------------------------------------
    # Environment creation with per-morphology URDF assets
    # ------------------------------------------------------------------

    def _create_envs(self):
        """Override to create environments with per-morphology URDF assets.

        For each unique xi vector, generates a scaled URDF and loads it
        as a separate Isaac Gym asset. Environments with the same xi share
        the same asset (via GPU instancing).
        """
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        # Base URDF path
        base_asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)

        # Set up URDF cache
        cache_dir = os.path.join(os.path.dirname(base_asset_path), "codesign_cache")
        self._urdf_cache = URDFCache(base_asset_path, cache_dir)

        # Get unique xi vectors for asset loading
        xi_cpu = self._xi_values.cpu()
        unique_xi, inverse_indices = torch.unique(xi_cpu, dim=0, return_inverse=True)
        self._xi_unique = unique_xi
        self._env_xi_idx = inverse_indices.to(self.device)  # which asset index each env uses

        # Load one asset per unique xi
        print(f"Loading {len(unique_xi)} unique URDF assets...")
        self._xi_to_asset = {}
        first_asset = None

        for idx in range(len(unique_xi)):
            xi_list = unique_xi[idx].tolist()
            urdf_path = self._urdf_cache.get_urdf_path(xi_list)
            asset_root = os.path.dirname(urdf_path)
            asset_file = os.path.basename(urdf_path)
            asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            xi_key = tuple(round(v, 6) for v in xi_list)
            self._xi_to_asset[xi_key] = asset
            if first_asset is None:
                first_asset = asset

        # Use the first asset for metadata extraction
        # (all variants have identical DOF/bone structure)
        robot_asset = first_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # Body and DOF names
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        # Create force sensors on each unique asset
        for xi_key, asset in self._xi_to_asset.items():
            for s in ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]:
                feet_idx = self.gym.find_asset_rigid_body_index(asset, s)
                sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
                self.gym.create_asset_force_sensor(asset, feet_idx, sensor_pose)

        # Penalized and termination contact names
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # Base init state
        base_init_state_list = (self.cfg.init_state.pos +
                                self.cfg.init_state.rot +
                                self.cfg.init_state.lin_vel +
                                self.cfg.init_state.ang_vel)
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # Environment origins
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float,
                                               device=self.device, requires_grad=False)

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper,
                                             int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            if self.cfg.env.randomize_start_pos:
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            if self.cfg.env.randomize_start_yaw:
                rand_yaw_quat = gymapi.Quat.from_euler_zyx(
                    0., 0., self.cfg.env.rand_yaw_range * np.random.uniform(-1, 1))
                start_pose.r = rand_yaw_quat
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            # Select the correct asset for this environment
            xi_list = xi_cpu[self._env_xi_idx[i].cpu().item()].tolist()
            xi_key = tuple(round(v, 6) for v in xi_list)
            asset = self._xi_to_asset[xi_key]

            # Get fresh rigid shape props from THIS asset (cannot deepcopy C++ objects)
            asset_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(asset)
            rigid_shape_props = self._process_rigid_shape_props(
                asset_rigid_shape_props, i)
            self.gym.set_asset_rigid_shape_properties(asset, rigid_shape_props)

            anymal_handle = self.gym.create_actor(env_handle, asset, start_pose,
                                                   "anymal", i,
                                                   self.cfg.asset.self_collisions, 0)
            # Get DOF props from the actual asset used for this env
            asset_dof_props = self.gym.get_asset_dof_properties(asset)
            dof_props = self._process_dof_props(asset_dof_props, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle,
                                                      body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)

            self.attach_camera(i, env_handle, anymal_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)

        # Index body parts (use first env, first actor; same across all morphologies)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long,
                                        device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names),
                                                      dtype=torch.long, device=self.device,
                                                      requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names),
                                                        dtype=torch.long, device=self.device,
                                                        requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # Joint indices
        hip_names = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]
        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long,
                                       device=self.device, requires_grad=False)
        for i, name in enumerate(hip_names):
            self.hip_indices[i] = self.dof_names.index(name)
        thigh_names = ["FR_thigh_joint", "FL_thigh_joint", "RR_thigh_joint", "RL_thigh_joint"]
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long,
                                         device=self.device, requires_grad=False)
        for i, name in enumerate(thigh_names):
            self.thigh_indices[i] = self.dof_names.index(name)
        calf_names = ["FR_calf_joint", "FL_calf_joint", "RR_calf_joint", "RL_calf_joint"]
        self.calf_indices = torch.zeros(len(calf_names), dtype=torch.long,
                                        device=self.device, requires_grad=False)
        for i, name in enumerate(calf_names):
            self.calf_indices[i] = self.dof_names.index(name)

    # ------------------------------------------------------------------
    # PD gains with per-environment correction (Eq 1-2)
    # ------------------------------------------------------------------

    def _init_buffers(self):
        """Extend base buffer init to compute per-environment PD gains."""
        # Move xi values to the correct device now that self.device is available
        self._xi_values = self._xi_values.to(self.device)

        super()._init_buffers()

        if not hasattr(self, 'p_gains_env') or self.p_gains_env is None:
            self.p_gains_env = torch.zeros(self.num_envs, self.num_dofs,
                                           dtype=torch.float, device=self.device,
                                           requires_grad=False)
            self.d_gains_env = torch.zeros(self.num_envs, self.num_dofs,
                                           dtype=torch.float, device=self.device,
                                           requires_grad=False)

        # Compute per-environment PD correction factors
        # For each DOF, find the corresponding xi based on which leg link the DOF belongs to
        # HipX joints (FR_hip_joint, etc.) and HipY joints (FR_thigh_joint, etc.)
        # get the correction from their respective xi values.
        #
        # Mapping of DOF names to xi indices:
        #   - FR_hip_joint, FR_thigh_joint → xi[0] (front thigh)
        #   - FR_calf_joint → xi[1] (front calf)
        #   - FL_hip_joint, FL_thigh_joint → xi[0]
        #   - FL_calf_joint → xi[1]
        #   - RR_hip_joint, RR_thigh_joint → xi[2] (rear thigh)
        #   - RR_calf_joint → xi[3] (rear calf)
        #   - RL_hip_joint, RL_thigh_joint → xi[2]
        #   - RL_calf_joint → xi[3]

        pd_coeffs = self.cfg.spatial_rand.pd_correction_coeffs

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            leg_prefix = name[:2]  # "FR", "FL", "RR", "RL"

            if "calf" in name:
                xi_idx = CALF_XI_IDX.get(leg_prefix, 0)
            else:
                # hip_joint or thigh_joint
                xi_idx = THIGH_XI_IDX.get(leg_prefix, 0)

            # xi value for this DOF across all environments
            xi_env = self._xi_values[:, xi_idx]  # (num_envs,)

            # Compute correction factor (Eq 1): eta = a*xi^3 + b*xi^2 + c*xi + d
            eta = compute_pd_correction(xi_env, pd_coeffs)  # (num_envs,)

            # Apply correction (Eq 2): kp_i = eta * k~p, kd_i = eta * k~d
            self.p_gains_env[:, i] = eta * self.p_gains[i]
            self.d_gains_env[:, i] = eta * self.d_gains[i]

    # ------------------------------------------------------------------
    # Torque computation with per-environment PD gains (Eq 9-11)
    # ------------------------------------------------------------------

    def _compute_torques(self, actions):
        """Override to use per-environment PD gains.

        Eq (9):  tau_target = kp * (q* - q) + kd * (qdot* - qdot)
        Eq (10): tau_scale  = tau_target * alpha  (motor strength)
        Eq (11): tau_real   = clip(tau_scale, -tau_max, tau_max)
        """
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if control_type == "P":
            if not self.cfg.domain_rand.randomize_motor:
                torques = (self.p_gains_env *
                           (actions_scaled + self.default_dof_pos_all - self.dof_pos) -
                           self.d_gains_env * self.dof_vel)
            else:
                torques = (self.motor_strength[0] * self.p_gains_env *
                           (actions_scaled + self.default_dof_pos_all - self.dof_pos) -
                           self.motor_strength[1] * self.d_gains_env * self.dof_vel)
        elif control_type == "V":
            torques = (self.p_gains_env * (actions_scaled - self.dof_vel) -
                       self.d_gains_env * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt)
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ------------------------------------------------------------------
    # Observation with structure parameters (paper Section 1.1.1)
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Override to include leg-length scaling factors xi in priv_latent.

        Paper specification:
          et = [mass(1), COM(3), structure_params(4), friction(1), motor_damping(24)] = 33
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)
        if self.global_counter % 5 == 0:
            self.delta_yaw = self.target_yaw - self.yaw
            self.delta_next_yaw = self.next_target_yaw - self.yaw
        obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            imu_obs,
            0 * self.delta_yaw[:, None],
            self.delta_yaw[:, None],
            self.delta_next_yaw[:, None],
            0 * self.commands[:, 0:2],
            self.commands[:, 0:1],
            (self.env_class != 17).float()[:, None],
            (self.env_class == 17).float()[:, None],
            self.reindex((self.dof_pos - self.default_dof_pos_all) * self.obs_scales.dof_pos),
            self.reindex(self.dof_vel * self.obs_scales.dof_vel),
            self.reindex(self.action_history_buf[:, -1]),
            self.reindex_feet(self.contact_filt.float() - 0.5),
        ), dim=-1)
        priv_explicit = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            0 * self.base_lin_vel,
            0 * self.base_lin_vel,
        ), dim=-1)
        # ---- Only line changed from parent: add self._xi_values (4 dims) ----
        priv_latent = torch.cat((
            self.mass_params_tensor,       # 4: mass + COM offsets
            self.friction_coeffs_tensor,   # 1: friction
            self.motor_strength[0] - 1,    # 12: Kp multipliers
            self.motor_strength[1] - 1,    # 12: Kd multipliers
            self._xi_values,               # 4: leg length scaling [xi0,xi1,xi2,xi3] ← NEW
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.3 - self.measured_heights, -1, 1.)
            self.obs_buf = torch.cat(
                [obs_buf, heights, priv_explicit, priv_latent,
                 self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat(
                [obs_buf, priv_explicit, priv_latent,
                 self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        obs_buf[:, 6:8] = 0  # mask yaw in proprioceptive history
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1)
        )
        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([self.contact_buf[:, 1:],
                       self.contact_filt.float().unsqueeze(1)], dim=1)
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def xi_values(self):
        """Return the per-environment scaling factors, shape (num_envs, 4)."""
        return self._xi_values

    def get_xi_for_env(self, env_id):
        """Return xi vector for a specific environment."""
        return self._xi_values[env_id]
