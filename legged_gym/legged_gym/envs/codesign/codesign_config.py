# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: BSD-3-Clause

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class CoDesignCfg(LeggedRobotCfg):
    """Configuration for the Co-Design LeggedRobot environment.

    Inherits all defaults from LeggedRobotCfg and overrides specific
    sections needed for the structure-control co-design framework.
    """

    # env class inherits directly from parent — observation dimensions
    # are computed automatically from n_proprio, n_scan, etc.
    # Override n_priv_latent to include structure parameters (xi values).
    class env(LeggedRobotCfg.env):
        # trunk mass offset(1) + leg masses(4) + COM(15) + friction(1) + motor(24) + xi(4)
        n_priv_latent = 49  # mass(5) + COM(15) + friction(1) + Kp(12) + Kd(12) + xi(4) = 49
        # Recompute total: 53(proprio) + 132(scan) + 530(history) + 49(latent) + 9(explicit)
        num_observations = 773

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/parkour_quadruped/urdf/parkour_quadruped_a1_style_v3.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        # Paper default PD parameters: k~p = 40, k~d = 1.0  (Eq 2)
        stiffness = {'joint': 40.}
        damping = {'joint': 1.0}
        action_scale = 0.25
        decimation = 4

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = {
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,
            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 1.0,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 1.0,
            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25

    # ---- NEW: Spatial domain randomization for co-design ----
    class spatial_rand:
        enable = True
        # Leg length scaling range: xi_i ~ U(cmin, cmax)
        # Paper uses U(0.6, 1.4)
        xi_range = [0.6, 1.4]

        # Number of unique URDF variants to generate.
        # Each variant is shared by (num_envs / num_groups) environments.
        num_groups = 100

        # PD correction polynomial coefficients (Eq 1):
        #   eta_i = a * xi^3 + b * xi^2 + c * xi + d
        # BO-optimized (30 eval, 500 iters/ea): fitness 0.2608 vs baseline 0.0807 (+223%)
        # eta = -0.455*xi^3 + 0.346*xi^2 + 0.935*xi - 0.279
        pd_correction_coeffs = [-0.4545, 0.3459, 0.9346, -0.2786]  # [a, b, c, d]

        # Anchor-optimized (6 anchors x 6 evals/ea): separate front/rear coefficients
        pd_correction_coeffs_front = [1.143392, -3.473005, 4.272445, -0.978001]
        pd_correction_coeffs_rear  = [4.624430, -13.867083, 14.072439, -3.899633]
        use_separate_front_rear = False  # set True to use front/rear separated coefficients

    # ---- Body mass/COM domain randomization for priv_latent ----
    # Trunk: uses original added_mass_range [0, 3] kg & added_com_range [-0.2, 0.2] m (additive)
    # Hip/Thigh/Calf: independent multiplicative mass & additive COM perturbation
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_body_mass = False
        leg_mass_range = [0.95, 1.05]   # multiplicative factor for hip/thigh/calf
        randomize_body_com = False
        leg_com_range = [-0.01, 0.01]   # additive offset (m) for hip/thigh/calf COM


class CoDesignCfgPPO(LeggedRobotCfgPPO):
    """PPO training configuration for the Co-Design framework.

    Uses discount regularization (gamma_reg = 0.98) by default,
    matching the paper's pre-training phase.
    """

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # Discount regularization: gamma = 0.98 (paper Section 1.2.2)
        # Standard PPO uses 0.99; lower gamma reduces reliance on distant rewards,
        # improving generalization across morphologies.
        gamma = 0.98
        gamma_reg = 0.98  # Used by CoDesignPPO for GAE computation
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'CoDesignPPO'  # Uses discount-regularized PPO
        run_name = ''
        experiment_name = 'codesign_pretrain'
