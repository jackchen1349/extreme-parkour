# Structure-Control Co-design package for quadruped robots.
# Implements the pre-training-fine-tuning framework from:
#   Chen et al., "Structure-control Co-design of Quadruped Robots
#   Based on Pre-training-Fine-tuning Framework", ROBOT, 2025.

from .codesign_config import CoDesignCfg, CoDesignCfgPPO
from .codesign_robot import CoDesignLeggedRobot
from .urdf_utils import build_scaled_urdf, URDFCache, compute_pd_correction
