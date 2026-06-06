# URDF modification utilities for the co-design framework.
#
# Implements Table 2 from the paper: scales leg link geometry, mass,
# inertia, and joint origins by leg-length scaling factors xi.
#
# Reference: Chen et al., "Structure-control Co-design of Quadruped
# Robots Based on Pre-training-Fine-tuning Framework", ROBOT, 2025.
#
# Nominal parameters extracted from parkour_quadruped_a1_style_v3.urdf.
# v3 geometry: visual/collision are boxes with explicit dimensions;
# collision boxes have rpy=(0, 90deg, 0) so the raw X dimension maps
# to the leg-length Z axis.

import os
import copy
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Per-link nominal parameters extracted from parkour_quadruped_a1_style_v3.urdf
# ---------------------------------------------------------------------------

# The 4 scaling factors:
#   xi[0] = front thigh  (FR_thigh, FL_thigh)
#   xi[1] = front calf   (FR_calf,  FL_calf,  plus calf joint and foot joint)
#   xi[2] = rear  thigh  (RR_thigh, RL_thigh)
#   xi[3] = rear  calf   (RR_calf,  RL_calf,  plus calf joint and foot joint)

# Leg link ordering: FR, FL, RR, RL
LEG_PREFIXES = ["FR", "FL", "RR", "RL"]

# Index mapping: prefix -> which xi index (0=front_thigh, 1=front_calf, 2=rear_thigh, 3=rear_calf)
THIGH_XI_IDX = {"FR": 0, "FL": 0, "RR": 2, "RL": 2}
CALF_XI_IDX  = {"FR": 1, "FL": 1, "RR": 3, "RL": 3}

# Nominal (xi=1.0) values for thigh links (v3: box geometry)
THIGH_NOMINAL = {
    # visual box (0.022, 0.022, 0.20) — only z-size scales with xi
    "visual_origin_z": -0.10,
    "visual_scale_axis": "z",          # which box dimension is the leg length
    "visual_length_val": 0.20,
    "visual_other_dims": [0.022, 0.022],  # cross-section (x, y or y, x)
    # collision box rpy=(0, 90deg, 0), raw(x, y, z) = (0.2, 0.0245, 0.034)
    # After Y-rotation: raw X → new Z (leg length). Must scale raw X.
    "collision_origin_z": -0.1,
    "collision_scale_axis": "x",       # raw X becomes leg length after rotation
    "collision_length_val": 0.2,
    "collision_other_dims": [0.0245, 0.034],  # (raw Y, raw Z) = cross-section
    # mass
    "mass": 1.013,
    # inertial origin z
    "inertial_origin_z": -0.027326,
    # inertia (ixx, iyy, izz) at nominal scale — scaled by xi per Table 2
    "inertia_ixx": 0.005529065,
    "inertia_iyy": 0.005139339,
    "inertia_izz": 0.001367788,
    # calf joint origin z (attached to thigh)
    "calf_joint_origin_z": -0.20,
}

CALF_NOMINAL = {
    # visual box (0.015, 0.015, 0.21012) — only z-size scales with xi
    "visual_origin_z": -0.10506,
    "visual_scale_axis": "z",
    "visual_length_val": 0.21012,
    "visual_other_dims": [0.015, 0.015],
    # collision box rpy=(0, 90deg, 0), raw(x, y, z) = (0.2, 0.016, 0.016)
    # After Y-rotation: raw X → new Z (leg length). Must scale raw X.
    "collision_origin_z": -0.1,
    "collision_scale_axis": "x",
    "collision_length_val": 0.2,
    "collision_other_dims": [0.016, 0.016],
    # mass
    "mass": 0.166,
    # inertial origin z
    "inertial_origin_z": -0.107388,
    # inertia
    "inertia_ixx": 0.002997972,
    "inertia_iyy": 0.003014022,
    "inertia_izz": 3.2426e-05,
    # foot fixed joint origin z (v3: -0.2, v2 was -0.21012)
    "foot_joint_origin_z": -0.2,
}


def _scale_inertia(ixx: float, iyy: float, izz: float,
                   mass_scale: float) -> Tuple[float, float, float]:
    """Scale inertia tensor per paper Table 2: I ∝ mass ∝ xi.

    The paper uses the original link dimensions in the inertia formula
    (I = (m/12)*(b² + l²) with original l), so only the mass factor
    changes: I_new = I_old * mass_scale = I_old * xi.
    """
    return ixx * mass_scale, iyy * mass_scale, izz * mass_scale


def _set_box_dim(elem, axis: str, new_val: float, other_vals: List[float]):
    """Set box size dimension identified by axis ('x','y','z').

    For a box <box size="sx sy sz"/>, sets the dimension corresponding
    to `axis` to `new_val` and leaves the other two at `other_vals`.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    idx = axis_map[axis]
    parts = [0.0, 0.0, 0.0]
    other_idx = 0
    for i in range(3):
        if i == idx:
            parts[i] = new_val
        else:
            parts[i] = other_vals[other_idx]
            other_idx += 1
    elem.set("size", f"{parts[0]:.8g} {parts[1]:.8g} {parts[2]:.8g}")


def build_scaled_urdf(base_urdf_path: str, xi: List[float]) -> str:
    """Build a scaled URDF string from the base URDF.

    Implements Table 2 scaling: origins, geometry, mass, inertia all
    scale by the appropriate xi factor. Only the leg-length dimension
    is scaled; cross-sections remain unchanged.

    Parameters
    ----------
    base_urdf_path : str
        Path to the base URDF file.
    xi : list of float, length 4
        Scaling factors: [front_thigh, front_calf, rear_thigh, rear_calf].

    Returns
    -------
    str
        The modified URDF as an XML string.
    """
    assert len(xi) == 4, f"xi must have 4 elements, got {len(xi)}"

    tree = ET.parse(base_urdf_path)
    root = tree.getroot()

    # ---- helpers ----
    def find_link(name: str):
        for link in root.findall("link"):
            if link.get("name") == name:
                return link
        return None

    def find_joint(name: str):
        for joint in root.findall("joint"):
            if joint.get("name") == name:
                return joint
        return None

    def set_origin_z(elem, new_z: float):
        """Set the z component of an <origin> element's xyz attribute."""
        origin = elem.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            parts = xyz.split()
            parts[2] = f"{new_z:.8g}"
            origin.set("xyz", " ".join(parts))

    def find_box(elem):
        """Find the <box> child of a geometry element, if any."""
        geom = elem.find("geometry")
        if geom is not None:
            return geom.find("box")
        return None

    def process_link(link, nominal: dict, xi_val: float):
        """Apply Table 2 scaling to a thigh or calf link."""
        # --- Visual (Table 2: visual origin z, geometry z) ---
        visual = link.find("visual")
        if visual is not None:
            set_origin_z(visual, nominal["visual_origin_z"] * xi_val)
            box = find_box(visual)
            if box is not None:
                new_len = nominal["visual_length_val"] * xi_val
                axis = nominal["visual_scale_axis"]
                other = nominal["visual_other_dims"]
                _set_box_dim(box, axis, new_len, other)

        # --- Collision (Table 2: collision origin z, geometry) ---
        # v3 has ONE collision element per link (previously v2 had two for calf)
        collisions = link.findall("collision")
        for collision in collisions:
            set_origin_z(collision, nominal["collision_origin_z"] * xi_val)
            box = find_box(collision)
            if box is not None:
                new_len = nominal["collision_length_val"] * xi_val
                axis = nominal["collision_scale_axis"]
                other = nominal["collision_other_dims"]
                _set_box_dim(box, axis, new_len, other)

        # --- Inertial (Table 2: inertial origin z, mass, inertia) ---
        inertial = link.find("inertial")
        if inertial is not None:
            set_origin_z(inertial, nominal["inertial_origin_z"] * xi_val)
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                mass_elem.set("value", f"{nominal['mass'] * xi_val:.8g}")
            inertia_elem = inertial.find("inertia")
            if inertia_elem is not None:
                si = _scale_inertia(nominal["inertia_ixx"],
                                    nominal["inertia_iyy"],
                                    nominal["inertia_izz"],
                                    xi_val)
                inertia_elem.set("ixx", f"{si[0]:.8g}")
                inertia_elem.set("ixy", "0.0")
                inertia_elem.set("ixz", "0.0")
                inertia_elem.set("iyy", f"{si[1]:.8g}")
                inertia_elem.set("iyz", "0.0")
                inertia_elem.set("izz", f"{si[2]:.8g}")

    # ---- Process each leg ----
    for prefix in LEG_PREFIXES:
        thigh_xi = xi[THIGH_XI_IDX[prefix]]
        calf_xi  = xi[CALF_XI_IDX[prefix]]

        # Thigh link
        thigh_link = find_link(f"{prefix}_thigh")
        if thigh_link is not None:
            process_link(thigh_link, THIGH_NOMINAL, thigh_xi)

        # Calf joint (Table 2: knee joint origin z)
        calf_joint = find_joint(f"{prefix}_calf_joint")
        if calf_joint is not None:
            set_origin_z(calf_joint, THIGH_NOMINAL["calf_joint_origin_z"] * thigh_xi)

        # Calf link
        calf_link = find_link(f"{prefix}_calf")
        if calf_link is not None:
            process_link(calf_link, CALF_NOMINAL, calf_xi)

        # Foot fixed joint (Table 2: ankle joint origin z)
        foot_joint = find_joint(f"{prefix}_foot_fixed")
        if foot_joint is not None:
            set_origin_z(foot_joint, CALF_NOMINAL["foot_joint_origin_z"] * calf_xi)

    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


# ---------------------------------------------------------------------------
# URDF Cache
# ---------------------------------------------------------------------------

class URDFCache:
    """Cache for generated URDF files to avoid redundant file I/O.

    Each unique xi vector gets its own URDF file on disk, which is
    written once and reused across environments with the same structure.
    """

    def __init__(self, base_urdf_path: str, cache_dir: str = "/tmp/codesign_urdf"):
        self.base_urdf_path = base_urdf_path
        self.cache_dir = cache_dir
        self._cache: Dict[Tuple[float, ...], str] = {}
        os.makedirs(self.cache_dir, exist_ok=True)

    def _xi_key(self, xi: List[float]) -> Tuple[float, ...]:
        """Round xi values to avoid floating-point key mismatches."""
        return tuple(round(float(v), 6) for v in xi)

    def get_urdf_path(self, xi: List[float]) -> str:
        """Return the file path to a URDF with the given scaling factors."""
        key = self._xi_key(xi)
        if key not in self._cache:
            # Integer representation to avoid dots in filename
            int_parts = [str(int(round(v * 10000))) for v in key]
            filename = f"robot_xi_{'_'.join(int_parts)}.urdf"
            filepath = os.path.join(self.cache_dir, filename)
            if not os.path.exists(filepath):
                urdf_str = build_scaled_urdf(self.base_urdf_path, list(xi))
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(urdf_str)
            self._cache[key] = filepath
        return self._cache[key]

    def clear(self):
        """Remove all cached URDF files from disk."""
        import shutil
        for path in self._cache.values():
            if os.path.exists(path):
                os.remove(path)
        self._cache.clear()
        if os.path.exists(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# PD correction factor (Eq 1)
# ---------------------------------------------------------------------------

def compute_pd_correction(xi: float, coeffs: List[float]) -> float:
    """Compute the PD correction factor eta for a given scaling factor xi.

    Eq (1): eta = a * xi^3 + b * xi^2 + c * xi + d

    Parameters
    ----------
    xi : float
        Leg length scaling factor.
    coeffs : list of float [a, b, c, d]
        Polynomial coefficients.

    Returns
    -------
    float
        Correction factor eta.
    """
    a, b, c, d = coeffs
    return a * xi**3 + b * xi**2 + c * xi + d
