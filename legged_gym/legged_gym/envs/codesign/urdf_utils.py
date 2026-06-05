# URDF modification utilities for the co-design framework.
#
# Implements Table 2 from the paper: scales leg link geometry, mass,
# inertia, and joint origins by leg-length scaling factors xi.

import os
import copy
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Per-link nominal parameters extracted from parkour_quadruped_a1_style_v2.urdf
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

# Nominal (xi=1.0) values for thigh links
THIGH_NOMINAL = {
    # visual cylinder
    "visual_origin_z": -0.10,
    "visual_cyl_length": 0.20,
    # collision box
    "collision_origin_z": -0.08,
    "collision_box_zsize": 0.20,
    # mass
    "mass": 1.013,
    # inertial origin z
    "inertial_origin_z": -0.027326,
    # inertia (ixx, iyy, izz) at nominal scale
    "inertia_ixx": 0.005529065,
    "inertia_iyy": 0.005139339,
    "inertia_izz": 0.001367788,
    # calf joint origin z (attached to thigh)
    "calf_joint_origin_z": -0.20,
}

CALF_NOMINAL = {
    # visual cylinder
    "visual_origin_z": -0.10506,
    "visual_cyl_length": 0.21012,
    # collision cylinders (two of them)
    "collision1_origin_z": -0.03,
    "collision1_cyl_length": 0.12,
    "collision2_origin_z": -0.14,
    "collision2_cyl_length": 0.10,
    # mass
    "mass": 0.166,
    # inertial origin z
    "inertial_origin_z": -0.107388,
    # inertia
    "inertia_ixx": 0.002997972,
    "inertia_iyy": 0.003014022,
    "inertia_izz": 3.2426e-05,
    # foot fixed joint origin z (attached to calf)
    "foot_joint_origin_z": -0.21012,
}


def _scale_inertia(ixx: float, iyy: float, izz: float,
                   mass_scale: float, length_scale: float) -> Tuple[float, float, float]:
    """Scale inertia tensor.

    For a cylinder, I ∝ m * l². When both mass and length scale, the
    perpendicular axes (ixx, iyy) scale as m*l² and the axial axis (izz)
    scales as m (or m*r² if radius also scales). The paper scales geometry
    proportionally, so we use:
        ixx' = ixx * mass_scale * length_scale^2
        iyy' = iyy * mass_scale * length_scale^2
        izz' = izz * mass_scale * length_scale^2
    """
    scale = mass_scale * (length_scale ** 2)
    return ixx * scale, iyy * scale, izz * scale


def build_scaled_urdf(base_urdf_path: str, xi: List[float]) -> str:
    """Build a scaled URDF string from the base URDF.

    Parameters
    ----------
    base_urdf_path : str
        Path to the base URDF file (parkour_quadruped_a1_style_v2.urdf).
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

    # ---- Helper: find element by link name ----
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

    def get_origin_z(elem) -> float:
        origin = elem.find("origin")
        if origin is not None:
            xyz = origin.get("xyz", "0 0 0")
            return float(xyz.split()[2])
        return 0.0

    # ---- Process each leg ----
    for prefix in LEG_PREFIXES:
        thigh_xi = xi[THIGH_XI_IDX[prefix]]
        calf_xi  = xi[CALF_XI_IDX[prefix]]

        # --- Thigh link ---
        thigh_link = find_link(f"{prefix}_thigh")
        if thigh_link is not None:
            # Visual: origin z and cylinder length
            visual = thigh_link.find("visual")
            if visual is not None:
                set_origin_z(visual, THIGH_NOMINAL["visual_origin_z"] * thigh_xi)
                geom = visual.find("geometry")
                if geom is not None:
                    cyl = geom.find("cylinder")
                    if cyl is not None:
                        cyl.set("length", f"{THIGH_NOMINAL['visual_cyl_length'] * thigh_xi:.8g}")

            # Collision: origin z and box z-size
            collision = thigh_link.find("collision")
            if collision is not None:
                set_origin_z(collision, THIGH_NOMINAL["collision_origin_z"] * thigh_xi)
                geom = collision.find("geometry")
                if geom is not None:
                    box = geom.find("box")
                    if box is not None:
                        size = box.get("size", "0 0 0")
                        parts = size.split()
                        parts[2] = f"{THIGH_NOMINAL['collision_box_zsize'] * thigh_xi:.8g}"
                        box.set("size", " ".join(parts))

            # Inertial: mass, inertia, origin z
            inertial = thigh_link.find("inertial")
            if inertial is not None:
                set_origin_z(inertial, THIGH_NOMINAL["inertial_origin_z"] * thigh_xi)
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    mass_elem.set("value", f"{THIGH_NOMINAL['mass'] * thigh_xi:.8g}")
                inertia_elem = inertial.find("inertia")
                if inertia_elem is not None:
                    si = _scale_inertia(THIGH_NOMINAL["inertia_ixx"],
                                        THIGH_NOMINAL["inertia_iyy"],
                                        THIGH_NOMINAL["inertia_izz"],
                                        thigh_xi, thigh_xi)
                    inertia_elem.set("ixx", f"{si[0]:.8g}")
                    inertia_elem.set("ixy", "0.0")  # simplified
                    inertia_elem.set("ixz", "0.0")
                    inertia_elem.set("iyy", f"{si[1]:.8g}")
                    inertia_elem.set("iyz", "0.0")
                    inertia_elem.set("izz", f"{si[2]:.8g}")

        # --- Calf joint ---
        calf_joint = find_joint(f"{prefix}_calf_joint")
        if calf_joint is not None:
            set_origin_z(calf_joint, THIGH_NOMINAL["calf_joint_origin_z"] * thigh_xi)

        # --- Calf link ---
        calf_link = find_link(f"{prefix}_calf")
        if calf_link is not None:
            visual = calf_link.find("visual")
            if visual is not None:
                set_origin_z(visual, CALF_NOMINAL["visual_origin_z"] * calf_xi)
                geom = visual.find("geometry")
                if geom is not None:
                    cyl = geom.find("cylinder")
                    if cyl is not None:
                        cyl.set("length", f"{CALF_NOMINAL['visual_cyl_length'] * calf_xi:.8g}")

            # Collisions (two collision elements per calf)
            collisions = calf_link.findall("collision")
            if len(collisions) >= 1:
                set_origin_z(collisions[0], CALF_NOMINAL["collision1_origin_z"] * calf_xi)
                geom = collisions[0].find("geometry")
                if geom is not None:
                    cyl = geom.find("cylinder")
                    if cyl is not None:
                        cyl.set("length", f"{CALF_NOMINAL['collision1_cyl_length'] * calf_xi:.8g}")
            if len(collisions) >= 2:
                set_origin_z(collisions[1], CALF_NOMINAL["collision2_origin_z"] * calf_xi)
                geom = collisions[1].find("geometry")
                if geom is not None:
                    cyl = geom.find("cylinder")
                    if cyl is not None:
                        cyl.set("length", f"{CALF_NOMINAL['collision2_cyl_length'] * calf_xi:.8g}")

            # Inertial
            inertial = calf_link.find("inertial")
            if inertial is not None:
                set_origin_z(inertial, CALF_NOMINAL["inertial_origin_z"] * calf_xi)
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    mass_elem.set("value", f"{CALF_NOMINAL['mass'] * calf_xi:.8g}")
                inertia_elem = inertial.find("inertia")
                if inertia_elem is not None:
                    si = _scale_inertia(CALF_NOMINAL["inertia_ixx"],
                                        CALF_NOMINAL["inertia_iyy"],
                                        CALF_NOMINAL["inertia_izz"],
                                        calf_xi, calf_xi)
                    inertia_elem.set("ixx", f"{si[0]:.8g}")
                    inertia_elem.set("ixy", "0.0")
                    inertia_elem.set("ixz", "0.0")
                    inertia_elem.set("iyy", f"{si[1]:.8g}")
                    inertia_elem.set("iyz", "0.0")
                    inertia_elem.set("izz", f"{si[2]:.8g}")

        # --- Foot fixed joint ---
        foot_joint = find_joint(f"{prefix}_foot_fixed")
        if foot_joint is not None:
            set_origin_z(foot_joint, CALF_NOMINAL["foot_joint_origin_z"] * calf_xi)

    # Return the modified URDF as a string
    # Use a declaration that preserves UTF-8
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


class URDFCache:
    """Cache for generated URDF files to avoid redundant file I/O.

    Each unique xi vector gets its own URDF file on disk, which is
    written once and reused across environments with the same structure.
    """

    def __init__(self, base_urdf_path: str, cache_dir: str = "/tmp/codesign_urdf"):
        self.base_urdf_path = base_urdf_path
        self.cache_dir = cache_dir
        self._cache: Dict[Tuple[float, ...], str] = {}  # (xi0,xi1,xi2,xi3) -> file_path
        os.makedirs(self.cache_dir, exist_ok=True)

    def _xi_key(self, xi: List[float]) -> Tuple[float, ...]:
        """Round xi values to avoid floating-point key mismatches."""
        return tuple(round(float(v), 6) for v in xi)

    def get_urdf_path(self, xi: List[float]) -> str:
        """Return the file path to a URDF with the given scaling factors.

        Generates the URDF on first access; returns cached path on
        subsequent calls.
        """
        key = self._xi_key(xi)
        if key not in self._cache:
            # Use integer representation to avoid dots in filename
            # (dots confuse Isaac Gym's asset format detection)
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
# PD correction factor  (Eq 1)
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
