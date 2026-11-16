# envs/tasks/reacher.py
"""Reacher domain with scalable number of planar joints."""

import collections
import math
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import reacher as _dm_reacher
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from lxml import etree
import numpy as np


SUITE = containers.TaggedTasks()

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 20
_BIG_TARGET = 0.05
_SMALL_TARGET = 0.015

# Scaled variants (total number of joints / actuators)
_SIZES = [2, 4, 8, 16, 32, 64]

# Geometry / scaling
_TOTAL_ARM_LENGTH = 0.24      # shoulder -> fingertip center (matches original approx)
_LINK_RADIUS = 0.01           # capsule thickness, constant across sizes
_FINGER_RADIUS = 0.01
_BASE_JOINT_RANGE_DEG = 160.0 # total curvature budget (matches original wrist range)
_MOTOR_GEAR = 0.05            # motor gear from reacher.xml defaults
_STIFFNESS_SCALE = 0.20       # fraction of max stiffness; 0.20 = 20% effort to reach limit
_DAMPING_RATIO = 0.5          # overdamped for stability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_size(size: int) -> int:
    if not isinstance(size, int):
        raise TypeError(f"size must be int, got {type(size).__name__}")
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if size not in _SIZES:
        raise ValueError(f"Unsupported size {size}. Supported sizes: {_SIZES}")
    return size


def _scaled_joint_range_deg(n_joints: int,
                            base_range_deg: float = _BASE_JOINT_RANGE_DEG) -> float:
    """Per-joint symmetric limit in degrees for non-shoulder joints.

    Distributes the total curvature budget (base_range_deg, default 160°)
    equally across (n-1) non-shoulder joints so worst-case total curl
    matches the original 2-joint reacher's wrist range.
    """
    n_non_shoulder = max(1, n_joints - 1)
    return base_range_deg / float(n_non_shoulder)


def _fmt3(x, y, z):
    return f"{x:.6g} {y:.6g} {z:.6g}"


def _build_scaled_arm(n_joints: int,
                      total_length: float = _TOTAL_ARM_LENGTH,
                      link_radius: float = _LINK_RADIUS,
                      finger_radius: float = _FINGER_RADIUS):
    """Build a planar serial chain arm as an lxml <body name='arm'> subtree.

    The shoulder joint is unlimited (free rotation for aiming), matching the
    original dm_control reacher.  All other joints get symmetric +/-limit
    where limit = 160 / (n-1), capping worst-case total curl at 160 deg.

    Returns
    -------
    arm_body : lxml Element
    joint_names : list[str]
    """
    if n_joints < 1:
        raise ValueError(f"n_joints must be >= 1, got {n_joints}")

    link_length = total_length / n_joints
    joint_limit_deg = _scaled_joint_range_deg(n_joints)

    # Stiffness: scaled fraction of (motor_torque / joint_limit_rad).
    # At _STIFFNESS_SCALE=0.25, motors need only 25% effort to reach
    # the joint limit — free enough for RL exploration while still
    # providing centering force against random curling.
    joint_limit_rad = math.radians(joint_limit_deg)
    joint_stiffness = _STIFFNESS_SCALE * _MOTOR_GEAR / joint_limit_rad
    # Matched damping (overdamped for numerical stability)
    # if n_joints == 64:
    #     # For 64-DOF the sqrt formula makes stiffness/damping ratio 15x,
    #     # causing severe underdamped oscillation.  Scale damping linearly
    #     # with stiffness so the ratio matches the well-behaved 2-joint arm.
        # _ref_stiff = _STIFFNESS_SCALE * _MOTOR_GEAR / math.radians(_BASE_JOINT_RANGE_DEG)
        # joint_damping = (_DAMPING_RATIO
        #                  * math.sqrt(_ref_stiff * 1e-3) / _ref_stiff
        #                  * joint_stiffness)
    # else:
    #     joint_damping = _DAMPING_RATIO * math.sqrt(joint_stiffness * 1e-3)

    _ref_stiff = _STIFFNESS_SCALE * _MOTOR_GEAR / math.radians(_BASE_JOINT_RANGE_DEG)
    joint_damping = (_DAMPING_RATIO
                        * math.sqrt(_ref_stiff * 1e-3) / _ref_stiff
                        * joint_stiffness)

    # Root arm body (matches original z offset)
    arm_body = etree.Element("body")
    arm_body.set("name", "arm")
    arm_body.set("pos", _fmt3(0.0, 0.0, 0.01))

    joint_names = []

    # Build nested chain: each link body has one hinge + one capsule geom
    parent = arm_body
    for i in range(n_joints):
        # Joint on current body
        if i == 0:
            joint_name = "shoulder"
            geom_name = "arm"   # preserve original names for n=2 compatibility-ish
        elif i == 1:
            joint_name = "wrist"
            geom_name = "hand"
        else:
            joint_name = f"joint_{i}"
            geom_name = f"link_{i}"

        j = etree.SubElement(parent, "joint")
        j.set("name", joint_name)
        j.set("type", "hinge")
        j.set("axis", "0 0 1")

        if i == 0:
            # Shoulder: unlimited, like the original reacher
            j.set("limited", "false")
        else:
            # Non-shoulder: symmetric +/-limit with stiffness to resist curling
            j.set("limited", "true")
            j.set("range", f"{-joint_limit_deg:.6g} {joint_limit_deg:.6g}")
            j.set("stiffness", f"{joint_stiffness:.6g}")
            j.set("damping", f"{joint_damping:.6g}")

        g = etree.SubElement(parent, "geom")
        g.set("name", geom_name)
        g.set("type", "capsule")
        g.set("fromto", _fmt3(0.0, 0.0, 0.0) + " " + _fmt3(link_length, 0.0, 0.0))
        g.set("size", f"{link_radius:.6g}")
        if i % 2 == 0:
            g.set("rgba", "0.9 0.6 0.2 1")
        else:
            g.set("rgba", "0.65 0.35 0.1 1")

        joint_names.append(joint_name)

        # Create next child body located at end of this link, unless this is the last link
        if i < n_joints - 1:
            child = etree.SubElement(parent, "body")
            child.set("name", f"link_body_{i+1}")
            child.set("pos", _fmt3(link_length, 0.0, 0.0))
            parent = child

    # Fingertip body at end of last link
    finger_body = etree.SubElement(parent, "body")
    finger_body.set("name", "finger")
    finger_body.set("pos", _fmt3(link_length, 0.0, 0.0))

    cam = etree.SubElement(finger_body, "camera")
    cam.set("name", "hand")
    cam.set("pos", _fmt3(0.0, 0.0, 0.2))
    cam.set("mode", "track")

    finger_geom = etree.SubElement(finger_body, "geom")
    finger_geom.set("name", "finger")
    finger_geom.set("type", "sphere")
    finger_geom.set("size", f"{finger_radius:.6g}")
    finger_geom.set("material", "effector")

    return arm_body, joint_names


def make_model(size=2):
    """Return MuJoCo XML string for a scalable-joint reacher."""
    n_joints = _validate_size(size)

    xml_string = resources.GetResource(os.path.join(_TASKS_DIR, "reacher.xml"))
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    worldbody = mjcf.find("worldbody")
    if worldbody is None:
        raise ValueError("Base XML missing <worldbody>")

    # Remove original arm subtree
    old_arm = worldbody.find("./body[@name='arm']")
    if old_arm is None:
        raise ValueError("Base XML missing body[@name='arm']")
    worldbody.remove(old_arm)

    # Remove old actuator section
    old_act = mjcf.find("actuator")
    if old_act is not None:
        mjcf.remove(old_act)

    # Build and insert new scaled arm subtree (insert after root geom if possible)
    new_arm, joint_names = _build_scaled_arm(n_joints=n_joints)
    worldbody.append(new_arm)

    # Rebuild actuator section: one motor per joint (same actuator semantics as original)
    actuator_elem = etree.SubElement(mjcf, "actuator")
    for jname in joint_names:
        a = etree.SubElement(actuator_elem, "motor")
        a.set("name", jname)
        a.set("joint", jname)
        # gear / ctrlrange / ctrllimited inherited from <default><motor ...>

    return etree.tostring(mjcf, pretty_print=True)


def get_model_and_assets(size=2):
    """Returns (xml_string, assets_dict) for the scaled reacher."""
    return make_model(size=size), common.ASSETS


# ---------------------------------------------------------------------------
# Task constructors
# ---------------------------------------------------------------------------

def _make_easy(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
        physics = Physics.from_xml_string(*get_model_and_assets(size=size))
        task = Reacher(target_size=_BIG_TARGET, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit, **environment_kwargs
        )
    fn.__name__ = fn.__qualname__ = f"{size}_easy"
    return fn


def _make_hard(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
        physics = Physics.from_xml_string(*get_model_and_assets(size=size))
        task = Reacher(target_size=_SMALL_TARGET, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit, **environment_kwargs
        )
    fn.__name__ = fn.__qualname__ = f"{size}_hard"
    return fn


# Optional: keep stock aliases for compatibility (2-joint variants)
@SUITE.add("benchmarking", "easy")
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    return _make_easy(2)(
        time_limit=time_limit, random=random, environment_kwargs=environment_kwargs
    )


@SUITE.add("benchmarking")
def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    return _make_hard(2)(
        time_limit=time_limit, random=random, environment_kwargs=environment_kwargs
    )


# Register all scaled tasks (including 2-joint) under a custom tag
for _size in _SIZES:
    _dm_reacher.SUITE.add("custom")(_make_easy(_size))
    _dm_reacher.SUITE.add("custom")(_make_hard(_size))


# ---------------------------------------------------------------------------
# Physics / Task (unchanged task logic)
# ---------------------------------------------------------------------------

class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Reacher domain."""

    def finger_to_target(self):
        """Returns the vector from target to finger in global coordinates."""
        return (
            self.named.data.geom_xpos["target", :2]
            - self.named.data.geom_xpos["finger", :2]
        )

    def finger_to_target_dist(self):
        """Returns distance between finger center and target center in XY plane."""
        return np.linalg.norm(self.finger_to_target())


class Reacher(base.Task):
    """A reacher task to reach the target."""

    def __init__(self, target_size, random=None):
        self._target_size = target_size
        super().__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state at the start of each episode."""
        physics.named.model.geom_size["target", 0] = self._target_size

        # Randomize all rotational joints within limits
        randomizers.randomize_limited_and_rotational_joints(physics, self.random)

        # Randomize target position (same as original; reachable since total arm length fixed)
        angle = self.random.uniform(0, 2 * np.pi)
        radius = self.random.uniform(0.05, 0.20)
        physics.named.model.geom_pos["target", "x"] = radius * np.sin(angle)
        physics.named.model.geom_pos["target", "y"] = radius * np.cos(angle)

        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["position"] = physics.position()
        obs["to_target"] = physics.finger_to_target()
        obs["velocity"] = physics.velocity()
        return obs

    def get_reward(self, physics):
        radii = physics.named.model.geom_size[["target", "finger"], 0].sum()
        return rewards.tolerance(physics.finger_to_target_dist(), (0, radii))