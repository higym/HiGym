"""Humanoid CMU DOF-Scaled Environment.

Registers humanoid_CMU stand/walk/run tasks with intermediate joints inserted
between existing joints in the arms and legs, increasing DOF from 56 to 80.

The approach inserts one intermediate body at the geometric midpoint between
each consecutive pair of jointed bodies in the major arm and leg segments.
Each intermediate mirrors the parent body's joint axes.

Insertion map (per side):
  Legs:  femur->tibia (+3j), tibia->foot (+1j), foot->toes (+2j)
  Arms:  clavicle->humerus (+2j), humerus->radius (+3j), radius->wrist (+1j)

Scaling strategy:
  - Joint ranges: 50/50 split (alpha=1.0, preserves total curvature)
  - Gear: 50/50 split (preserves total torque budget per segment)
  - Damping/stiffness: inherit parent's class
  - Mass: explicit per-geom mass (parent geom split, half mass each)
  - Geom radius: constant (no shrinkage)
"""

import collections
import math
import os
import re

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import humanoid_CMU as _dm_humanoid_CMU
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources
from lxml import etree
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02
_STAND_HEIGHT = 1.4
_WALK_SPEED = 1
_RUN_SPEED = 10

# Scaling parameters (can be tuned based on test results)
_RANGE_SCALE = 0.5    # Each of 2 joints gets this fraction of original range
_GEAR_SCALE = 0.5     # Each of 2 actuators gets this fraction of original gear

# ---------------------------------------------------------------------------
# Default stiffness/damping classes from the original humanoid_CMU.xml
# ---------------------------------------------------------------------------
_STIFFNESS_CLASSES = {
    'main':         {'stiffness': 0.1,  'damping': 1.0},
    'stiff_low':    {'stiffness': 0.5,  'damping': 4.0},
    'stiff_medium': {'stiffness': 10.0, 'damping': 5.0},
    'stiff_high':   {'stiffness': 30.0, 'damping': 10.0},
}

# ---------------------------------------------------------------------------
# Insertion specification
# ---------------------------------------------------------------------------
# Each entry: (parent_body, child_body, intermediate_name,
#              joint_axes, stiffness_class_name)
#
# joint_axes: list of axis suffixes to create (e.g., ['rz', 'ry', 'rx'])
#   These match the parent body's joint axes.

_LEG_INSERTIONS = [
    # Left leg
    ('lfemur',  'ltibia', 'lmidthigh', ['rz', 'ry', 'rx'], 'stiff_medium'),
    ('ltibia',  'lfoot',  'lmidshin',  ['rx'],              'stiff_low'),
    ('lfoot',   'ltoes',  'lmidfoot',  ['rz', 'rx'],        'stiff_medium'),
    # Right leg
    ('rfemur',  'rtibia', 'rmidthigh', ['rz', 'ry', 'rx'], 'stiff_medium'),
    ('rtibia',  'rfoot',  'rmidshin',  ['rx'],              'stiff_low'),
    ('rfoot',   'rtoes',  'rmidfoot',  ['rz', 'rx'],        'stiff_medium'),
]

_ARM_INSERTIONS = [
    # Left arm
    ('lclavicle', 'lhumerus', 'lmidshoulder', ['rz', 'ry'],        'stiff_high'),
    ('lhumerus',  'lradius',  'lmidhumerus',  ['rz', 'ry', 'rx'],  'stiff_low'),
    ('lradius',   'lwrist',   'lmidradius',   ['rx'],               'stiff_low'),
    # Right arm
    ('rclavicle', 'rhumerus', 'rmidshoulder', ['rz', 'ry'],        'stiff_high'),
    ('rhumerus',  'rradius',  'rmidhumerus',  ['rz', 'ry', 'rx'],  'stiff_low'),
    ('rradius',   'rwrist',   'rmidradius',   ['rx'],               'stiff_low'),
]

_ALL_INSERTIONS = _LEG_INSERTIONS + _ARM_INSERTIONS

# Axis name -> MuJoCo axis vector
_AXIS_MAP = {
    'rx': '1 0 0',
    'ry': '0 1 0',
    'rz': '0 0 1',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_pos(pos_str):
    """Parse a space-separated position string into a numpy array."""
    return np.array([float(x) for x in pos_str.split()])


def _fmt(v):
    return f'{v:.6g}'


def _fmt_pos(p):
    return f'{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}'


def _capsule_volume(radius, half_length):
    """Volume of a capsule (cylinder + 2 hemispheres)."""
    return math.pi * radius**2 * 2.0 * half_length + (4.0 / 3.0) * math.pi * radius**3


def _get_geom_mass_estimate(geom_elem, density=1000.0):
    """Estimate mass of a geom element from its geometry."""
    size_str = geom_elem.get('size', '')
    size_vals = [float(x) for x in size_str.split()]
    geom_type = geom_elem.get('type', 'capsule')

    if geom_type == 'capsule' and len(size_vals) >= 2:
        radius, half_length = size_vals[0], size_vals[1]
        return density * _capsule_volume(radius, half_length)
    elif geom_type == 'sphere' and len(size_vals) >= 1:
        radius = size_vals[0]
        return density * (4.0 / 3.0) * math.pi * radius**3
    elif geom_type == 'ellipsoid' and len(size_vals) >= 3:
        return density * (4.0 / 3.0) * math.pi * size_vals[0] * size_vals[1] * size_vals[2]
    return 0.1  # fallback


def _get_parent_geom_radius(parent_body):
    """Get the radius of the first capsule geom on the parent body."""
    for g in parent_body.findall('geom'):
        gtype = g.get('type', 'capsule')
        if gtype == 'capsule':
            size_vals = [float(x) for x in g.get('size', '0.04').split()]
            return size_vals[0]
    return 0.04  # fallback


# ---------------------------------------------------------------------------
# Core: insert intermediate body between parent and child
# ---------------------------------------------------------------------------

def _insert_intermediate(mjcf, parent_name, child_name, inter_name,
                         joint_axes, stiffness_class_name,
                         range_scale=_RANGE_SCALE, gear_scale=_GEAR_SCALE,
                         color_index=0):
    """Insert an intermediate body between parent and child in the XML tree.

    Parameters
    ----------
    mjcf : lxml Element
        Root MuJoCo XML element.
    parent_name, child_name : str
        Body names in the existing hierarchy.
    inter_name : str
        Name for the new intermediate body.
    joint_axes : list[str]
        Joint axis suffixes to create (e.g., ['rz', 'ry', 'rx']).
    stiffness_class_name : str
        Stiffness class for the new joints.
    range_scale : float
        Fraction of original range for each split joint.
    gear_scale : float
        Fraction of original gear for each split actuator.

    Returns
    -------
    new_joint_specs : list[(joint_name, axis_str, range_low, range_high,
                           stiffness, damping, gear)]
        Specs for the new intermediate joints (for actuator generation).
    modified_joint_specs : list[(joint_name, new_gear)]
        Original parent joints whose gear should be scaled down.
    """
    parent = mjcf.find(f'.//body[@name="{parent_name}"]')
    if parent is None:
        raise ValueError(f'Parent body "{parent_name}" not found')

    child = parent.find(f'./body[@name="{child_name}"]')
    if child is None:
        raise ValueError(
            f'Child body "{child_name}" not found under "{parent_name}"')

    # Parse child position relative to parent
    child_pos_str = child.get('pos', '0 0 0')
    child_pos = _parse_pos(child_pos_str)
    child_quat = child.get('quat')
    child_euler = child.get('euler')

    # Midpoint position
    mid_pos = child_pos / 2.0

    # Get parent's geom radius for the intermediate geom
    geom_radius = _get_parent_geom_radius(parent)

    # Estimate parent geom mass for mass conservation
    parent_geom = None
    for g in parent.findall('geom'):
        if g.get('name', '').startswith(parent_name.replace('hip', '').replace('joint', '')):
            parent_geom = g
            break
    if parent_geom is None:
        # Use first capsule geom on parent
        for g in parent.findall('geom'):
            if g.get('type', 'capsule') == 'capsule':
                parent_geom = g
                break

    parent_mass = _get_geom_mass_estimate(parent_geom) if parent_geom is not None else 0.2
    half_mass = parent_mass / 2.0

    # Shorten parent geom to cover only origin → midpoint, and set half mass.
    # Convert to `fromto` format so MuJoCo auto-computes orientation/length.
    if parent_geom is not None:
        parent_geom.set('mass', _fmt(half_mass))
        parent_geom.set('fromto', f'0 0 0 {_fmt_pos(mid_pos)}')
        parent_geom.set('size', _fmt(geom_radius))
        # Remove pos/quat/size that conflict with fromto
        for attr in ('pos', 'quat'):
            if attr in parent_geom.attrib:
                del parent_geom.attrib[attr]

    # Get stiffness/damping values for the class
    sclass = _STIFFNESS_CLASSES.get(stiffness_class_name, _STIFFNESS_CLASSES['main'])
    stiffness = sclass['stiffness']
    damping = sclass['damping']

    # Find parent's joints to get range info for matching axes
    parent_joints = {}
    for j in parent.findall('joint'):
        jname = j.get('name', '')
        for axis_suffix in ['rx', 'ry', 'rz']:
            if jname.endswith(axis_suffix):
                rng = j.get('range', '-30 30')
                lo, hi = [float(x) for x in rng.split()]
                parent_joints[axis_suffix] = (jname, lo, hi, j)
                break

    # Find original gear values for parent's joints from actuator section
    actuator_elem = mjcf.find('actuator')
    parent_gears = {}
    if actuator_elem is not None:
        for act in actuator_elem:
            act_joint = act.get('joint', '')
            if act_joint in [info[0] for info in parent_joints.values()]:
                gear_str = act.get('gear', '20')
                parent_gears[act_joint] = float(gear_str)

    # --- Create intermediate body ---
    inter_body = etree.Element('body')
    inter_body.set('name', inter_name)
    inter_body.set('pos', _fmt_pos(mid_pos))

    new_joint_specs = []
    modified_joint_specs = []

    for axis_suffix in joint_axes:
        axis_str = _AXIS_MAP[axis_suffix]
        joint_name = f'{inter_name}{axis_suffix}'

        if axis_suffix in parent_joints:
            # Scale the parent's range and use same scaled range for intermediate
            pj_name, orig_lo, orig_hi, pj_elem = parent_joints[axis_suffix]
            new_lo = orig_lo * range_scale
            new_hi = orig_hi * range_scale

            # Update parent joint range
            pj_elem.set('range', f'{_fmt(new_lo)} {_fmt(new_hi)}')

            # Get and scale gear
            orig_gear = parent_gears.get(pj_name, 20.0)
            new_gear = orig_gear * gear_scale

            # Update parent actuator gear
            modified_joint_specs.append((pj_name, new_gear))
        else:
            # No matching parent joint — use conservative default range
            new_lo, new_hi = -15.0, 15.0
            new_gear = 20.0 * gear_scale

        # Add joint to intermediate body
        j = etree.SubElement(inter_body, 'joint')
        j.set('name', joint_name)
        j.set('type', 'hinge')
        j.set('axis', axis_str)
        j.set('range', f'{_fmt(new_lo)} {_fmt(new_hi)}')
        j.set('limited', 'true')
        j.set('stiffness', _fmt(stiffness))
        j.set('damping', _fmt(damping))
        j.set('armature', '0.01')
        j.set('solimplimit', '0 0.99 0.01')

        new_joint_specs.append((joint_name, axis_str, new_lo, new_hi,
                                stiffness, damping, new_gear))

    # Add capsule geom to intermediate body using fromto format.
    # Geom spans from origin (0,0,0) to the remaining offset to child.
    remaining_offset = child_pos - mid_pos

    ig = etree.SubElement(inter_body, 'geom')
    ig.set('name', inter_name)
    ig.set('type', 'capsule')
    ig.set('fromto', f'0 0 0 {_fmt_pos(remaining_offset)}')
    ig.set('size', _fmt(geom_radius))
    ig.set('mass', _fmt(half_mass))
    # No self-collision for intermediate geoms
    ig.set('contype', '0')
    ig.set('conaffinity', '0')
    ig.set('rgba', '0.65 0.35 0.1 1')  # dark brown to highlight new joints

    # --- Reparent: move child from parent to intermediate ---
    parent.remove(child)

    # Update child's position to be relative to intermediate
    new_child_pos = child_pos - mid_pos
    child.set('pos', _fmt_pos(new_child_pos))

    # Append child to intermediate, then intermediate to parent
    inter_body.append(child)
    parent.append(inter_body)

    return new_joint_specs, modified_joint_specs


# ---------------------------------------------------------------------------
# Model generation
# ---------------------------------------------------------------------------

def make_model(range_scale=None, gear_scale=None):
    """Return MuJoCo XML string for the DOF-scaled humanoid CMU.

    Loads the original 56-DOF humanoid_CMU.xml and inserts 24 intermediate
    joints across arms and legs, producing an 80-DOF model.

    Parameters
    ----------
    range_scale : float or None
        Fraction of original range for each split joint.
        Default: _RANGE_SCALE (0.5).
    gear_scale : float or None
        Fraction of original gear for each split actuator.
        Default: _GEAR_SCALE (0.5).
    """
    if range_scale is None:
        range_scale = _RANGE_SCALE
    if gear_scale is None:
        gear_scale = _GEAR_SCALE

    xml_string = resources.GetResource(
        os.path.join(_TASKS_DIR, 'humanoid_CMU.xml'))
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    # Collect all new joint specs and modified parent gear values
    all_new_joints = []
    all_modified_gears = {}

    for i, (parent_name, child_name, inter_name,
            joint_axes, stiffness_class) in enumerate(_ALL_INSERTIONS):
        new_specs, modified_specs = _insert_intermediate(
            mjcf, parent_name, child_name, inter_name,
            joint_axes, stiffness_class,
            range_scale=range_scale, gear_scale=gear_scale,
            color_index=i)
        all_new_joints.extend(new_specs)
        for jname, new_gear in modified_specs:
            all_modified_gears[jname] = new_gear

    # --- Update existing actuator gears ---
    actuator_elem = mjcf.find('actuator')
    if actuator_elem is not None:
        for act in actuator_elem:
            act_name = act.get('name', '')
            if act_name in all_modified_gears:
                act.set('gear', _fmt(all_modified_gears[act_name]))

    # --- Add new actuators for intermediate joints ---
    if actuator_elem is None:
        actuator_elem = etree.SubElement(mjcf, 'actuator')

    for (jname, axis_str, rng_lo, rng_hi,
         stiffness, damping, gear) in all_new_joints:
        motor = etree.SubElement(actuator_elem, 'motor')
        motor.set('name', jname)
        motor.set('joint', jname)
        motor.set('gear', _fmt(gear))
        # Inherit ctrllimited and ctrlrange from defaults (class="main")

    # --- Add contact exclusions for new intermediate bodies ---
    contact_elem = mjcf.find('contact')
    if contact_elem is None:
        contact_elem = etree.SubElement(mjcf, 'contact')

    # Exclude collisions between adjacent intermediates and their neighbors
    for (parent_name, child_name, inter_name, _, _) in _ALL_INSERTIONS:
        exc = etree.SubElement(contact_elem, 'exclude')
        exc.set('body1', parent_name)
        exc.set('body2', inter_name)

    return etree.tostring(mjcf, pretty_print=True)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

class Physics(mujoco.Physics):
    """Physics simulation for the DOF-scaled humanoid_CMU."""

    def thorax_upright(self):
        """Returns projection from y-axes of thorax to the z-axes of world."""
        return self.named.data.xmat['thorax', 'zy']

    def head_height(self):
        """Returns the height of the head."""
        return self.named.data.xpos['head', 'z']

    def center_of_mass_position(self):
        """Returns position of the center-of-mass."""
        return self.named.data.subtree_com['thorax']

    def center_of_mass_velocity(self):
        """Returns the velocity of the center-of-mass."""
        return self.named.data.sensordata['thorax_subtreelinvel'].copy()

    def torso_vertical_orientation(self):
        """Returns the z-projection of the thorax orientation matrix."""
        return self.named.data.xmat['thorax', ['zx', 'zy', 'zz']]

    def joint_angles(self):
        """Returns the state without global orientation or position."""
        return self.data.qpos[7:].copy()

    def extremities(self):
        """Returns end effector positions in egocentric frame."""
        torso_frame = self.named.data.xmat['thorax'].reshape(3, 3)
        torso_pos = self.named.data.xpos['thorax']
        positions = []
        for side in ('l', 'r'):
            for limb in ('hand', 'foot'):
                torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
                positions.append(torso_to_limb.dot(torso_frame))
        return np.hstack(positions)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class HumanoidCMU(base.Task):
    """Task for the DOF-scaled CMU Humanoid."""

    def __init__(self, move_speed, random=None):
        self._move_speed = move_speed
        super().__init__(random=random)

    def initialize_episode(self, physics):
        # Start from the default straight standing pose (all joints at 0).
        # The XML places the root at z=1 with euler="90 0 0", so the default
        # qpos already represents a straight upright humanoid.
        physics.reset()
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['joint_angles'] = physics.joint_angles()
        obs['head_height'] = physics.head_height()
        obs['extremities'] = physics.extremities()
        obs['torso_vertical'] = physics.torso_vertical_orientation()
        obs['com_velocity'] = physics.center_of_mass_velocity()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        standing = rewards.tolerance(
            physics.head_height(),
            bounds=(_STAND_HEIGHT, float('inf')),
            margin=_STAND_HEIGHT / 4)
        upright = rewards.tolerance(
            physics.thorax_upright(),
            bounds=(0.9, float('inf')), sigmoid='linear',
            margin=1.9, value_at_margin=0)
        stand_reward = standing * upright
        small_control = rewards.tolerance(
            physics.control(), margin=1,
            value_at_margin=0, sigmoid='quadratic').mean()
        small_control = (4 + small_control) / 5

        if self._move_speed == 0:
            horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
            return small_control * stand_reward * dont_move
        else:
            com_velocity = np.linalg.norm(
                physics.center_of_mass_velocity()[[0, 1]])
            move = rewards.tolerance(
                com_velocity,
                bounds=(self._move_speed, float('inf')),
                margin=self._move_speed, value_at_margin=0,
                sigmoid='linear')
            move = (5 * move + 1) / 6
            return small_control * stand_reward * move


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _make_stand():
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = make_model()
        physics = Physics.from_xml_string(xml, common.ASSETS)
        task = HumanoidCMU(move_speed=0, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = '80_stand'
    return fn


def _make_walk():
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = make_model()
        physics = Physics.from_xml_string(xml, common.ASSETS)
        task = HumanoidCMU(move_speed=_WALK_SPEED, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = '80_walk'
    return fn


def _make_run():
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = make_model()
        physics = Physics.from_xml_string(xml, common.ASSETS)
        task = HumanoidCMU(move_speed=_RUN_SPEED, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = '80_run'
    return fn


# ---------------------------------------------------------------------------
# Original 56-DOF tasks (no intermediate joints)
# ---------------------------------------------------------------------------

def _make_56_stand():
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        physics = Physics.from_xml_string(
            resources.GetResource(os.path.join(_TASKS_DIR, 'humanoid_CMU.xml')),
            common.ASSETS)
        task = HumanoidCMU(move_speed=0, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = '56_stand'
    return fn


def _make_56_walk():
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        physics = Physics.from_xml_string(
            resources.GetResource(os.path.join(_TASKS_DIR, 'humanoid_CMU.xml')),
            common.ASSETS)
        task = HumanoidCMU(move_speed=_WALK_SPEED, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = '56_walk'
    return fn


def _make_56_run():
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        physics = Physics.from_xml_string(
            resources.GetResource(os.path.join(_TASKS_DIR, 'humanoid_CMU.xml')),
            common.ASSETS)
        task = HumanoidCMU(move_speed=_RUN_SPEED, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = '56_run'
    return fn


# Register tasks at import time
# 56-DOF: original humanoid_CMU (no modifications)
_dm_humanoid_CMU.SUITE.add('custom')(_make_56_stand())
_dm_humanoid_CMU.SUITE.add('custom')(_make_56_walk())
_dm_humanoid_CMU.SUITE.add('custom')(_make_56_run())
# 80-DOF: with 24 intermediate joints inserted
_dm_humanoid_CMU.SUITE.add('custom')(_make_stand())
_dm_humanoid_CMU.SUITE.add('custom')(_make_walk())
_dm_humanoid_CMU.SUITE.add('custom')(_make_run())
