"""Walker DOF-Scaling Variant Environments.

Registers walker-{size}_{stand,walk,run} tasks on dm_control's walker SUITE.
Sizes: 4, 8, 16, 32, 64, 128, 256 DOF (always 2 identical legs).

The walker is a planar biped with 2 identical legs, each with 3 joints
baseline (hip, knee, ankle) = 6 actuated DOF.  We scale by subdividing
each leg into more joints along the same physical path.

Design follows cheetah.py (profile interpolation for per-joint properties)
and quadruped_final.py (geom radius interpolation):
  - Gear ratios: 3-point profile 100→50→20, no DOF scaling
  - Joint ranges: 3-point asymmetric profile, (3/n)^1.0 scaling
  - Geom radii: 3-point profile 0.05→0.04→0.05 (original walker values)
  - Stiffness: 3-point profile 200→100→40 (2x gear, matching cheetah ratio)
  - Damping: 3-point profile 5.0→2.5→1.0 (0.05x gear, matching cheetah ratio)
  - Armature: 0.1 constant (matching cheetah)

Collision: walker's default conaffinity=0 on body geoms already prevents
all inter-body contacts.  No contype bit separation needed (unlike cheetah).

Body geometry (torso, cameras, floor) is DOF-invariant — only legs change.
"""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import walker as _dm_walker
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.utils import rewards
from lxml import etree
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Horizontal speeds (meters/second) — same as dm_control walker.
_WALK_SPEED = 1
_RUN_SPEED = 8

# Supported DOF counts (must be even, >= 4).
_SIZES = [4, 8, 16, 32, 64, 128, 256]
_BASELINE_PER_LEG = 3  # original walker has 3 joints per leg

# Joint range scaling exponent (1.0 preserves total curvature budget).
_RANGE_ALPHA = 1.0

# ---------------------------------------------------------------------------
# Leg path waypoints (hip-local frame, extracted from walker.xml)
# Both legs are identical — only y-offset differs at attachment.
# ---------------------------------------------------------------------------

_LEG_WAYPOINTS = np.array([
    [0.0,   0.0,  0.0],     # hip joint origin
    [0.0,   0.0, -0.7],     # knee joint (straight down)
    [0.06,  0.0, -0.95],    # ankle joint (slight forward offset)
    [0.16,  0.0, -0.95],    # toe tip (foot extends horizontally)
])

# ---------------------------------------------------------------------------
# Joint range profiles at t = [0.0, 0.5, 1.0]  (degrees)
# From walker.xml: hip (-20,100), knee (-150,0), ankle (-45,45)
# ---------------------------------------------------------------------------

_RANGE_LOW = [-20, -150, -45]
_RANGE_HIGH = [100, 0, 45]

# ---------------------------------------------------------------------------
# Gear ratio profile at t = [0.0, 0.5, 1.0]
# From walker.xml: hip=100, knee=50, ankle=20
# ---------------------------------------------------------------------------

_GEAR = [100, 50, 20]

# ---------------------------------------------------------------------------
# Stiffness & damping profiles at t = [0.0, 0.5, 1.0]
# Derived from cheetah ratios: stiffness = 2.0 × gear, damping = 0.05 × gear.
# The original walker.xml has damping=0.1 and no stiffness, which leaves
# high-DOF legs wobbly — the cheetah's ratio-matched profiles give composed
# passive dynamics at any DOF count.
# ---------------------------------------------------------------------------

_STIFFNESS = [200, 100, 40]
_DAMPING = [5.0, 2.5, 1.0]

# ---------------------------------------------------------------------------
# Geom radius profile at t = [0.0, 0.5, 1.0]
# From walker.xml: thigh=0.05, shin=0.04, foot=0.05
# ---------------------------------------------------------------------------

_RADII = [0.05, 0.04, 0.05]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_size(size):
    """Validate DOF count and return joints-per-leg."""
    if not isinstance(size, int):
        raise TypeError(f'size must be int, got {type(size).__name__}')
    if size <= 0:
        raise ValueError(f'size must be positive, got {size}')
    if size % 2 != 0:
        raise ValueError(f'size must be even (2 legs), got {size}')
    if size < 4:
        raise ValueError(f'size must be >= 4, got {size}')
    if size not in _SIZES:
        raise ValueError(f'Unsupported size {size}. Supported: {_SIZES}')
    return size // 2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _interpolate_path(waypoints, n_joints):
    """Place n_joints body nodes at equal arc-length along the leg path.

    Parameters
    ----------
    waypoints : (K, 3) array
        Waypoints defining the leg path.
    n_joints : int
        Number of joints (bodies) to place along the path.

    Returns
    -------
    positions : (n_joints, 3)
        Absolute positions in hip-local frame.
    relative_positions : (n_joints, 3)
        Each row is the position of body i relative to body i-1
        (row 0 is [0, 0, 0]).
    toe_pos : (3,)
        Position of the toe tip (end of path).
    """
    segments = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    total_length = seg_lengths.sum()
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # n_joints bodies → first at start, last near the end
    arc_positions = np.linspace(0.0, total_length, n_joints + 1)[:-1]

    positions = np.zeros((n_joints, 3))
    for i, s in enumerate(arc_positions):
        seg_idx = int(np.searchsorted(cumulative[1:], s, side='right'))
        seg_idx = min(seg_idx, len(segments) - 1)
        local_s = s - cumulative[seg_idx]
        t = local_s / seg_lengths[seg_idx] if seg_lengths[seg_idx] > 0 else 0.0
        t = np.clip(t, 0.0, 1.0)
        positions[i] = waypoints[seg_idx] + t * segments[seg_idx]

    toe_pos = waypoints[-1]

    relative_positions = np.zeros_like(positions)
    relative_positions[0] = positions[0]  # always [0,0,0]
    for i in range(1, n_joints):
        relative_positions[i] = positions[i] - positions[i - 1]

    return positions, relative_positions, toe_pos


# ---------------------------------------------------------------------------
# Leg builder
# ---------------------------------------------------------------------------

def _build_leg(prefix, n_joints, waypoints):
    """Build an lxml Element for one leg body chain.

    Parameters
    ----------
    prefix : str
        'r' for right leg, 'l' for left leg.
    n_joints : int
        Number of joints in this leg.
    waypoints : (K, 3) array
        Leg path waypoints.

    Returns
    -------
    hip_body : lxml Element
    joint_specs : list[(name, range_low, range_high, gear)]
    """
    positions, relative_positions, toe_pos = _interpolate_path(
        waypoints, n_joints)

    t_profile = [0.0, 0.5, 1.0]

    # Joint ranges: scale down to preserve total curvature budget.
    range_scale = (_BASELINE_PER_LEG / n_joints) ** _RANGE_ALPHA

    # Gear: no DOF scaling — more joints = more total torque.
    gear_scale = 1.0

    # Armature: constant (matching cheetah).
    armature = 0.1

    def _fmt(v):
        return f'{v:.6g}'

    def _fmt_pos(p):
        return f'{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}'

    joint_specs = []

    # Build the nested body chain
    parent = None
    for i in range(n_joints):
        t = i / max(n_joints - 1, 1)

        # Compute joint properties via profile interpolation
        rng_low = np.interp(t, t_profile, _RANGE_LOW) * range_scale
        rng_high = np.interp(t, t_profile, _RANGE_HIGH) * range_scale
        # Ensure minimum range of 0.5 degrees
        if rng_high - rng_low < 0.5:
            mid = (rng_high + rng_low) / 2
            rng_low = mid - 0.25
            rng_high = mid + 0.25
        gear = np.interp(t, t_profile, _GEAR) * gear_scale
        stiffness = float(np.interp(t, t_profile, _STIFFNESS))
        damping = float(np.interp(t, t_profile, _DAMPING))
        radius = float(np.interp(t, t_profile, _RADII))

        body_name = f'{prefix}seg_{i}'
        joint_name = f'{prefix}joint_{i}'
        geom_name = f'{prefix}geom_{i}'

        body = etree.Element('body')
        body.set('name', body_name)
        if i == 0:
            # First body is at hip position (pos set by caller)
            pass
        else:
            body.set('pos', _fmt_pos(relative_positions[i]))

        # Joint — all use the walker's default axis (0 -1 0)
        j = etree.SubElement(body, 'joint')
        j.set('name', joint_name)
        j.set('range', f'{_fmt(rng_low)} {_fmt(rng_high)}')
        j.set('stiffness', _fmt(stiffness))
        j.set('damping', _fmt(damping))
        j.set('armature', _fmt(armature))

        # Geom: capsule from origin to next body offset (or to toe tip)
        if i < n_joints - 1:
            fromto_end = relative_positions[i + 1]
        else:
            # Last segment points to the toe tip
            fromto_end = toe_pos - positions[i]

        g = etree.SubElement(body, 'geom')
        g.set('name', geom_name)
        g.set('type', 'capsule')
        g.set('fromto', f'0 0 0 {_fmt_pos(fromto_end)}')
        g.set('size', _fmt(radius))
        # Alternate light/dark orange so individual segments are visible.
        if i % 2 == 0:
            g.set('rgba', '0.65 0.35 0.1 1')  # dark brown
        else:
            g.set('rgba', '0.9 0.6 0.2 1')    # light orange

        # Last body gets a foot site
        if i == n_joints - 1:
            site = etree.SubElement(body, 'site')
            site.set('name', f'{prefix}foot')
            site.set('pos', _fmt_pos(fromto_end))
            site.set('size', '0.01')

        joint_specs.append((joint_name, rng_low, rng_high, gear))

        if parent is None:
            hip_body = body
        else:
            parent.append(body)
        parent = body

    return hip_body, joint_specs


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def make_model(size):
    """Return MuJoCo XML string for the given walker DOF variant.

    The body geometry (torso, cameras, floor) is identical to the original
    6-DOF walker.  Only the legs are rebuilt with more joints.
    """
    n_per_leg = _validate_size(size)

    # Parse base XML
    xml_string = resources.GetResource(
        os.path.join(_TASKS_DIR, 'walker.xml'))
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    torso = mjcf.find('.//body[@name="torso"]')

    # Add dark body material to match leg stripe color
    asset = mjcf.find('asset')
    if asset is None:
        asset = etree.SubElement(mjcf, 'asset')
    body_mat = etree.SubElement(asset, 'material')
    body_mat.set('name', 'body_dark')
    body_mat.set('rgba', '0.9 0.6 0.2 1')
    torso_geom = torso.find('geom[@name="torso"]')
    torso_geom.set('material', 'body_dark')

    # 1. Remove existing leg bodies (right_thigh, left_thigh)
    for name in ('right_thigh', 'left_thigh'):
        b = torso.find(f'./body[@name="{name}"]')
        if b is not None:
            torso.remove(b)

    # 2. Remove existing actuator section
    old_act = mjcf.find('actuator')
    if old_act is not None:
        mjcf.remove(old_act)

    # 3. Increase arena memory for large models.
    memory_mb = max(20, size)
    size_elem = mjcf.find('size')
    if size_elem is None:
        size_elem = etree.SubElement(mjcf, 'size')
    size_elem.set('memory', f'{memory_mb}M')

    # 4. Build legs (both use the same waypoints, differ only in y-offset)
    all_joint_specs = []

    # Right leg (y = -0.05)
    right_hip, right_specs = _build_leg(
        prefix='r', n_joints=n_per_leg, waypoints=_LEG_WAYPOINTS)
    right_hip.set('pos', '0 -0.05 -0.3')
    torso.append(right_hip)
    all_joint_specs.extend(right_specs)

    # Left leg (y = 0.05)
    left_hip, left_specs = _build_leg(
        prefix='l', n_joints=n_per_leg, waypoints=_LEG_WAYPOINTS)
    left_hip.set('pos', '0 0.05 -0.3')
    torso.append(left_hip)
    all_joint_specs.extend(left_specs)

    # 5. Rebuild actuator section with motor actuators
    actuator_elem = etree.SubElement(mjcf, 'actuator')
    for jname, _, _, gear in all_joint_specs:
        motor = etree.SubElement(actuator_elem, 'motor')
        motor.set('name', jname)
        motor.set('joint', jname)
        motor.set('gear', f'{gear:.6g}')

    return etree.tostring(mjcf, pretty_print=True)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

_STAND_HEIGHT = 1.2

class Physics(_dm_walker.Physics):
    """Physics with foot-contact detection for high-DOF walker."""

    def _reload_from_data(self, data):
        super()._reload_from_data(data)
        self._floor_geom_id = None
        self._foot_geom_ids = None
        self._seg_geom_ids = None

    def only_toes_on_ground(self):
        """Return True if foot geoms touch the floor and no leg segments do.

        Foot geoms are identified as those on the foot body (containing
        rfoot / lfoot site) and its parent (ankle).  All other leg geoms
        (rgeom_* / lgeom_*) are considered segments.  Torso geom is ignored.
        """
        if self._floor_geom_id is None:
            self._floor_geom_id = self.model.name2id('floor', 'geom')
            self._foot_geom_ids = set()
            self._seg_geom_ids = set()

            # Find body IDs for the last 4 segments per leg
            # (foot site body + 3 parents: geom_31, 30, 29, 28)
            foot_body_ids = set()
            for i in range(self.model.nsite):
                name = self.model.id2name(i, 'site')
                if name in ('rfoot', 'lfoot'):
                    bid = self.model.site_bodyid[i]
                    for _ in range(4):
                        foot_body_ids.add(bid)
                        bid = self.model.body_parentid[bid]

            # Categorize leg geoms
            for i in range(self.model.ngeom):
                name = self.model.id2name(i, 'geom')
                if name.startswith(('rgeom_', 'lgeom_')):
                    body_id = self.model.geom_bodyid[i]
                    if body_id in foot_body_ids:
                        self._foot_geom_ids.add(i)
                    else:
                        self._seg_geom_ids.add(i)

        floor_id = self._floor_geom_id
        foot_ids = self._foot_geom_ids
        seg_ids = self._seg_geom_ids
        has_foot_contact = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == floor_id:
                if g2 in seg_ids:
                    return False
                if g2 in foot_ids:
                    has_foot_contact = True
            elif g2 == floor_id:
                if g1 in seg_ids:
                    return False
                if g1 in foot_ids:
                    has_foot_contact = True
        return has_foot_contact


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class PlanarWalker(_dm_walker.PlanarWalker):
    """PlanarWalker with near-straight init and toe contact reward.

    Initialization: joints start around zero (straight legs) with small
    noise, clamped to joint limits.

    Reward: adds a toe_contact_reward multiplier (1.0 if only feet touch
    the floor, 0.6 otherwise) matching the quadruped's Move reward.
    """

    def initialize_episode(self, physics):
        is_limited = physics.model.jnt_limited.astype(bool)
        lower, upper = physics.model.jnt_range[is_limited].T
        spread = (upper - lower) * 0.05
        noise = self.random.uniform(-spread, spread)
        physics.data.qpos[is_limited] = np.clip(noise, lower, upper)
        super(_dm_walker.PlanarWalker, self).initialize_episode(physics)

    def get_reward(self, physics):
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        if self._move_speed == 0:
            reward = stand_reward
        else:
            move_reward = rewards.tolerance(
                physics.horizontal_velocity(),
                bounds=(self._move_speed, float('inf')),
                margin=self._move_speed / 2,
                value_at_margin=0.5,
                sigmoid='linear')
            reward = stand_reward * (5 * move_reward + 1) / 6
        toe_contact_reward = 1.0 if physics.only_toes_on_ground() else 0.6
        return reward 


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _make_stand(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        model_string = make_model(size)
        physics = Physics.from_xml_string(model_string, common.ASSETS)
        task = PlanarWalker(move_speed=0, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = f'{size}_stand'
    return fn


def _make_walk(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        model_string = make_model(size)
        physics = Physics.from_xml_string(model_string, common.ASSETS)
        task = PlanarWalker(move_speed=_WALK_SPEED, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = f'{size}_walk'
    return fn


def _make_run(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        model_string = make_model(size)
        physics = Physics.from_xml_string(model_string, common.ASSETS)
        task = PlanarWalker(move_speed=_RUN_SPEED, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = f'{size}_run'
    return fn


# Register all task combinations (7 sizes × 3 tasks) at import time.
for _size in _SIZES:
    _dm_walker.SUITE.add('custom')(_make_stand(_size))
    _dm_walker.SUITE.add('custom')(_make_walk(_size))
    _dm_walker.SUITE.add('custom')(_make_run(_size))
