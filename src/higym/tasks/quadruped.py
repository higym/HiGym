"""Quadruped DOF-Scaling Variant Environments (Final).

Registers quadruped-{size}_{walk,run,escape,fetch} tasks on dm_control's
quadruped SUITE.  Legs are procedurally generated with direct joint actuation
for all sizes (16–512 DOF, always 4 legs).
"""

import math
import os

from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import quadruped as _dm_quadruped
from dm_control.utils import rewards
from dm_control.utils import xml_tools
from dm_control.utils import io as resources
from lxml import etree
import numpy as np

enums = mjbindings.enums
mjlib = mjbindings.mjlib

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02
_RUN_SPEED = 5
_WALK_SPEED = 0.5
_WALLS = ['wall_px', 'wall_py', 'wall_nx', 'wall_ny']

# Supported DOF counts (must be divisible by 4).
_SIZES = [16, 32, 64, 128, 256, 512]

# Original leg waypoints in hip-local frame (absolute coordinates).
_WAYPOINTS = np.array([
    [0.0,  0.0,  0.0],    # hip origin
    [0.3,  0.0,  0.11],   # knee
    [0.55, 0.0, -0.14],   # ankle
    [0.55, 0.0, -0.44],   # toe
])

# Original dm_control radii at each waypoint (hip, knee, ankle).
_ORIGINAL_RADII = np.array([0.08, 0.065, 0.055])
_ORIGINAL_RADII_FRACS = None  # computed lazily

# Joint range scaling: alpha=1.0 preserves total curvature budget.
_PITCH_RANGE_ALPHA = 1.0

# Original 4-leg placements: (name, (x, y, z), euler_z_degrees).
_LEG_PLACEMENTS = [
    ('front_left',  (0.2,  0.2, 0.0),   45),
    ('front_right', (0.2, -0.2, 0.0),  -45),
    ('back_right',  (-0.2, -0.2, 0.0), -135),
    ('back_left',   (-0.2,  0.2, 0.0),  135),
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_size(size):
    """Validate DOF count and return joints-per-leg."""
    if not isinstance(size, int):
        raise TypeError(f'size must be int, got {type(size).__name__}')
    if size <= 0:
        raise ValueError(f'size must be positive, got {size}')
    if size % 4 != 0:
        raise ValueError(f'size must be divisible by 4 (4 legs), got {size}')
    n_joints = size // 4
    if n_joints < 2:
        raise ValueError(
            f'size={size} gives n_joints={n_joints}; need at least 8 DOF')
    if size not in _SIZES:
        raise ValueError(f'Unsupported size {size}. Supported: {_SIZES}')
    return n_joints


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _get_radius(t):
    """Interpolate original dm_control radius at normalised arc position t in [0, 1].

    Preserves the exact hip (0.08) -> knee (0.065) -> ankle (0.055) profile
    from the original dm_control quadruped.  No DOF-dependent shrinkage is
    applied because contype/conaffinity filtering already prevents self-
    collision between adjacent capsules.
    """
    global _ORIGINAL_RADII_FRACS
    if _ORIGINAL_RADII_FRACS is None:
        segs = np.diff(_WAYPOINTS, axis=0)
        lens = np.linalg.norm(segs, axis=1)
        total = lens.sum()
        _ORIGINAL_RADII_FRACS = np.array([
            0.0,
            lens[0] / total,
            (lens[0] + lens[1]) / total,
        ])
    return float(np.interp(t, _ORIGINAL_RADII_FRACS, _ORIGINAL_RADII))


def _compute_original_leg_mass():
    """Compute total mass of one original leg (thigh + shin + foot + toe)."""
    density = 500.0

    def capsule_vol(radius, start, end):
        length = np.linalg.norm(np.asarray(end) - np.asarray(start))
        return math.pi * radius**2 * length + (4.0 / 3.0) * math.pi * radius**3

    thigh = density * capsule_vol(0.08,  [0, 0, 0], [0.3, 0, 0.11])
    shin  = density * capsule_vol(0.065, [0, 0, 0], [0.25, 0, -0.25])
    foot  = density * capsule_vol(0.055, [0, 0, 0], [0, 0, -0.25])
    toe   = density * (4.0 / 3.0) * math.pi * 0.08**3
    return thigh + shin + foot + toe

_ORIGINAL_LEG_MASS = _compute_original_leg_mass()


def _interpolate_path(n_joints):
    """Place *n_joints* body nodes at equal arc-length along the leg path.

    Returns
    -------
    positions : (n_joints, 3)
        Absolute positions in hip-local frame.
    relative_positions : (n_joints, 3)
        Each row is the position of body *i* relative to body *i-1*
        (row 0 is the hip offset, always [0, 0, 0]).
    """
    segments = np.diff(_WAYPOINTS, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    total_length = seg_lengths.sum()
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    arc_positions = np.linspace(0.0, total_length, n_joints)

    positions = np.zeros((n_joints, 3))
    for i, s in enumerate(arc_positions):
        seg_idx = int(np.searchsorted(cumulative[1:], s, side='right'))
        seg_idx = min(seg_idx, len(segments) - 1)
        local_s = s - cumulative[seg_idx]
        t = local_s / seg_lengths[seg_idx] if seg_lengths[seg_idx] > 0 else 0.0
        t = np.clip(t, 0.0, 1.0)
        positions[i] = _WAYPOINTS[seg_idx] + t * segments[seg_idx]

    relative_positions = np.zeros_like(positions)
    relative_positions[0] = positions[0]
    for i in range(1, n_joints):
        relative_positions[i] = positions[i] - positions[i - 1]

    return positions, relative_positions


# ---------------------------------------------------------------------------
# Leg builder
# ---------------------------------------------------------------------------

def _build_leg(leg_name, pos, euler_z, n_joints,
               mass_per_geom, relative_positions,
               damping=30.0, armature=0.01,
               pitch_range_alpha=_PITCH_RANGE_ALPHA):
    """Build an lxml Element for one leg body chain.

    Geom radii are interpolated from the original dm_control quadruped
    (0.08 at hip -> 0.065 at knee -> 0.055 at ankle) via _get_radius().
    Yaw range is fixed; pitch ranges scale with configurable alpha.

    Returns
    -------
    hip_element : lxml Element
    joint_specs : list[(str, (float, float))]
        Ordered (joint_name, (low_deg, high_deg)) for actuator creation.
    """
    n_pitch = n_joints - 1

    # Yaw range fixed across sizes (one yaw joint per leg).
    yaw_scale = 1.0

    # Pitch scaling: preserve total curvature budget.
    n_pitch_base = 3
    if n_pitch <= 0:
        raise ValueError(f'n_joints must be >= 2, got {n_joints}')
    pitch_scale = (n_pitch_base / n_pitch) ** pitch_range_alpha

    def _pitch_range(pitch_idx):
        """Interpolate joint range from hip-pitch to ankle along the chain."""
        if n_pitch <= 1:
            t = 0.0
        else:
            t = pitch_idx / (n_pitch - 1)
        low_unscaled  = -20.0 + t * (-45.0 - (-20.0))
        high_unscaled =  60.0 + t * ( 55.0 -  60.0 )
        return low_unscaled * pitch_scale, high_unscaled * pitch_scale

    def _fmt(v):
        return f'{v:.6g}'

    def _fmt_pos(p):
        return f'{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}'

    def _add_joint(parent_elem, name, axis, rng):
        j = etree.SubElement(parent_elem, 'joint')
        j.set('name', name)
        j.set('type', 'hinge')
        j.set('axis', axis)
        j.set('range', f'{_fmt(rng[0])} {_fmt(rng[1])}')
        j.set('damping', _fmt(damping))
        j.set('armature', _fmt(armature))
        j.set('limited', 'true')
        j.set('solimplimit', '0 .99 .01')
        joint_specs.append((name, rng))
        return j

    joint_specs = []

    # --- Hip body (child of torso) ---
    hip = etree.Element('body')
    hip.set('name', f'hip_{leg_name}')
    hip.set('pos', f'{pos[0]:.6g} {pos[1]:.6g} {pos[2]:.6g}')
    hip.set('euler', f'0 0 {euler_z}')

    yaw_name = f'yaw_{leg_name}'
    yaw_range = (-50.0 * yaw_scale, 50.0 * yaw_scale)
    _add_joint(hip, yaw_name, '0 0 1', yaw_range)

    p0_name = f'pitch_0_{leg_name}'
    _add_joint(hip, p0_name, '0 1 0', _pitch_range(0))

    hip_radius = _get_radius(0.0)

    # Extend capsule backward into torso so hemispherical cap overlaps.
    seg1 = relative_positions[1]
    seg1_len = np.linalg.norm(seg1)
    if seg1_len > 0:
        overlap = -(seg1 / seg1_len) * min(hip_radius * 2.5, 0.15)
    else:
        overlap = np.zeros(3)

    g = etree.SubElement(hip, 'geom')
    g.set('name', f'seg_0_{leg_name}')
    g.set('type', 'capsule')
    g.set('fromto', f'{_fmt_pos(overlap)} {_fmt_pos(relative_positions[1])}')
    g.set('size', _fmt(hip_radius))
    g.set('mass', _fmt(mass_per_geom))
    g.set('contype', '0')
    g.set('conaffinity', '0')
    g.set('rgba', '0.65 0.35 0.1 1')

    # --- Intermediate bodies (1 .. n_joints-2) and toe (n_joints-1) ---
    parent = hip
    for i in range(1, n_joints):
        body = etree.SubElement(parent, 'body')
        body.set('pos', _fmt_pos(relative_positions[i]))

        if i < n_joints - 1:
            # Intermediate segment
            body.set('name', f'seg_{i}_{leg_name}')

            pname = f'pitch_{i}_{leg_name}'
            _add_joint(body, pname, '0 1 0', _pitch_range(i))

            t_along = i / max(n_joints - 2, 1)
            seg_radius = _get_radius(t_along)

            sg = etree.SubElement(body, 'geom')
            sg.set('name', f'seg_{i}_{leg_name}')
            sg.set('type', 'capsule')
            sg.set('fromto', f'0 0 0 {_fmt_pos(relative_positions[i + 1])}')
            sg.set('size', _fmt(seg_radius))
            sg.set('mass', _fmt(mass_per_geom))
            sg.set('contype', '2')
            sg.set('conaffinity', '1')
            if i % 2 == 0:
                sg.set('rgba', '0.65 0.35 0.1 1')
            else:
                sg.set('rgba', '0.9 0.6 0.2 1')
        else:
            # Toe (terminal body) -- no joint
            body.set('name', f'toe_{leg_name}')

            toe_r = 0.08
            tg = etree.SubElement(body, 'geom')
            tg.set('name', f'toe_{leg_name}')
            tg.set('type', 'sphere')
            tg.set('size', _fmt(toe_r))
            tg.set('mass', _fmt(mass_per_geom))
            tg.set('material', 'effector')
            tg.set('condim', '3')
            tg.set('friction', '1.5')

            ts = etree.SubElement(body, 'site')
            ts.set('name', f'toe_{leg_name}')
            ts.set('type', 'sphere')
            ts.set('size', _fmt(toe_r * 1.05))
            ts.set('material', 'site')
            ts.set('group', '4')

        parent = body

    return hip, joint_specs


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def make_model(size=16, floor_size=None, terrain=False,
               rangefinders=False, walls_and_ball=False):
    """Return MuJoCo XML string for the given quadruped size variant."""
    n_joints = _validate_size(size)

    xml_string = resources.GetResource(
        os.path.join(_TASKS_DIR, 'quadruped.xml'))
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

    # 1. Remove existing leg bodies
    for b in list(torso):
        if b.tag == 'body' and (b.get('name') or '').startswith('hip_'):
            torso.remove(b)

    # 2. Remove tendon / equality / actuator sections
    for tag in ('tendon', 'equality', 'actuator'):
        elem = mjcf.find(tag)
        if elem is not None:
            mjcf.remove(elem)

    # 3. Remove old toe force/torque sensors
    sensor_elem = mjcf.find('sensor')
    if sensor_elem is None:
        sensor_elem = etree.SubElement(mjcf, 'sensor')
    else:
        for s in list(sensor_elem):
            name = s.get('name', '')
            if name.startswith('force_toe_') or name.startswith('torque_toe_'):
                sensor_elem.remove(s)

    # 4. Generate new legs
    _positions, relative_positions = _interpolate_path(n_joints)
    mass_per_geom = _ORIGINAL_LEG_MASS / n_joints

    # Scale damping and armature up for stability with many DOFs
    damping = 30.0 * (n_joints / 4.0) ** 0.5
    armature = 0.01 * (n_joints / 4.0) ** 0.3

    all_joint_specs = []
    toe_names = []

    for leg_name, pos, euler_z in _LEG_PLACEMENTS:
        leg_elem, joint_specs = _build_leg(
            leg_name, pos, euler_z, n_joints,
            mass_per_geom, relative_positions,
            damping=damping, armature=armature)
        torso.append(leg_elem)
        all_joint_specs.extend(joint_specs)
        toe_names.append(f'toe_{leg_name}')

    # 5. Generate actuators with matched ctrlrange and gain compensation.
    #
    # These are position servos: torque = gain * (act - qpos).
    # With ctrlrange matched to joint range, max |act - qpos| = joint_range_rad.
    # To preserve the same max torque as the original [-1,1] ctrlrange:
    #   gain_new * joint_range_rad = gain_base * baseline_range
    #   gain_new = gain_base * baseline_range / joint_range_rad
    base_gain = 1000.0 * (4.0 / n_joints) ** 0.4
    baseline_ctrl_range = 2.0  # original [-1, 1] span

    actuator_elem = etree.SubElement(mjcf, 'actuator')
    for jname, (low_deg, high_deg) in all_joint_specs:
        low_rad = math.radians(low_deg)
        high_rad = math.radians(high_deg)
        joint_range_rad = high_rad - low_rad

        gain = base_gain * (baseline_ctrl_range / joint_range_rad)

        act = etree.SubElement(actuator_elem, 'general')
        act.set('name', jname)
        act.set('joint', jname)
        act.set('ctrllimited', 'true')
        act.set('ctrlrange', f'{low_rad:.6g} {high_rad:.6g}')
        act.set('gainprm', f'{gain:.6g}')
        act.set('biasprm', f'0 {-gain:.6g}')
        act.set('biastype', 'affine')
        act.set('dyntype', 'filter')
        act.set('dynprm', '0.1')

    # 6. Generate force/torque sensors for new toes
    for toe_name in toe_names:
        f_s = etree.SubElement(sensor_elem, 'force')
        f_s.set('name', f'force_{toe_name}')
        f_s.set('site', toe_name)
        t_s = etree.SubElement(sensor_elem, 'torque')
        t_s.set('name', f'torque_{toe_name}')
        t_s.set('site', toe_name)

    # ---- Standard modifications (all sizes) ----
    if floor_size is not None:
        floor_geom = mjcf.find('.//geom[@name="floor"]')
        floor_geom.attrib['size'] = f'{floor_size} {floor_size} .5'

    if not walls_and_ball:
        for wall in _WALLS:
            wg = xml_tools.find_element(mjcf, 'geom', wall)
            wg.getparent().remove(wg)
        ball_body = xml_tools.find_element(mjcf, 'body', 'ball')
        ball_body.getparent().remove(ball_body)
        target_site = xml_tools.find_element(mjcf, 'site', 'target')
        target_site.getparent().remove(target_site)

    if not terrain:
        terrain_geom = xml_tools.find_element(mjcf, 'geom', 'terrain')
        terrain_geom.getparent().remove(terrain_geom)

    if not rangefinders:
        for rf in mjcf.findall('.//rangefinder'):
            rf.getparent().remove(rf)

    return etree.tostring(mjcf, pretty_print=True)


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

class Physics(_dm_quadruped.Physics):
    """Physics with dynamic toe-site discovery (works for any leg count)."""

    def _reload_from_data(self, data):
        super()._reload_from_data(data)
        self._toe_names = []
        self._floor_geom_id = None
        self._leg_seg_geom_ids = None

    def only_toes_on_ground(self):
        """Return True if toes touch the floor and no leg segments do.

        Requires at least one toe geom contacting the floor AND zero leg-segment
        (seg_*) geoms contacting the floor.  Torso and other non-leg geoms are
        ignored (torso collapse is handled separately by torso_height_reward).
        """
        if self._floor_geom_id is None:
            self._floor_geom_id = self.model.name2id('floor', 'geom')
            self._leg_seg_geom_ids = set()
            self._toe_geom_ids = set()
            for i in range(self.model.ngeom):
                name = self.model.id2name(i, 'geom')
                if name.startswith('seg_'):
                    self._leg_seg_geom_ids.add(i)
                elif name.startswith('toe_'):
                    self._toe_geom_ids.add(i)

        floor_id = self._floor_geom_id
        seg_ids = self._leg_seg_geom_ids
        toe_ids = self._toe_geom_ids
        has_toe_contact = False

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 == floor_id:
                if g2 in seg_ids:
                    return False
                if g2 in toe_ids:
                    has_toe_contact = True
            elif g2 == floor_id:
                if g1 in seg_ids:
                    return False
                if g1 in toe_ids:
                    has_toe_contact = True
        return has_toe_contact

    def toe_positions(self):
        """Return toe positions in egocentric frame."""
        if not self._toe_names:
            for i in range(self.model.nbody):
                name = self.model.id2name(i, 'body')
                if name and name.startswith('toe_'):
                    self._toe_names.append(name)
        if not self._toe_names:
            return np.zeros((0, 3), dtype=np.float64)
        torso_frame = self.named.data.xmat['torso'].reshape(3, 3)
        torso_pos = self.named.data.xpos['torso']
        torso_to_toe = self.named.data.xpos[self._toe_names] - torso_pos
        return torso_to_toe.dot(torso_frame)


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------

def _upright_orientation(random_state):
    """Random yaw with slight random tilt (~5 deg) for robustness."""
    azimuth = random_state.uniform(0, 2 * np.pi)
    q = np.array([np.cos(azimuth / 2), 0, 0, np.sin(azimuth / 2)])
    q[1] += random_state.uniform(-0.05, 0.05)
    q[2] += random_state.uniform(-0.05, 0.05)
    return q / np.linalg.norm(q)


class Move(_dm_quadruped.Move):
    def initialize_episode(self, physics):
        orientation = _upright_orientation(self.random)
        _dm_quadruped._find_non_contacting_height(physics, orientation)
        base.Task.initialize_episode(self, physics)

    def get_reward(self, physics):
        move_reward = rewards.tolerance(
            physics.torso_velocity()[0],
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.5,
            sigmoid='linear')
        upright = _dm_quadruped._upright_reward(physics)

        torso_height_reward = rewards.tolerance(
            float(physics.named.data.xpos['torso', 'z']),
            bounds=(0.47, 0.55),
            margin=0.2,
            sigmoid='linear',
            value_at_margin=0.35)

        toe_contact_reward = 1.0 if physics.only_toes_on_ground() else 0.6
        return upright * move_reward * torso_height_reward * toe_contact_reward


class Escape(_dm_quadruped.Escape):
    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        orientation = _upright_orientation(self.random)
        _dm_quadruped._find_non_contacting_height(physics, orientation)


Fetch = _dm_quadruped.Fetch


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _make_walk(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = make_model(size=size,
                         floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
        physics = Physics.from_xml_string(xml, common.ASSETS)
        task = Move(desired_speed=_WALK_SPEED, random=random)
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **(environment_kwargs or {}))
    fn.__name__ = fn.__qualname__ = f'{size}_walk'
    return fn


def _make_run(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = make_model(size=size,
                         floor_size=_DEFAULT_TIME_LIMIT * _RUN_SPEED)
        physics = Physics.from_xml_string(xml, common.ASSETS)
        task = Move(desired_speed=_RUN_SPEED, random=random)
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **(environment_kwargs or {}))
    fn.__name__ = fn.__qualname__ = f'{size}_run'
    return fn


def _make_escape(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = make_model(size=size, floor_size=40,
                         terrain=True, rangefinders=True)
        physics = Physics.from_xml_string(xml, common.ASSETS)
        task = Escape(random=random)
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **(environment_kwargs or {}))
    fn.__name__ = fn.__qualname__ = f'{size}_escape'
    return fn


def _make_fetch(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = make_model(size=size, walls_and_ball=True)
        physics = Physics.from_xml_string(xml, common.ASSETS)
        task = Fetch(random=random)
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **(environment_kwargs or {}))
    fn.__name__ = fn.__qualname__ = f'{size}_fetch'
    return fn


# Register all task combinations (6 sizes x 4 tasks) at import time.
for _size in _SIZES:
    _dm_quadruped.SUITE.add('custom')(_make_walk(_size))
    _dm_quadruped.SUITE.add('custom')(_make_run(_size))
    _dm_quadruped.SUITE.add('custom')(_make_escape(_size))
    _dm_quadruped.SUITE.add('custom')(_make_fetch(_size))
