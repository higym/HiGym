"""Cheetah DOF-Scaling Variant Environments (v4 — structural stiffness).

Registers cheetah-{size}_run tasks on dm_control's cheetah SUITE.
Sizes: 4, 8, 16, 32, 64, 128, 256, 512 DOF (always 2 legs).

The cheetah has 2 legs (back + front), each with 3 joints baseline
(6 actuated DOF total).  We scale to higher DOF counts by subdividing
each leg into more joints.  The body (torso, head, mass, cameras)
stays FIXED — only the leg topology changes.

Design goal: ISOLATE the effect of dimensionality while keeping the
cheetah physically viable (can stand and run) at all DOF counts.

Scaling laws:
  - Joint ranges: (3/n)^1.0 per joint — total curvature exactly conserved
  - Gear ratios: (3/n)^0.8 per joint — total torque grows mildly (~n^0.2)
    to compensate serial chain mechanical advantage loss
  - Stiffness: max(profile*(3/n)^0.8, FRAC*gear/range_rad) — structural
    minimum prevents chain buckling under gravity at high DOF
  - Damping: profile*(3/n)^0.8 — gear/damping ratio preserved (max velocity)
  - Armature: 0.1*(n/3)^0.5 — numerical stability for long chains
  - Geom mass: explicit per-geom (preserves per-leg mass regardless of DOF)

What is conserved (within 0.2 exponent tolerance):
  - Total curvature budget per leg (exact, alpha=1.0)
  - Total torque budget per leg (near-exact, alpha=0.8)
  - Gear/damping ratio per joint (exact, same alpha)
  - Total leg mass (explicit mass conservation)
  - Total body mass (14 kg via settotalmass)

Structural stiffness (key to high-DOF viability):
  At n=3 (baseline), the XML stiffness profile dominates — original
  dynamics are preserved exactly.  At high DOF, a structural minimum
  of STIFFNESS_FRAC * gear / range_rad takes over, ensuring the
  spring-to-motor torque ratio at the joint limit stays >= STIFFNESS_FRAC.
  This provides passive rigidity so the chain holds its shape under
  gravity without the policy needing to actively stabilize every joint.
"""

import collections
import os

from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import cheetah as _dm_cheetah
from dm_control.utils import io as resources
from dm_control.utils import rewards
from lxml import etree
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 10

# Supported DOF counts (must be even, >= 4).
_SIZES = [4, 8, 16, 32, 64, 128, 256, 512]
_BASELINE_PER_LEG = 3  # original cheetah has 3 joints per leg

# Range scaling exponent: 1.0 preserves total curvature exactly.
_RANGE_ALPHA = 1.0

# Gear / stiffness / damping alpha: mild sub-linear scaling ("Option B").
# gear_scale = (3/n)^_GEAR_ALPHA, applied UNIFORMLY to gear, stiffness,
# AND damping to preserve their ratios:
#   stiffness/gear = 2.0 (unchanged)
#   damping/gear   = 0.05 (unchanged)
#   v_max = gear/damping = 20 rad/s (unchanged)
#
# Alpha sweep:
#   alpha = 0.0 → walker strategy (no scaling, total torque grows as n)
#   alpha = 0.1 → current setting (total torque grows as n^0.9)
#   alpha = 0.5 → v4's setting (FAILED: per-joint too weak, nose-dive)
#   alpha = 1.0 → v1's setting (FAILED: cannot stand)
#
# At alpha=0.1, per-joint gear weakens ~9% at n=8, ~26% at n=64.
# This is a small hedge against unbounded torque growth at extreme DOF
# while staying close to the proven walker recipe.  CRUCIAL: because
# stiffness and damping scale WITH gear, the stiffness/gear ratio stays
# at 2.0 — otherwise the spring would overpower the motor and freeze
# the joints.
_GEAR_ALPHA = 0.1

# Minimum per-joint range span in degrees.
_MIN_RANGE_SPAN_DEG = 0.5

# Base geom radius from the original cheetah (constant across all DOFs).
_GEOM_RADIUS = 0.046

# Armature scaling: base * (n/3)^exp.  Grows with DOF to keep the mass
# matrix well-conditioned for long serial chains under gravity + contact.
_BASE_ARMATURE = 0.1
_ARMATURE_SCALE_EXP = 0.5

# Only every _COLLISION_STRIDE-th intermediate segment has collision.
_MAX_COLLISION_GEOMS = 6

# ---------------------------------------------------------------------------
# Leg path waypoints (hip-local frame, from cheetah.xml)
# ---------------------------------------------------------------------------

_BACK_WAYPOINTS = np.array([
    [0.0,   0.0,  0.0],      # bthigh origin (hip)
    [0.16,  0.0, -0.25],     # bshin origin (knee)
    [-0.12, 0.0, -0.39],     # bfoot origin (ankle)
    [-0.09, 0.0, -0.49],     # toe tip
])
# Original front leg waypoints (different orientation from back leg):
_FRONT_WAYPOINTS = np.array([
    [0.0,    0.0,  0.0],     # fthigh origin (hip)
    [-0.14,  0.0, -0.24],    # fshin origin (knee)
    [-0.01,  0.0, -0.42],    # ffoot origin (ankle)
    [0.035,  0.0, -0.49],    # toe tip
])
# _FRONT_WAYPOINTS = np.array([
#     [0.0,   0.0,  0.0],      # fthigh origin (hip)
#     [0.16,  0.0, -0.25],     # fshin origin (knee)
#     [-0.12, 0.0, -0.39],     # ffoot origin (ankle)
#     [-0.09, 0.0, -0.49],     # toe tip
# ])

# ---------------------------------------------------------------------------
# Joint range profiles at t = [0.0, 0.5, 1.0]  (degrees)
# ---------------------------------------------------------------------------

_BACK_RANGE_LOW = [-30, -50, -230]
_BACK_RANGE_HIGH = [60, 50, 50]
_FRONT_RANGE_LOW = [-57, -70, -28]
_FRONT_RANGE_HIGH = [30, 70, 57]

# ---------------------------------------------------------------------------
# Gear ratio profiles at t = [0.0, 0.5, 1.0]
# ---------------------------------------------------------------------------

_BACK_GEAR = [120, 90, 60]
_FRONT_GEAR = [90, 60, 30]

# ---------------------------------------------------------------------------
# Stiffness & damping profiles at t = [0.0, 0.5, 1.0]
# Damping scales with gear (same alpha) to preserve gear/damping ratio.
# Stiffness uses max(profile*gear_scale, structural_minimum).
# ---------------------------------------------------------------------------

_BACK_STIFFNESS = [240, 180, 120]
_BACK_DAMPING = [6.0, 4.5, 3.0]
_FRONT_STIFFNESS = [180, 120, 60]
_FRONT_DAMPING = [4.5, 3.0, 1.5]

# ---------------------------------------------------------------------------
# Mass conservation
# ---------------------------------------------------------------------------

def _capsule_raw_mass(half_length, radius=_GEOM_RADIUS, density=1000.0):
    """Raw mass of a capsule (cylinder + 2 hemispheres)."""
    return density * (np.pi * radius**2 * 2.0 * half_length
                      + (4.0 / 3.0) * np.pi * radius**3)

_ORIGINAL_BACK_LEG_RAW_MASS = sum(
    _capsule_raw_mass(h) for h in [0.145, 0.150, 0.094])
_ORIGINAL_FRONT_LEG_RAW_MASS = sum(
    _capsule_raw_mass(h) for h in [0.133, 0.106, 0.070])

_BACK_LEG_MASS_SCALE = 1.3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_size(size):
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
    """Place n_joints body nodes at equal arc-length along the leg path."""
    segments = np.diff(waypoints, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    total_length = seg_lengths.sum()
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

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
    relative_positions[0] = positions[0]
    for i in range(1, n_joints):
        relative_positions[i] = positions[i] - positions[i - 1]

    return positions, relative_positions, toe_pos


# ---------------------------------------------------------------------------
# Leg builder
# ---------------------------------------------------------------------------

def _build_leg(prefix, n_joints, waypoints, range_low_profile,
               range_high_profile, gear_profile, stiffness_profile,
               damping_profile, original_raw_mass):
    """Build an lxml Element for one leg body chain.

    Returns
    -------
    hip_body : lxml Element
    joint_specs : list[(name, range_low, range_high, gear, stiffness, damping)]
    """
    positions, relative_positions, toe_pos = _interpolate_path(
        waypoints, n_joints)

    t_profile = [0.0, 0.5, 1.0]

    range_scale = (_BASELINE_PER_LEG / n_joints) ** _RANGE_ALPHA
    # Shared scale for gear, stiffness, and damping — preserves all ratios.
    gear_scale = (_BASELINE_PER_LEG / n_joints) ** _GEAR_ALPHA

    # Armature: grows with DOF for mass matrix conditioning.
    armature = _BASE_ARMATURE * (n_joints / _BASELINE_PER_LEG) ** _ARMATURE_SCALE_EXP

    # Mass per geom.
    mass_per_geom = original_raw_mass / n_joints

    # Collision stride.
    n_intermediate = max(n_joints - 2, 1)
    collision_stride = max(1, n_intermediate // _MAX_COLLISION_GEOMS)

    def _fmt(v):
        return f'{v:.6g}'

    def _fmt_pos(p):
        return f'{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}'

    joint_specs = []

    parent = None
    for i in range(n_joints):
        t = i / max(n_joints - 1, 1)

        # Joint range: scaled by (3/n)^1.0 — total curvature conserved
        rng_low = np.interp(t, t_profile, range_low_profile) * range_scale
        rng_high = np.interp(t, t_profile, range_high_profile) * range_scale
        span = rng_high - rng_low
        if span < _MIN_RANGE_SPAN_DEG:
            mid = (rng_high + rng_low) / 2
            rng_low = mid - _MIN_RANGE_SPAN_DEG / 2
            rng_high = mid + _MIN_RANGE_SPAN_DEG / 2

        # Gear / stiffness / damping: all scaled by the SAME factor
        # (3/n)^_GEAR_ALPHA.  Shared scaling preserves stiffness/gear = 2.0
        # and damping/gear = 0.05 at every DOF count.
        gear = np.interp(t, t_profile, gear_profile) * gear_scale
        stiffness = np.interp(t, t_profile, stiffness_profile) * gear_scale
        damping = np.interp(t, t_profile, damping_profile) * gear_scale

        body_name = f'{prefix}seg_{i}'
        joint_name = f'{prefix}joint_{i}'
        geom_name = f'{prefix}geom_{i}'

        body = etree.Element('body')
        body.set('name', body_name)
        if i > 0:
            body.set('pos', _fmt_pos(relative_positions[i]))

        # Joint
        j = etree.SubElement(body, 'joint')
        j.set('name', joint_name)
        j.set('range', f'{_fmt(rng_low)} {_fmt(rng_high)}')
        j.set('stiffness', _fmt(stiffness))
        j.set('damping', _fmt(damping))
        j.set('armature', _fmt(armature))

        # Geom
        if i < n_joints - 1:
            fromto_end = relative_positions[i + 1]
        else:
            fromto_end = toe_pos - positions[i]

        g = etree.SubElement(body, 'geom')
        g.set('name', geom_name)
        g.set('type', 'capsule')
        g.set('fromto', f'0 0 0 {_fmt_pos(fromto_end)}')
        g.set('size', _fmt(_GEOM_RADIUS))
        g.set('mass', _fmt(mass_per_geom))
        if i % 2 == 0:
            g.set('rgba', '0.65 0.35 0.1 1')
        else:
            g.set('rgba', '0.9 0.6 0.2 1')
        if i == 0:
            g.set('contype', '0')
            g.set('conaffinity', '0')
        elif i == n_joints - 1:
            g.set('contype', '4')
            g.set('conaffinity', '1')
        else:
            dist_from_foot = (n_joints - 1) - i
            if dist_from_foot % collision_stride == 0:
                g.set('contype', '2')
                g.set('conaffinity', '1')
            else:
                g.set('contype', '0')
                g.set('conaffinity', '0')

        # Foot site
        if i == n_joints - 1:
            site = etree.SubElement(body, 'site')
            site.set('name', f'{prefix}foot')
            site.set('pos', _fmt_pos(fromto_end))
            site.set('size', '0.01')

        joint_specs.append((joint_name, rng_low, rng_high, gear, stiffness,
                            damping))

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
    """Return MuJoCo XML string for the given cheetah DOF variant."""
    n_per_leg = _validate_size(size)

    # Parse base XML
    xml_string = resources.GetResource(
        os.path.join(_TASKS_DIR, 'cheetah.xml'))
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    torso = mjcf.find('.//body[@name="torso"]')

    # 1. Remove existing leg bodies
    for name in ('bthigh', 'fthigh'):
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

    # 3b. Increase solver iterations for high-DOF models.
    if n_per_leg >= 32:
        option = mjcf.find('option')
        if option is not None:
            option.set('iterations', '400')

    # 4. Body collision setup
    asset = mjcf.find('asset')
    if asset is None:
        asset = etree.SubElement(mjcf, 'asset')
    body_mat = etree.SubElement(asset, 'material')
    body_mat.set('name', 'body_dark')
    body_mat.set('rgba', '0.9 0.6 0.2 1')

    torso_geom = torso.find('geom[@name="torso"]')
    torso_geom.set('contype', '4')
    torso_geom.set('conaffinity', '1')
    torso_geom.set('material', 'body_dark')

    head_geom = torso.find('geom[@name="head"]')
    head_geom.set('contype', '4')
    head_geom.set('conaffinity', '1')
    head_geom.set('material', 'body_dark')

    # 5. Build legs
    all_joint_specs = []

    # Back leg
    back_hip, back_specs = _build_leg(
        prefix='b', n_joints=n_per_leg,
        waypoints=_BACK_WAYPOINTS,
        range_low_profile=_BACK_RANGE_LOW,
        range_high_profile=_BACK_RANGE_HIGH,
        gear_profile=_BACK_GEAR,
        stiffness_profile=_BACK_STIFFNESS,
        damping_profile=_BACK_DAMPING,
        original_raw_mass=_ORIGINAL_BACK_LEG_RAW_MASS * _BACK_LEG_MASS_SCALE)
    back_hip.set('pos', '-0.5 0 0')
    torso.append(back_hip)
    all_joint_specs.extend(back_specs)

    # Front leg
    front_hip, front_specs = _build_leg(
        prefix='f', n_joints=n_per_leg,
        waypoints=_FRONT_WAYPOINTS,
        range_low_profile=_FRONT_RANGE_LOW,
        range_high_profile=_FRONT_RANGE_HIGH,
        gear_profile=_FRONT_GEAR,
        stiffness_profile=_FRONT_STIFFNESS,
        damping_profile=_FRONT_DAMPING,
        original_raw_mass=_ORIGINAL_FRONT_LEG_RAW_MASS)
    front_hip.set('pos', '0.5 0 0')
    torso.append(front_hip)
    all_joint_specs.extend(front_specs)

    # 6. Rebuild actuator section with motor actuators
    actuator_elem = etree.SubElement(mjcf, 'actuator')
    for jname, _, _, gear, _, _ in all_joint_specs:
        motor = etree.SubElement(actuator_elem, 'motor')
        motor.set('name', jname)
        motor.set('joint', jname)
        motor.set('gear', f'{gear:.6g}')

    return etree.tostring(mjcf, pretty_print=True)


# ---------------------------------------------------------------------------
# Physics & Task
# ---------------------------------------------------------------------------

Physics = _dm_cheetah.Physics


class Cheetah(base.Task):
    """Cheetah task — standing-pose initialization."""

    _NOISE_SCALE = 0.1
    _SETTLE_STEPS = 10

    def __init__(self, random=None):
        super().__init__(random=random)

    def initialize_episode(self, physics):
        assert physics.model.nq == physics.model.njnt

        physics.data.qpos[:] = 0.0
        physics.data.qvel[:] = 0.0

        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        half_range = (upper - lower) / 2.0
        noise = self.random.uniform(
            -self._NOISE_SCALE * half_range,
             self._NOISE_SCALE * half_range)
        physics.data.qpos[is_limited] = np.clip(noise, lower, upper)

        physics.step(nstep=self._SETTLE_STEPS)
        physics.data.qvel[:] = 0.0

        physics.data.time = 0
        self._timeout_progress = 0
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(
            physics.speed(),
            bounds=(_dm_cheetah._RUN_SPEED, float('inf')),
            margin=_dm_cheetah._RUN_SPEED,
            value_at_margin=0,
            sigmoid='linear')


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _make_run(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        model_string = make_model(size)
        physics = Physics.from_xml_string(model_string, common.ASSETS)
        task = Cheetah(random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = f'{size}_run'
    return fn


for _size in _SIZES:
    try:
        _dm_cheetah.SUITE.add('custom')(_make_run(_size))
    except ValueError:
        pass
