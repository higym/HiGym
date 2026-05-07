"""Finger DOF-Scaling Variant Environments.

Registers finger-{size}_{spin,turn_easy,turn_hard} tasks on dm_control's
finger SUITE.  Sizes: 2, 4, 8, 16, 32, 64 DOF.

The original finger has 2 actuated joints (proximal + distal) that
manipulate a free-spinning hinge body.  Gravity is disabled — this is
a pure planar manipulation task in the XZ plane.

We scale by subdividing the finger into more joints along the same
physical path.  The spinner body and its hinge remain unchanged.

Scaling laws (chosen from experimental testing):
  - Joint ranges: (2/n)^1.0 per joint — preserves total curvature at 440 deg
  - Gear ratios: interpolated profile 30→15 along finger, scaled by (2/n)^0.55
  - Damping: 2.5 for n≤32; scaled by (2/n)^0.55 at n=64 to restore gear/damp ratio
  - Geom radii: interpolated from original 0.03→0.028→0.03 (constant w.r.t. DOF)
  - Geom mass: explicit per-geom mass = original_total / n_joints
  - Spinner mass (n>2): density scaled by (2/n)^0.55, matching finger torque
  - Stiffness (n=64): 0.25 * gear / range_rad — passive structural rigidity
    against buckling on contact; motor needs only 25% effort to overcome
  - Solver iterations: 400 at n=64 (up from 200) for constraint convergence

What scales with DOF:
  - Number of joints/actuators in the finger chain
  - Per-joint range (shrinks to preserve total curvature)
  - Per-joint gear (shrinks by (2/n)^0.55 to conserve total torque)
  - Per-geom mass (explicit, = original_total / n_joints)
  - Spinner mass (density × gear_scale, preserves torque/inertia ratio)

64-DOF specific tuning (chain rigidity fix):
  - Per-joint stiffness (proportional to gear/range, prevents buckling)
  - Scaled damping (gear/damping ratio restored to original 12→6)
  - Solver iterations increased to 400

What does NOT scale:
  - Total finger length (proximal joint to fingertip)
  - Total finger mass (conserved via explicit mass)
  - Per-joint damping (constant at 2.5 for n≤32)
  - Spinner geometry and position (shape unchanged, only mass scales)
  - Contact physics (fingertip collides with spinner)
  - Gravity (disabled), physics options, timestep
  - Touch sensors (on fingertip)
  - Task semantics (spin, turn_easy, turn_hard)
"""

import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite import finger as _dm_finger
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import io as resources
from dm_control.utils import rewards
from lxml import etree
import numpy as np

_TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tasks')

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = .02

# Target sizes for the Turn task (same as original).
_EASY_TARGET_SIZE = 0.07
_HARD_TARGET_SIZE = 0.03

# Spin task constants (same as original).
_SPIN_VELOCITY = 15.0

# Supported DOF counts (number of finger joints).
_SIZES = [2, 4, 8, 16, 32, 64]
_BASELINE_JOINTS = 2  # original finger has 2 joints

# Joint range scaling exponent (1.0 preserves total curvature budget).
_RANGE_ALPHA = 1.0
_BASE_RANGE_DEG = 110.0  # original symmetric range for each joint

# Gear ratio profile at t = [0.0, 1.0] — base values before DOF scaling.
_GEAR_PROFILE = [30.0, 15.0]

# Gear conservation exponent: gear_scale = (2/n)^exp.
# 0.7 → original, changed for 64 dof, total torque grows mildly, matching cheetah/quadruped pattern.
_GEAR_SCALE_EXP = 0.85

# Damping: constant at original value (no DOF scaling needed with gear conservation).
_BASE_DAMPING = 2.5

# Stiffness fraction: per-joint stiffness = frac * gear / range_rad.
# At the joint limit, the restoring torque is (frac × motor_max_torque),
# so the motor needs only `frac` of max effort to overcome the spring.
# This provides passive structural rigidity against buckling on contact
# while leaving the chain freely controllable.  Only applied for n > 2
# (original 2-DOF finger has no stiffness).
_STIFFNESS_FRAC = 0.25

# Geom radius profile at t = [0.0, 0.5, 1.0] (original values).
_RADIUS_PROFILE_T = [0.0, 0.5, 1.0]
_RADIUS_PROFILE_V = [0.03, 0.028, 0.03]

# Fingertip radius (original).
_FINGERTIP_RADIUS = 0.03

# Finger waypoints in proximal-joint local frame (XZ plane).
# Extracted from finger.xml: proximal goes down 0.18, distal goes down 0.16.
_FINGER_WAYPOINTS = np.array([
    [0.0, 0.0,  0.0],     # proximal joint
    [0.0, 0.0, -0.18],    # distal joint
    [0.0, 0.0, -0.34],    # fingertip end
])

# Original finger mass — computed from the 2-joint finger.xml capsule geometry.
# proximal: capsule r=0.03, fromto=(0,0,0)→(0,0,-0.17), half_length=0.085
# distal:   capsule r=0.028, fromto=(0,0,0)→(0,0,-0.16), half_length=0.08
# fingertip: capsule r=0.03, fromto=(0,0,-0.13)→(0,0,-0.161), half_length=0.0155
# density = 1000 (MuJoCo default)
def _compute_original_finger_mass():
    import math as _m
    density = 1000.0
    def _cap_mass(radius, half_length):
        return density * (_m.pi * radius**2 * 2.0 * half_length
                          + (4.0 / 3.0) * _m.pi * radius**3)
    proximal = _cap_mass(0.03, 0.085)
    distal = _cap_mass(0.028, 0.08)
    fingertip = _cap_mass(0.03, 0.0155)
    return proximal + distal + fingertip

_ORIGINAL_FINGER_MASS = _compute_original_finger_mass()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_size(size):
    """Validate DOF count and return it."""
    if not isinstance(size, int):
        raise TypeError(f'size must be int, got {type(size).__name__}')
    if size <= 0:
        raise ValueError(f'size must be positive, got {size}')
    if size not in _SIZES:
        raise ValueError(f'Unsupported size {size}. Supported: {_SIZES}')
    return size


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _interpolate_path(waypoints, n_joints):
    """Place n_joints body nodes at equal arc-length along the finger path.

    Returns
    -------
    positions : (n_joints, 3)
        Absolute positions in proximal-joint local frame.
    relative_positions : (n_joints, 3)
        Each row is position of body i relative to body i-1.
    fingertip_pos : (3,)
        Position of the fingertip end.
    """
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

    fingertip_pos = waypoints[-1]

    relative_positions = np.zeros_like(positions)
    relative_positions[0] = positions[0]  # [0,0,0]
    for i in range(1, n_joints):
        relative_positions[i] = positions[i] - positions[i - 1]

    return positions, relative_positions, fingertip_pos


# ---------------------------------------------------------------------------
# Finger builder
# ---------------------------------------------------------------------------

def _build_finger(n_joints):
    """Build an lxml Element for the finger body chain.

    The finger is a serial chain of n_joints hinge joints in the XZ plane,
    replacing the original 2-body (proximal + distal) structure.

    Returns
    -------
    finger_body : lxml Element
        The root body element (placed at the proximal joint position).
    joint_specs : list[(name, range_low_deg, range_high_deg, gear, damping)]
        Per-joint specifications for actuator and sensor creation.
    """
    positions, relative_positions, fingertip_pos = _interpolate_path(
        _FINGER_WAYPOINTS, n_joints)

    range_scale = (_BASELINE_JOINTS / n_joints) ** _RANGE_ALPHA
    per_joint_range = _BASE_RANGE_DEG * range_scale
    joint_limit_rad = np.radians(per_joint_range)
    gear_exp = _GEAR_SCALE_EXP if n_joints == 64 else 0.7
    gear_scale = (_BASELINE_JOINTS / n_joints) ** gear_exp
    if n_joints == 64:
        # Scale damping proportionally to gear so the gear/damping ratio
        # (which determines max joint velocity) matches the original 2-DOF.
        # Original: gear/damp = 30/2.5=12 (prox), 15/2.5=6 (dist).
        # At 64 DOF with gear_scale≈0.15: unscaled damping gives ratio<2,
        # making the finger far too slow to generate meaningful impulse.
        damping = _BASE_DAMPING * gear_scale
    else:
        damping = _BASE_DAMPING  # constant for all other sizes
    mass_per_geom = _ORIGINAL_FINGER_MASS / n_joints

    def _fmt(v):
        return f'{v:.6g}'

    def _fmt_pos(p):
        return f'{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}'

    joint_specs = []

    # Root finger body
    finger_body = etree.Element('body')
    finger_body.set('name', 'proximal')
    finger_body.set('pos', '-0.2 0 0.4')
    finger_body.set('childclass', 'finger')

    # Decoration cylinder at the pivot
    dec = etree.SubElement(finger_body, 'geom')
    dec.set('name', 'proximal_decoration')
    dec.set('type', 'cylinder')
    dec.set('fromto', '0 -0.033 0 0 0.033 0')
    dec.set('size', '0.034')
    dec.set('material', 'decoration')

    parent = finger_body
    for i in range(n_joints):
        t = i / max(n_joints - 1, 1)

        # Gear: interpolated along the chain, scaled by DOF
        gear = float(np.interp(t, [0.0, 1.0], _GEAR_PROFILE)) * gear_scale

        # Radius: interpolated from original profile
        radius = float(np.interp(t, _RADIUS_PROFILE_T, _RADIUS_PROFILE_V))

        # Naming: first two keep original names for compatibility
        if i == 0:
            joint_name = 'proximal'
            geom_name = 'proximal'
        elif i == 1:
            joint_name = 'distal'
            geom_name = 'distal'
        else:
            joint_name = f'joint_{i}'
            geom_name = f'seg_{i}'

        # Create child body for all joints after the first
        if i > 0:
            child = etree.SubElement(parent, 'body')
            if i == 1:
                child.set('name', 'distal')
            else:
                child.set('name', f'seg_body_{i}')
            child.set('pos', _fmt_pos(relative_positions[i]))
            child.set('childclass', 'finger')
            parent = child

        # Joint
        j = etree.SubElement(parent, 'joint')
        j.set('name', joint_name)
        j.set('range', f'{_fmt(-per_joint_range)} {_fmt(per_joint_range)}')
        j.set('damping', _fmt(damping))
        if(n_joints==64):
            j.set('armature', '0.01')
        # Stiffness for structural rigidity: prevents the chain from
        # buckling on contact.  Proportional to gear/range so restoring
        # torque at the joint limit is _STIFFNESS_FRAC of motor max torque.
        # Skipped at n=2 to preserve original 2-DOF dynamics exactly.
        if n_joints == 64 and _STIFFNESS_FRAC > 0:
            stiffness = _STIFFNESS_FRAC * gear / joint_limit_rad
            j.set('stiffness', _fmt(stiffness))
        if i == 0:
            j.set('ref', '-90')

        # Geom: capsule from origin to next body offset (or toward fingertip)
        if i < n_joints - 1:
            fromto_end = relative_positions[i + 1]
        else:
            fromto_end = fingertip_pos - positions[i]

        g = etree.SubElement(parent, 'geom')
        g.set('name', geom_name)
        g.set('type', 'capsule')
        g.set('material', 'self')
        g.set('size', _fmt(radius))
        g.set('fromto', f'0 0 0 {_fmt_pos(fromto_end)}')
        g.set('mass', _fmt(mass_per_geom))
        # Non-terminal segments: disable collision to avoid interfering
        # with the finger-spinner interaction.  Only the fingertip collides.
        g.set('contype', '0')
        g.set('conaffinity', '0')
        # Alternate colors for visibility
        if i % 2 == 0:
            g.set('rgba', '0.9 0.6 0.2 1')
        else:
            g.set('rgba', '0.65 0.35 0.1 1')

        joint_specs.append((joint_name, -per_joint_range, per_joint_range,
                            gear, damping))

    # Fingertip geom on the last body — this is the effector that contacts
    # the spinner.  Uses default contype/conaffinity (1/1) for collision.
    ft_end = fingertip_pos - positions[-1]
    ft_start = ft_end * 0.7  # slight overlap for visual continuity
    fingertip_g = etree.SubElement(parent, 'geom')
    fingertip_g.set('name', 'fingertip')
    fingertip_g.set('type', 'capsule')
    fingertip_g.set('size', _fmt(_FINGERTIP_RADIUS))
    fingertip_g.set('material', 'effector')
    fingertip_g.set('fromto', f'{_fmt_pos(ft_start)} {_fmt_pos(ft_end)}')

    # Touch sensor sites on the fingertip
    ts1 = etree.SubElement(parent, 'site')
    ts1.set('name', 'touchtop')
    ts1.set('pos', _fmt_pos(ft_end + np.array([0.01, 0, 0])))

    ts2 = etree.SubElement(parent, 'site')
    ts2.set('name', 'touchbottom')
    ts2.set('pos', _fmt_pos(ft_end + np.array([-0.01, 0, 0])))

    return finger_body, joint_specs


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def make_model(size=2):
    """Return MuJoCo XML string for the given finger DOF variant.

    The spinner body, arena, physics options, and cameras are identical
    to the original 2-DOF finger.  Only the finger chain is rebuilt.
    """
    n_joints = _validate_size(size)

    xml_string = resources.GetResource(os.path.join(_TASKS_DIR, 'finger.xml'))
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    worldbody = mjcf.find('worldbody')

    # 1. Remove existing finger body (proximal + distal chain)
    old_finger = worldbody.find("./body[@name='proximal']")
    if old_finger is not None:
        worldbody.remove(old_finger)

    # 2. Remove old actuator section
    old_act = mjcf.find('actuator')
    if old_act is not None:
        mjcf.remove(old_act)

    # 3. Remove old sensor section (rebuilt completely)
    old_sensor = mjcf.find('sensor')
    if old_sensor is not None:
        mjcf.remove(old_sensor)

    # 4. Increase arena memory for large models
    if size > 16:
        size_elem = mjcf.find('size')
        if size_elem is None:
            size_elem = etree.SubElement(mjcf, 'size')
        size_elem.set('memory', f'{max(20, size)}M')

    # 4b. Increase solver iterations for high-DOF models.
    # At 32+ joints, the constraint solver must satisfy many joint limits
    # plus fingertip contact simultaneously; 200 iterations may not converge.
    if size == 64:
        option = mjcf.find('option')
        if option is not None:
            option.set('iterations', '400')

    # 5. Scale spinner mass for high-DOF models.
    # The effective tip mass of a long serial chain drops dramatically
    # (chain buckles on contact), so we reduce the spinner's inertia
    # proportionally to the finger's torque capacity (gear_scale) to
    # preserve the torque/inertia ratio and keep the task learnable.
    # At n=2 gear_scale=1.0 (no change); density scales smoothly with DOF.
    spinner_body = worldbody.find("./body[@name='spinner']")
    if n_joints == 64:
        gear_scale = (_BASELINE_JOINTS / n_joints) ** _GEAR_SCALE_EXP
        scaled_density = 1000.0 * gear_scale  # default density=1000
        for geom in spinner_body.findall('geom'):
            geom.set('density', f'{scaled_density:.6g}')
    if n_joints == 64:
        # Scale hinge frictionloss: the 64-DOF finger delivers many small
        # taps rather than one big hit.  With original friction the spinner
        # decelerates between taps and just rocks back and forth.  Scaling
        # friction by gear_scale lets angular momentum accumulate over
        # multiple contacts, enabling sustained rotation.
        gear_scale_64 = (_BASELINE_JOINTS / 64) ** _GEAR_SCALE_EXP
        hinge_joint = spinner_body.find("./joint[@name='hinge']")
        if hinge_joint is not None:
            base_frictionloss = 0.1  # original value from finger.xml
            hinge_joint.set('frictionloss', f'{base_frictionloss * gear_scale_64:.6g}')

    # 6. Build and insert new finger chain
    finger_body, joint_specs = _build_finger(n_joints)
    # Insert finger body before the spinner body
    spinner_idx = list(worldbody).index(spinner_body)
    worldbody.insert(spinner_idx, finger_body)

    # 6. Rebuild actuator section
    actuator_elem = etree.SubElement(mjcf, 'actuator')
    for jname, _, _, gear, _ in joint_specs:
        motor = etree.SubElement(actuator_elem, 'motor')
        motor.set('name', jname)
        motor.set('joint', jname)
        motor.set('gear', f'{gear:.6g}')

    # 7. Rebuild sensor section
    sensor_elem = etree.SubElement(mjcf, 'sensor')

    # Joint position sensors (all finger joints)
    for jname, _, _, _, _ in joint_specs:
        s = etree.SubElement(sensor_elem, 'jointpos')
        s.set('name', jname)
        s.set('joint', jname)

    # Joint velocity sensors (all finger joints)
    for jname, _, _, _, _ in joint_specs:
        s = etree.SubElement(sensor_elem, 'jointvel')
        s.set('name', f'{jname}_velocity')
        s.set('joint', jname)

    # Hinge velocity
    s = etree.SubElement(sensor_elem, 'jointvel')
    s.set('name', 'hinge_velocity')
    s.set('joint', 'hinge')

    # Frame position sensors
    for objname, objtype in [('tip', 'site'), ('target', 'site'),
                             ('spinner', 'xbody')]:
        s = etree.SubElement(sensor_elem, 'framepos')
        s.set('name', objname)
        s.set('objtype', objtype)
        s.set('objname', objname)

    # Touch sensors
    for site_name in ['touchtop', 'touchbottom']:
        s = etree.SubElement(sensor_elem, 'touch')
        s.set('name', site_name)
        s.set('site', site_name)

    # Touch position sensors
    for site_name in ['touchtop', 'touchbottom']:
        s = etree.SubElement(sensor_elem, 'framepos')
        s.set('name', f'{site_name}_pos')
        s.set('objtype', 'site')
        s.set('objname', site_name)

    return etree.tostring(mjcf, pretty_print=True)


def get_model_and_assets(size=2):
    """Returns (xml_string, assets_dict) for the scaled finger."""
    return make_model(size=size), common.ASSETS


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

class Physics(mujoco.Physics):
    """Physics simulation with features for the scaled Finger domain.

    Adapts the original finger.Physics to work with variable joint counts.
    All observations are derived from sensors for differentiability.
    """

    def _reload_from_data(self, data):
        super()._reload_from_data(data)
        # Discover joint position and velocity sensor names
        self._joint_pos_sensors = []
        self._joint_vel_sensors = []
        for i in range(self.model.nsensor):
            name = self.model.id2name(i, 'sensor')
            if name.endswith('_velocity'):
                self._joint_vel_sensors.append(name)
            elif name in ('tip', 'target', 'spinner', 'touchtop', 'touchbottom',
                          'touchtop_pos', 'touchbottom_pos', 'hinge_velocity'):
                pass  # skip non-jointpos sensors
            else:
                self._joint_pos_sensors.append(name)

    def touch(self):
        """Returns logarithmically scaled signals from the two touch sensors."""
        return np.log1p(self.named.data.sensordata[['touchtop', 'touchbottom']])

    def hinge_velocity(self):
        """Returns the velocity of the hinge joint."""
        return self.named.data.sensordata['hinge_velocity']

    def tip_position(self):
        """Returns the (x,z) position of the tip relative to the hinge."""
        return (self.named.data.sensordata['tip'][[0, 2]] -
                self.named.data.sensordata['spinner'][[0, 2]])

    def bounded_position(self):
        """Returns all joint positions and the tip position.

        For the original 2-DOF finger, this returns [proximal, distal, tip_x, tip_z].
        For higher DOF, returns all joint positions plus tip position.
        """
        joint_pos = self.named.data.sensordata[self._joint_pos_sensors]
        return np.hstack((joint_pos, self.tip_position()))

    def velocity(self):
        """Returns all joint velocities (including hinge)."""
        return self.named.data.sensordata[self._joint_vel_sensors]

    def target_position(self):
        """Returns the (x,z) position of the target relative to the hinge."""
        return (self.named.data.sensordata['target'][[0, 2]] -
                self.named.data.sensordata['spinner'][[0, 2]])

    def to_target(self):
        """Returns the vector from the tip to the target."""
        return self.target_position() - self.tip_position()

    def dist_to_target(self):
        """Returns the signed distance to the target surface, negative is inside."""
        return (np.linalg.norm(self.to_target()) -
                self.named.model.site_size['target', 0])


# ---------------------------------------------------------------------------
# Task classes
# ---------------------------------------------------------------------------

def _set_random_joint_angles(physics, random, max_attempts=1000):
    """Sets the joints to a random collision-free state."""
    for _ in range(max_attempts):
        randomizers.randomize_limited_and_rotational_joints(physics, random)
        physics.after_reset()
        if physics.data.ncon == 0:
            break
    else:
        raise RuntimeError(
            f'Could not find a collision-free state after {max_attempts} attempts')


class Spin(base.Task):
    """A Finger Task to spin the stopped body (same as original)."""

    def __init__(self, random=None, n_joints=2):
        self._n_joints = n_joints
        super().__init__(random=random)

    def initialize_episode(self, physics):
        physics.named.model.site_rgba['target', 3] = 0
        physics.named.model.site_rgba['tip', 3] = 0
        # Scale hinge viscous damping with DOF: the 64-DOF finger delivers
        # many small taps rather than one big hit, so the spinner must coast
        # longer between contacts for angular momentum to accumulate.
        base_hinge_damping = 0.03
        if self._n_joints == 64:
            gear_scale = (_BASELINE_JOINTS / self._n_joints) ** _GEAR_SCALE_EXP
            physics.named.model.dof_damping['hinge'] = base_hinge_damping * gear_scale
        else:
            physics.named.model.dof_damping['hinge'] = base_hinge_damping
        _set_random_joint_angles(physics, self.random)
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.bounded_position()
        obs['velocity'] = physics.velocity()
        obs['touch'] = physics.touch()
        return obs

    def get_reward(self, physics):
        return float(physics.hinge_velocity() <= -_SPIN_VELOCITY)


class Turn(base.Task):
    """A Finger Task to turn the body to a target angle (same as original)."""

    def __init__(self, target_radius, random=None):
        self._target_radius = target_radius
        super().__init__(random=random)

    def initialize_episode(self, physics):
        target_angle = self.random.uniform(-np.pi, np.pi)
        hinge_x, hinge_z = physics.named.data.xanchor['hinge', ['x', 'z']]
        radius = physics.named.model.geom_size['cap1'].sum()
        target_x = hinge_x + radius * np.sin(target_angle)
        target_z = hinge_z + radius * np.cos(target_angle)
        physics.named.model.site_pos['target', ['x', 'z']] = target_x, target_z
        physics.named.model.site_size['target', 0] = self._target_radius

        _set_random_joint_angles(physics, self.random)

        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.bounded_position()
        obs['velocity'] = physics.velocity()
        obs['touch'] = physics.touch()
        obs['target_position'] = physics.target_position()
        obs['dist_to_target'] = physics.dist_to_target()
        return obs

    def get_reward(self, physics):
        return float(physics.dist_to_target() <= 0)


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

def _make_spin(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        physics = Physics.from_xml_string(*get_model_and_assets(size=size))
        task = Spin(random=random, n_joints=size)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = f'{size}_spin'
    return fn


def _make_turn_easy(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        physics = Physics.from_xml_string(*get_model_and_assets(size=size))
        task = Turn(target_radius=_EASY_TARGET_SIZE, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = f'{size}_turn_easy'
    return fn


def _make_turn_hard(size):
    def fn(time_limit=_DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        physics = Physics.from_xml_string(*get_model_and_assets(size=size))
        task = Turn(target_radius=_HARD_TARGET_SIZE, random=random)
        environment_kwargs = environment_kwargs or {}
        return control.Environment(
            physics, task, time_limit=time_limit,
            control_timestep=_CONTROL_TIMESTEP,
            **environment_kwargs)
    fn.__name__ = fn.__qualname__ = f'{size}_turn_hard'
    return fn


# Register all task combinations (6 sizes x 3 tasks) at import time.
for _size in _SIZES:
    _dm_finger.SUITE.add('custom')(_make_spin(_size))
    _dm_finger.SUITE.add('custom')(_make_turn_easy(_size))
    _dm_finger.SUITE.add('custom')(_make_turn_hard(_size))
