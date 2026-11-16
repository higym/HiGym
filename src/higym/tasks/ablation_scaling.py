"""Scaling-law ablation variants for the Walker and Reacher families.

Purpose (RLC 2026 rebuttal, Reviewers JyA6 / 26B6 / eLFJ): quantify how
sensitive the DoF-scaling results are to the per-family gear / stiffness /
damping / mass schedules, analogously to the Cheetah ablation in Appendix B.2.

Nothing in this module touches ``walker.py`` or ``reacher.py``.  Each variant
is produced by post-processing the XML those modules already emit, so the
baseline cells in the paper are bit-identical to what was run before and the
ablation cells differ only in the swept constant.

Axes
----
alpha : uniform exponent on the (gear, stiffness, damping) triple,
        factor = (baseline_per_limb / n)^alpha.
        alpha = 0   -> current benchmark setting (per-joint gear unscaled,
                       total limb torque grows linearly in n)
        alpha = 1   -> total limb torque exactly conserved
        Applying the same factor to all three preserves stiffness/gear and
        damping/gear, i.e. the passive-dynamics ratios (see Appendix B.2).

mass : (walker only) give every leg geom an explicit mass so per-leg mass is
       DoF-invariant, instead of letting MuJoCo derive it from density x
       volume -- which makes the limb ~6.6x heavier from 8 to 256 DoF because
       every extra capsule adds a constant-volume pair of end caps.

passive : (walker only) restore the original DMControl passive dynamics
          (stiffness = 0, damping = 0.1, armature = 0.01) instead of the
          cheetah-ratio-matched profiles.

stiff : (reacher only) the anti-curl spring torque at full joint deflection as
        a fraction of the joint's max motor torque.  0.20 is the benchmark
        setting.  Damping is scaled by the same factor so the joint damping
        ratio is unchanged and spring strength is varied on its own.

Task names
----------
    walker-<dof>a<AA>[m][p]-run        e.g. walker-256a25-run, walker-256a00m-run
    reacher-<dof>a<AA>-<easy|hard>     e.g. reacher-64a50-hard
    reacher-<dof>s<SSS>-<easy|hard>    e.g. reacher-64s000-hard

where AA = round(100 * alpha) and SSS = round(1000 * stiff).
"""

import os

from dm_control.rl import control
from dm_control.suite import common
from dm_control.suite import reacher as _dm_reacher
from dm_control.suite import walker as _dm_walker
from lxml import etree

from . import reacher as _hg_reacher
from . import walker as _hg_walker

# Per-leg mass of the unmodified DMControl walker (thigh + leg + foot).
_WALKER_REF_LEG_MASS = 8.9336

# Original DMControl walker <default> joint properties.
_WALKER_ORIG_DAMPING = 0.1
_WALKER_ORIG_ARMATURE = 0.01

_REACHER_BASE_STIFF = _hg_reacher._STIFFNESS_SCALE  # 0.20


# ---------------------------------------------------------------------------
# XML post-processing
# ---------------------------------------------------------------------------

def _parse(xml_bytes):
    return etree.XML(xml_bytes, etree.XMLParser(remove_blank_text=True))


#: Actuator elements whose peak torque is `gear x ctrl` with ctrl in [-1, 1].
#: Walker, reacher, cheetah, finger and humanoid_CMU are all of this kind.  The
#: quadruped is NOT: it uses <general> position servos (biastype=affine,
#: dyntype=filter) where peak torque is gainprm[0] * |act - qpos|, is
#: pose-dependent, and has a first-order lag.  Scaling `gear` there would be a
#: no-op, so this module refuses to touch such a model rather than silently
#: emitting an "ablation" identical to the baseline.
_TORQUE_ACTUATORS = ('motor',)
_SERVO_ACTUATORS = ('general', 'position', 'velocity', 'intvelocity')


def _scale_actuation(root, factor):
    """Multiply gear, joint stiffness and joint damping by `factor`."""
    if factor == 1.0:
        return root
    act = root.find('actuator')
    if act is not None:
        bad = sorted({e.tag for e in act if e.tag in _SERVO_ACTUATORS})
        if bad:
            raise NotImplementedError(
                f'alpha scaling assumes torque motors (torque = gear x ctrl) '
                f'but this model uses {bad} actuators, whose peak torque is set '
                f'by gainprm and depends on joint position. Scaling `gear` '
                f'would be a silent no-op. Define the peak-torque law for this '
                f'actuator type before ablating it.')
    n_scaled = 0
    for m in root.iter(*_TORQUE_ACTUATORS):
        if m.get('gear') is not None:
            m.set('gear', f"{float(m.get('gear')) * factor:.8g}")
            n_scaled += 1
    if n_scaled == 0:
        raise ValueError('no <motor gear=...> actuators found to scale')
    for j in root.iter('joint'):
        for attr in ('stiffness', 'damping'):
            if j.get(attr) is not None:
                j.set(attr, f"{float(j.get(attr)) * factor:.8g}")
    return root


def _scale_springs(root, factor):
    """Scale the anti-curl springs only: stiffness and its matched damping.

    Restricted to joints that carry an explicit `stiffness` attribute, which is
    exactly the set of non-shoulder arm joints built by `reacher._build_scaled_arm`.
    This deliberately excludes:
      * the <default> joint prototype (damping=0.01), and hence
      * the shoulder, which inherits that default and has no spring of its own.
    Without the restriction the shoulder's damping would track `s` -- it would
    fall to zero at s=0 and quintuple at s=1.0 -- confounding the spring axis
    with the aiming joint's passive dynamics, which `s` is not meant to touch.
    (The alpha axis is different: there `_scale_actuation` scales every damping,
    including the shoulder's, which is what keeps damping/gear invariant.)
    """
    if factor == 1.0:
        return root
    for j in root.iter('joint'):
        if j.get('stiffness') is None:
            continue
        for attr in ('stiffness', 'damping'):
            if j.get(attr) is not None:
                j.set(attr, f"{float(j.get(attr)) * factor:.8g}")
    return root


# ---------------------------------------------------------------------------
# Walker variants
# ---------------------------------------------------------------------------

def walker_model(size, alpha=0.0, conserve_mass=False, orig_passive=False):
    n_per_leg = size // 2
    root = _parse(_hg_walker.make_model(size))

    if conserve_mass:
        per_geom = _WALKER_REF_LEG_MASS / n_per_leg
        for g in root.iter('geom'):
            if (g.get('name') or '').startswith(('rgeom_', 'lgeom_')):
                g.set('mass', f'{per_geom:.8g}')

    if orig_passive:
        for j in root.iter('joint'):
            if (j.get('name') or '').startswith(('rjoint_', 'ljoint_')):
                j.set('stiffness', '0')
                j.set('damping', f'{_WALKER_ORIG_DAMPING:g}')
                j.set('armature', f'{_WALKER_ORIG_ARMATURE:g}')

    _scale_actuation(root, (_hg_walker._BASELINE_PER_LEG / n_per_leg) ** alpha)
    return etree.tostring(root)


def _make_walker(size, task, alpha, conserve_mass, orig_passive, name):
    speed = {'stand': 0, 'walk': _hg_walker._WALK_SPEED,
             'run': _hg_walker._RUN_SPEED}[task]

    def fn(time_limit=_hg_walker._DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = walker_model(size, alpha, conserve_mass, orig_passive)
        physics = _hg_walker.Physics.from_xml_string(xml, common.ASSETS)
        t = _hg_walker.PlanarWalker(move_speed=speed, random=random)
        return control.Environment(
            physics, t, time_limit=time_limit,
            control_timestep=_hg_walker._CONTROL_TIMESTEP,
            **(environment_kwargs or {}))
    fn.__name__ = fn.__qualname__ = name
    return fn


# ---------------------------------------------------------------------------
# Reacher variants
# ---------------------------------------------------------------------------

def reacher_model(size, alpha=0.0, stiff=_REACHER_BASE_STIFF):
    root = _parse(_hg_reacher.make_model(size))

    # Motor gear lives in the <default> block; make it explicit per actuator
    # so it can be scaled without touching the default (which other elements
    # may also inherit from).
    default_gear = _hg_reacher._MOTOR_GEAR
    for d in root.iter('default'):
        for m in d.iter('motor'):
            if m.get('gear') is not None:
                default_gear = float(m.get('gear'))
    act = root.find('actuator')
    if act is not None:
        for m in act.iter('motor'):
            if m.get('gear') is None:
                m.set('gear', f'{default_gear:.8g}')

    # Spring strength first (relative to the unscaled gear), then the uniform
    # actuation scale, so `stiff` keeps its meaning of "spring torque at full
    # deflection / max motor torque" at every alpha.
    _scale_springs(root, stiff / _REACHER_BASE_STIFF)
    _scale_actuation(root, (2.0 / size) ** alpha)
    return etree.tostring(root)


def _make_reacher(size, task, alpha, stiff, name):
    target = {'easy': _hg_reacher._BIG_TARGET,
              'hard': _hg_reacher._SMALL_TARGET}[task]

    def fn(time_limit=_hg_reacher._DEFAULT_TIME_LIMIT, random=None,
           environment_kwargs=None):
        xml = reacher_model(size, alpha, stiff)
        physics = _hg_reacher.Physics.from_xml_string(xml, common.ASSETS)
        t = _hg_reacher.Reacher(target_size=target, random=random)
        return control.Environment(
            physics, t, time_limit=time_limit, **(environment_kwargs or {}))
    fn.__name__ = fn.__qualname__ = name
    return fn


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _tag(alpha):
    return f'a{round(alpha * 100):02d}'


# (size, alpha, conserve_mass, orig_passive)
_WALKER_ABLATIONS = [
    # alpha response at the two DoF levels where the gear rule bites hardest
    (64, 0.25, False, False), (64, 0.50, False, False), (64, 1.00, False, False),
    (256, 0.25, False, False), (256, 0.50, False, False), (256, 1.00, False, False),
    # mass conservation (removes the 6.6x limb-mass growth confound).  With the
    # limb mass held fixed, alpha = 0.10 is the cheetah's adopted setting and the
    # edge of the walker's statically-feasible band -- see the support analysis.
    (64, 0.00, True, False), (256, 0.00, True, False), (256, 0.10, True, False),
    # full alpha sweep with mass conserved at 64 DoF: separates the torque-budget
    # effect from the limb-mass artifact, which are confounded in the rows above.
    (64, 0.25, True, False), (64, 0.50, True, False), (64, 1.00, True, False),
    # original DMControl passive dynamics (no added springs)
    (256, 0.00, False, True),
    # low-DoF negative control: the gear factor is ~1 here, so alpha must not matter
    (8, 1.00, False, False),
]

for _s, _a, _m, _p in _WALKER_ABLATIONS:
    _name_stub = f'{_s}{_tag(_a)}{"m" if _m else ""}{"p" if _p else ""}'
    for _t in ('run',):
        _dm_walker.SUITE.add('custom')(
            _make_walker(_s, _t, _a, _m, _p, f'{_name_stub}_{_t}'))


# (size, alpha, stiff)
_REACHER_ABLATIONS = [
    # alpha response (torque-budget axis)
    (64, 0.25, 0.20), (64, 0.50, 0.20), (64, 1.00, 0.20),
    (16, 0.25, 0.20), (16, 0.50, 0.20), (16, 1.00, 0.20),
    # anti-curl spring axis at fixed alpha = 0
    (64, 0.00, 0.00), (64, 0.00, 0.50), (64, 0.00, 1.00),
    # low-DoF control for the spring axis
    (4, 0.00, 0.00),
]

for _s, _a, _k in _REACHER_ABLATIONS:
    if _k == _REACHER_BASE_STIFF:
        _stub = f'{_s}{_tag(_a)}'
    else:
        _stub = f'{_s}s{round(_k * 1000):03d}'
    for _t in ('easy', 'hard'):
        _dm_reacher.SUITE.add('custom')(
            _make_reacher(_s, _t, _a, _k, f'{_stub}_{_t}'))
