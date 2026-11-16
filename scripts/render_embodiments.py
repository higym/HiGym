"""Render close-up images of each embodiment at all DOF sizes (>=32)."""

import os
os.environ['MUJOCO_GL'] = 'egl'

import sys
import numpy as np
from lxml import etree

from dm_control.suite import common
from dm_control import mujoco
from dm_control.mujoco.wrapper.core import MjvOption
from PIL import Image

_TASKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'higym', 'tasks')
sys.path.insert(0, _TASKS_DIR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

WIDTH = 1920
HEIGHT = 1080


def inject_offscreen_size(xml_bytes):
    """Inject offscreen framebuffer size into the XML."""
    parser = etree.XMLParser(remove_blank_text=True, huge_tree=True)
    root = etree.XML(xml_bytes, parser)
    visual = root.find('visual')
    if visual is None:
        visual = etree.SubElement(root, 'visual')
    gl = visual.find('global')
    if gl is None:
        gl = etree.SubElement(visual, 'global')
    gl.set('offwidth', str(WIDTH))
    gl.set('offheight', str(HEIGHT))
    return etree.tostring(root, pretty_print=True)


def render_movable(physics, distance, lookat, azimuth, elevation,
                   scene_option=None, width=WIDTH, height=HEIGHT):
    """Render using a movable camera."""
    camera = mujoco.MovableCamera(physics, height=height, width=width)
    camera.set_pose(distance=distance, lookat=lookat,
                    azimuth=azimuth, elevation=elevation)
    return camera.render(scene_option=scene_option)


def save_img(img, name):
    path = os.path.join(RESULTS_DIR, f'{name}.png')
    Image.fromarray(img).save(path)
    print(f'  Saved {path}')


def no_sites_option():
    """Return MjvOption with all site groups hidden."""
    opt = MjvOption()
    opt.sitegroup[:] = 0
    return opt


# ---- Cheetah ----
# Match cheetah.png: close-up side view, slightly elevated
# Sizes: 32, 64, 128, 256, 512
def render_cheetah():
    import cheetah
    for size in [8, 16, 32, 64, 128, 256]:
        print(f'Rendering cheetah-{size}...')
        xml = inject_offscreen_size(cheetah.make_model(size))
        physics = cheetah.Physics.from_xml_string(xml, common.ASSETS)
        physics.step(nstep=25)
        lookat = physics.named.data.xpos['torso'].copy()
        img = render_movable(physics, distance=1.8, lookat=lookat,
                             azimuth=90, elevation=-20)
        save_img(img, f'cheetah-{size}')


# ---- Finger ----
# Side view showing finger + spinner, hide sites (pink balls are touchtop/touchbottom)
# Sizes: 32, 64
def render_finger():
    import finger
    opt = no_sites_option()
    for size in [32, 64]:
        print(f'Rendering finger-{size}...')
        xml = inject_offscreen_size(finger.make_model(size))
        physics = finger.Physics.from_xml_string(xml, common.ASSETS)
        physics.step(nstep=50)
        spinner_pos = physics.named.data.xpos['spinner'].copy()
        finger_pos = physics.named.data.xpos['proximal'].copy()
        lookat = (spinner_pos + finger_pos) / 2.0
        lookat[2] = 0.35
        img = render_movable(physics, distance=0.9, lookat=lookat,
                             azimuth=90, elevation=-20, scene_option=opt)
        save_img(img, f'finger-{size}')


# ---- Walker ----
# Camera side view is fine. Ensure standing straight: zero joints, minimal settling.
# Sizes: 32, 64, 128, 256
def render_walker():
    import walker
    for size in [32, 64, 128, 256]:
        print(f'Rendering walker-{size}...')
        xml = inject_offscreen_size(walker.make_model(size))
        physics = walker.Physics.from_xml_string(xml, common.ASSETS)
        # Zero all joint positions for straight standing
        physics.data.qpos[:] = 0
        physics.data.qvel[:] = 0
        # Set root height so feet touch ground (~1.3m from torso to feet)
        physics.forward()
        img = render_movable(physics, distance=3.0,
                             lookat=physics.named.data.xpos['torso'].copy(),
                             azimuth=90, elevation=-15)
        save_img(img, f'walker-{size}')


# ---- Reacher ----
# Hard task: small target placed randomly. Top-down view, slightly zoomed out.
# Sizes: 32, 64
def render_reacher():
    import reacher
    rng = np.random.RandomState(42)
    for size in [32, 64]:
        print(f'Rendering reacher-{size}...')
        xml = inject_offscreen_size(reacher.make_model(size))
        physics = reacher.Physics.from_xml_string(xml, common.ASSETS)
        # Set hard task target (small ball, randomly placed)
        physics.named.model.geom_size['target', 0] = 0.015  # _SMALL_TARGET
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.05, 0.20)
        physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
        physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
        physics.step(nstep=50)
        # Top-down, slightly zoomed out vs "fixed" camera (z=0.75)
        img = render_movable(physics, distance=0.85,
                             lookat=np.array([0.0, 0.0, 0.01]),
                             azimuth=0, elevation=-90)
        save_img(img, f'reacher-{size}')


# ---- Quadruped ----
# Match quadruped.png angle (azimuth=135, elevation=-30) but slightly more zoomed out
# Sizes: 32, 64, 128, 256, 512
def render_quadruped():
    import quadruped
    from dm_control.suite import quadruped as _dm_quadruped
    for size in [16, 32, 64, 128, 256, 512]:
        print(f'Rendering quadruped-{size}...')
        xml = inject_offscreen_size(quadruped.make_model(size))
        physics = quadruped.Physics.from_xml_string(xml, common.ASSETS)
        quat = np.array([1.0, 0, 0, 0])
        _dm_quadruped._find_non_contacting_height(physics, quat)
        physics.step(nstep=200)
        lookat = physics.named.data.xpos['torso'].copy()
        img = render_movable(physics, distance=3.0, lookat=lookat,
                             azimuth=135, elevation=-30)
        save_img(img, f'quadruped-{size}')


# ---- Humanoid CMU ----
# "front" camera (trackcom)
# Only size: 80
def render_humanoid_cmu():
    import humanoid_CMU
    print('Rendering humanoid_CMU-80...')
    xml = inject_offscreen_size(humanoid_CMU.make_model())
    physics = humanoid_CMU.Physics.from_xml_string(xml, common.ASSETS)
    physics.reset()
    physics.step(nstep=100)
    img = physics.render(height=HEIGHT, width=WIDTH, camera_id='front')
    save_img(img, 'humanoid_CMU-80')


# ---- Reacher Free ----
# Same as reacher but different scaling. XML camera "fixed"
# Sizes: 32, 64
def render_reacher_free():
    from dm_control.suite import reacher as _dm_reacher
    _orig_add = _dm_reacher.SUITE.add
    def _noop_add(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    _dm_reacher.SUITE.add = _noop_add
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'reacher_free_mod', os.path.join(_TASKS_DIR, 'reacher_free.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        _dm_reacher.SUITE.add = _orig_add

    rng = np.random.RandomState(42)
    for size in [32, 64]:
        print(f'Rendering reacher_free-{size}...')
        xml = inject_offscreen_size(mod.make_model(size))
        physics = mujoco.Physics.from_xml_string(xml, common.ASSETS)
        physics.named.model.geom_size['target', 0] = 0.015
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0.05, 0.20)
        physics.named.model.geom_pos['target', 'x'] = radius * np.sin(angle)
        physics.named.model.geom_pos['target', 'y'] = radius * np.cos(angle)
        physics.step(nstep=50)
        img = render_movable(physics, distance=0.85,
                             lookat=np.array([0.0, 0.0, 0.01]),
                             azimuth=0, elevation=-90)
        save_img(img, f'reacher_free-{size}')


# ---- Cheetah leg (back leg close-up) ----
# Sizes: 16, 32, 64, 128, 256  (size=8 is the baseline full-body render)
def render_cheetah_leg():
    import cheetah
    W = H = 640
    for size in [16, 32, 64, 128, 256]:
        print(f'Rendering cheetah_leg-{size}...')
        xml = inject_offscreen_size(cheetah.make_model(size))
        physics = cheetah.Physics.from_xml_string(xml, common.ASSETS)
        physics.step(nstep=25)

        hip_pos = physics.named.data.xpos['bseg_0'].copy()
        toe_pos = physics.named.data.site_xpos['bfoot'].copy()
        lookat  = (hip_pos + toe_pos) / 2.0
        leg_len = float(np.linalg.norm(toe_pos - hip_pos))
        dist    = max(0.45, leg_len * 2.0)

        img = render_movable(physics, distance=dist, lookat=lookat,
                             azimuth=90, elevation=-15, width=W, height=H)
        save_img(img, f'cheetah_leg-{size}')


# ---- Quadruped leg (front-left leg close-up) ----
# Sizes: 32, 64, 128, 256, 512  (size=16 is the baseline full-body render)
def render_quadruped_leg():
    import quadruped as _quad
    from dm_control.suite import quadruped as _dm_quad
    W = H = 640
    for size in [32, 64, 128, 256, 512]:
        print(f'Rendering quadruped_leg-{size}...')
        xml = inject_offscreen_size(_quad.make_model(size))
        physics = _quad.Physics.from_xml_string(xml, common.ASSETS)

        quat = np.array([1.0, 0.0, 0.0, 0.0])
        _dm_quad._find_non_contacting_height(physics, quat)
        physics.step(nstep=200)

        hip_pos = physics.named.data.xpos['hip_front_left'].copy()
        toe_pos = physics.named.data.xpos['toe_front_left'].copy()
        lookat  = (hip_pos + toe_pos) / 2.0
        # Fixed distance so all DoFs are framed the same way regardless of
        # how the physics settles (high-DoF legs can extend further, which
        # would otherwise pull the camera back).
        dist = 1.05

        img = render_movable(physics, distance=dist, lookat=lookat,
                             azimuth=135, elevation=-10, width=W, height=H)
        save_img(img, f'quadruped_leg-{size}')


if __name__ == '__main__':
    render_cheetah()
    render_finger()
    render_walker()
    render_reacher()
    render_quadruped()
    render_humanoid_cmu()
    render_reacher_free()
    render_quadruped_leg()
    print('Done!')
