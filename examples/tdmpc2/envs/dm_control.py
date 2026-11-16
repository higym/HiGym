# envs/dm_control.py
import gymnasium as gym

from higym import HiGymEnv


class TDMPC2Wrapper(gym.Wrapper):
	"""
	What TD-MPC2's trainer expects on top of HiGymEnv: the task seeded once at
	construction, terminated/truncated/success in info, max_episode_steps, and
	render(width, height).
	"""

	def __init__(self, env, domain, seed):
		super().__init__(env)
		self._seed = seed
		self.camera_id = 2 if domain == 'quadruped' else 0
		self.max_episode_steps = env.max_episode_steps

	def _info(self, info, terminated=False, truncated=False):
		return {
			'terminated': terminated,
			'truncated': truncated,
			'success': float('nan'),
			'score': info['score'],
		}

	def reset(self, **kwargs):
		obs, info = self.env.reset(seed=kwargs.get('seed', self._seed))
		self._seed = None  # later episodes continue the task's random stream
		return obs, self._info(info)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		return obs, reward, terminated, truncated, self._info(info, terminated, truncated)

	def render(self, width=224, height=224, camera_id=None):
		physics = self.env.unwrapped.dm_env.physics
		return physics.render(height, width, camera_id or self.camera_id).copy()


def make_env(cfg):
	"""
	Make a HiGym environment, e.g. cfg.task = 'cheetah-128_run'.
	"""
	domain, task = cfg.task.split('-', 1)
	env = HiGymEnv(domain, task, action_repeat=cfg.action_repeat)
	return TDMPC2Wrapper(env, domain, cfg.seed)
