# envs/dm_control.py
import gymnasium as gym
import numpy as np

from envs.tasks import quadruped, reacher, walker, cheetah, finger, humanoid_CMU
from dm_control import suite
suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
from dm_control.suite.wrappers import action_scale

from envs.wrappers.timeout import Timeout


def get_obs_shape(env):
	obs_shp = []
	for v in env.observation_spec().values():
		try:
			shp = np.prod(v.shape)
		except:
			shp = 1
		obs_shp.append(shp)
	return (int(np.sum(obs_shp)),)


class DMControlWrapper:
	def __init__(self, env, domain, action_repeat=2):
		self.env = env
		self.action_repeat = action_repeat
		self.camera_id = 2 if domain == 'quadruped' else 0
		obs_shape = get_obs_shape(env)
		action_shape = env.action_spec().shape
		self.observation_space = gym.spaces.Box(
			low=np.full(obs_shape, -np.inf, dtype=np.float32),
			high=np.full(obs_shape, np.inf, dtype=np.float32),
			dtype=np.float32)
		self.action_space = gym.spaces.Box(
			low=np.full(action_shape, env.action_spec().minimum),
			high=np.full(action_shape, env.action_spec().maximum),
			dtype=env.action_spec().dtype)
		self.action_spec_dtype = env.action_spec().dtype
		self._cumulative_reward = 0

	@property
	def unwrapped(self):
		return self.env
	
	@property
	def metadata(self):
		return None
	
	@property
	def info(self):
		return {
			'terminated': False,
			'truncated': False,
			'success': float('nan'),
			'score': self._cumulative_reward/1000,
		}
	
	def _obs_to_array(self, obs):
		return np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32)
	
	def reset(self):
		self._cumulative_reward = 0
		return self._obs_to_array(self.env.reset().observation), self.info

	def step(self, action):
		reward = 0
		action = action.astype(self.action_spec_dtype)
		for _ in range(self.action_repeat):
			step = self.env.step(action)
			reward += step.reward
		self._cumulative_reward += reward
		return self._obs_to_array(step.observation), reward, False, False, self.info
	
	def render(self, width=224, height=224, camera_id=None):
		return self.env.physics.render(height, width, camera_id or self.camera_id).copy()
	
	def close(self):
		self.env.close()


def make_env(cfg):
	"""
	Make DMControl environment.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	normalized = cfg.task.replace('-', '_')
	domain, task = normalized.split('_', 1)
	domain = dict(cup='ball_in_cup', pointmass='point_mass').get(domain, domain)
	# For multi-word domains (e.g., humanoid_CMU), try progressively longer
	# domain prefixes until we find a match in suite.ALL_TASKS.
	if (domain, task) not in suite.ALL_TASKS:
		parts = normalized.split('_')
		for i in range(2, len(parts)):
			candidate_domain = '_'.join(parts[:i])
			candidate_task = '_'.join(parts[i:])
			if (candidate_domain, candidate_task) in suite.ALL_TASKS:
				domain, task = candidate_domain, candidate_task
				break
	if (domain, task) not in suite.ALL_TASKS:
		raise ValueError('Unknown task:', task)
	env = suite.load(domain,
					 task,
					 task_kwargs={'random': cfg.seed},
					 visualize_reward=False)
	env = action_scale.Wrapper(env, minimum=-1., maximum=1.)
	env = DMControlWrapper(env, domain, action_repeat=cfg.action_repeat)
	env = Timeout(env, max_episode_steps=1000//cfg.action_repeat)
	return env
