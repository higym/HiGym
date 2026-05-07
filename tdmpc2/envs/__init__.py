from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym

from envs.wrappers.vectorized import make_vectorized_env
from envs.wrappers.render import Render
from envs.dm_control import make_env as make_dm_control_env


def make_env(cfg):
	"""
	Make an environment.
	"""
	gym.logger.set_level(40)
	if cfg.child_env or cfg.num_envs == 1:
		env = make_dm_control_env(cfg)
	else:
		env = make_vectorized_env(cfg, make_env)	
	cfg.obs_shape = {"state": env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	return env
