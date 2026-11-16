# envs/wrappers/vectorized.py
from copy import deepcopy

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict


class VectorizedWrapper(gym.Wrapper):
	"""
	Wrapper for vectorized environments.
	"""

	def __init__(self, cfg, make_fn):
		self.cfg = deepcopy(cfg)

		# Create configs
		cfgs = [deepcopy(self.cfg) for _ in range(self.cfg.num_envs)]
		for i in range(len(cfgs)):
			cfgs[i].task = cfgs[i].tasks[i%self.cfg.num_tasks]
			cfgs[i].tasks = [cfgs[i].task]
			cfgs[i].num_envs = 1
			cfgs[i].child_env = True
			cfgs[i].save_video = False
			cfgs[i].seed = cfgs[i].seed + np.random.randint(1000)
		
		# Create environment
		wrapper = AsyncVectorEnv if self.cfg.get('env_mode', 'sync') == 'async' else SyncVectorEnv
		env_fns = [lambda c=cfgs[i]: make_fn(c) for i in range(self.cfg.num_envs)]
		self.env = wrapper(env_fns)
		super().__init__(self.env)
		
		# Create non-vectorized eval environment
		eval_cfg = deepcopy(cfgs[0])
		eval_cfg.save_video = self.cfg.save_video
		self.eval_env = make_fn(eval_cfg)
		self.observation_space = self.eval_env.observation_space
		self.action_space = self.eval_env.action_space
		self.is_training = True
		
		self._max_episode_steps = eval_cfg.episode_length
		self.max_episode_steps = self._max_episode_steps
		if self.cfg.rank == 0:
			print('Episode length:', self._max_episode_steps)

	def train(self, mode=True):
		self.is_training = mode
		return self

	def eval(self):
		return self.train(False)

	def rand_act(self):
		return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1
	
	def _preprocess_obs(self, obs):
		return torch.tensor(obs, dtype=torch.float32)

	def reset(self):
		obs, info = self.env.reset()

		# Override first env if eval
		if not self.is_training:
			obs[0], info[0] = self.eval_env.reset()

		obs = self._preprocess_obs(obs)
		info = self._preprocess_info(info)
		return obs, info
	
	def _preprocess_info(self, info):
		if 'final_info' in info:  # Handle final transitions
			assert 'final_observation' in info, \
				'Expected final observation in info when final_info is present.'
			fp64_to_fp32 = lambda x: x.astype(np.float32) if isinstance(x, np.ndarray) and x.dtype == np.float64 else x
			np_to_torch = lambda x: torch.from_numpy(fp64_to_fp32(x)) if isinstance(x, np.ndarray) else TensorDict({k: np_to_torch(v) for k, v in x.items()})
			info['final_observation'] = torch.stack([np_to_torch(d) for d in info['final_observation'] if d is not None])
			keys = next(d.keys() for d in info['final_info'] if d is not None)
			pad = lambda vals: torch.from_numpy(
				np.stack([np.asarray(v, dtype=np.float32) if v is not None
					else np.full_like(next(x for x in vals if x is not None), np.nan, dtype=np.float32) for v in vals]))
			info['final_info'] = {
				k: pad([d[k] if d is not None else None for d in info['final_info']]) for k in keys}
		info['success'] = torch.tensor(info['success'], dtype=torch.float32)
		if 'frame' in info:
			info['frame'] = torch.stack([torch.from_numpy(d) for d in info['frame']])
		return info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action.numpy())

		# Override first env if eval
		if not self.is_training:
			try:
				obs[0], reward[0], terminated[0], truncated[0], info[0] = self.eval_env.step(action[0].numpy())
			except TypeError as e:  # Env needs reset
				# TODO: implement this
				pass

		return self._preprocess_obs(obs), \
			   torch.tensor(reward, dtype=torch.float32), \
			   torch.tensor(terminated, dtype=torch.bool), \
			   torch.tensor(truncated, dtype=torch.bool), \
			   self._preprocess_info(info)
	
	def render(self, *args, **kwargs):
		if self.cfg.env_mode == 'async':
			# Render only the first environment in async mode
			assert not self.is_training, \
				'Rendering is only supported in eval mode for async environments.'
			return torch.from_numpy(self.eval_env.render(*args, **kwargs))
		frames = []
		for env in self.env.envs:
			frame = env.render(*args, **kwargs)
			if frame is not None:
				frames.append(torch.from_numpy(frame))
		return torch.cat(frames, dim=1) if len(frames) > 0 else None


def make_vectorized_env(cfg, make_fn):
	"""Make a vectorized environment."""
	print(f'[Rank {cfg.rank}] Creating vectorized environment with tasks: {cfg.tasks}')
	return VectorizedWrapper(cfg, make_fn)
