# trainer.py
import os
from collections import defaultdict, OrderedDict
from time import time

import torch
import numpy as np
from termcolor import colored
from tqdm import tqdm
from tensordict.tensordict import TensorDict

from common import barrier


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def empty_metrics():
	return {'reward': [], 'length': []}


class Trainer():
	"""Trainer class for MMBench experiments."""

	def __init__(self, cfg, env, agent, buffer, logger):
		self.cfg = cfg
		self.env = env
		self.agent = agent
		self.buffer = buffer
		self.logger = logger
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._tasks = torch.tensor(split_by_rank(range(cfg.num_global_tasks), cfg.rank, cfg.world_size),
			dtype=torch.int32, device='cpu')
		self._tasks = self._tasks.repeat_interleave(cfg.num_envs)
		self._tds = TensorDict({}, batch_size=(self.cfg.episode_length+1, self.cfg.num_envs), device='cpu')
		self._update_freq = self.cfg.num_envs * self.cfg.episode_length * self.cfg.world_size
		self._update_tokens = 0
		if cfg.rank == 0:
			print('Architecture:', self.agent.model)
			print(f'Update frequency: {self._update_freq:,}')
			print(f'Episode length: {self.cfg.episode_length:,}')
			print(f'Action repeat: {self.cfg.action_repeat:,}')

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	@torch.no_grad()
	def compute_value_estimates(self):
		"""
		Compute overestimation bias E_ρ0[V̂ − V^π] per TDM(PC)² (arXiv:2502.03550) Figure 1.

		V̂  = Q̂(s₀, π_mean(s₀))           — critic estimate at episode-initial states
		V^π = discounted MC return          — policy rollouts with deterministic mean action

		Returns a dict of scalar metrics ready to merge into eval_metrics.
		"""
		# Use the single eval env (avoids async/multi-env complexity)
		single_env = getattr(self.env, 'eval_env', self.env)
		device = self.agent.device
		task = self._tasks[0:1].to(device)  # shape (1,), int32
		n = self.cfg.value_bias_episodes

		self.agent.model.eval()
		true_values, estimated_values = [], []

		for _ in range(n):
			obs, _ = single_env.reset()

			# --- estimated value at initial state ---
			obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
			z0 = self.agent.model.encode(obs_t, task)
			_, pi0_info = self.agent.model.pi(z0, task)
			v_hat = self.agent.model.Q(
				z0, pi0_info['mean'], task, return_type='avg'
			).squeeze().item()
			estimated_values.append(v_hat)

			# --- MC rollout with deterministic policy mean ---
			G, gamma, done = 0.0, 1.0, False
			while not done:
				obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
				z = self.agent.model.encode(obs_t, task)
				_, pi_info = self.agent.model.pi(z, task)
				action_np = pi_info['mean'].squeeze(0).cpu().numpy()
				obs, reward, terminated, truncated, _ = single_env.step(action_np)
				G += gamma * reward
				gamma *= self.cfg.discount
				done = terminated or truncated

			true_values.append(G)

		true_v = float(np.mean(true_values))
		est_v = float(np.mean(estimated_values))
		bias = est_v - true_v

		return {
			'value/true_value': true_v,
			'value/estimated_value': est_v,
			'value/overestimation_bias': bias,
			'value/overestimation_ratio_pct': bias / (abs(true_v) + 1e-8) * 100.0,
		}

	def eval(self):
		"""Evaluate agent and aggregate all completed episodes per unique task name."""
		task_results = defaultdict(empty_metrics)

		self.env.eval()  # Set eval mode for vectorized env
		obs, info = self.env.reset()
		episode_reward = torch.zeros(self.cfg.num_envs)
		episode_len = torch.zeros(self.cfg.num_envs)
		episodes_completed = torch.zeros(self.cfg.num_envs, dtype=torch.int32)

		if self.cfg.save_video:
			self.logger.video.init(self.env, enabled=self.cfg.rank==0)

		while (episodes_completed < self.cfg.eval_episodes).any():
			torch.compiler.cudagraph_mark_step_begin()
			action = self.agent(obs, t0=episode_len==0, step=self._step, eval_mode=True, task=self._tasks, mpc=self._step > 0)
			obs, reward, terminated, truncated, info = self.env.step(action)

			done = terminated | truncated
			episode_reward += reward
			episode_len += 1

			if 'final_info' in info:
				for i in range(self.cfg.num_envs):
					if done[i]:
						task_id = self._tasks[i].item()
						task_name = self.cfg.global_tasks[task_id]

						task_results[task_name]['reward'].append(episode_reward[i].item())
						task_results[task_name]['length'].append(episode_len[i].item())

						episode_reward[i] = 0.0
						episode_len[i] = 0.0
						episodes_completed[i] += 1

			if self.cfg.save_video and episodes_completed.min() == 0:
				self.logger.video.record(self.env)

		if self.cfg.save_video:
			self.logger.video.save(self._step)

		barrier()  # Ensure all processes have completed evaluation

		if self.cfg.world_size > 1:
			# Gather results from all ranks
			gathered_results = [None for _ in range(self.cfg.world_size)] if self.cfg.rank == 0 else None
			torch.distributed.gather_object(task_results, gathered_results, dst=0)
			if self.cfg.rank == 0:
				# Combine results from all ranks
				for rank_results in gathered_results:
					for task_name, metrics in rank_results.items():
						for metric_name, values in metrics.items():
							task_results[task_name][metric_name].extend(values)

		results = {}

		# Sort tasks by order in cfg.global_tasks
		task_results = OrderedDict(sorted(task_results.items(), key=lambda x: self.cfg.global_tasks.index(x[0])))

		# Compute per-task averages
		for task_name, metrics in task_results.items():
			n = len(metrics['reward'])
			results[f'episode_reward+{task_name}'] = sum(metrics['reward']) / n
			results[f'episode_length+{task_name}'] = sum(metrics['length']) / n

		# Compute unweighted averages *across tasks*
		num_tasks = len(task_results)
		if self.cfg.rank == 0:
			assert num_tasks == self.cfg.num_global_tasks, \
				f'Number of eval tasks ({num_tasks}) does not match expected ({self.cfg.num_global_tasks})'
		results['episode_reward'] = sum(
			sum(m['reward']) / len(m['reward']) for m in task_results.values()
		) / num_tasks

		self.env.train()  # Set training mode for vectorized env

		# Overestimation bias (rank 0 only — single-env sequential rollouts)
		if self.cfg.rank == 0 and self.cfg.get('value_bias_eval', False):
			results.update(self.compute_value_estimates())

		return results

	def to_td(self, obs, action=None, reward=None, terminated=None, mu=None, std=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		if terminated is None:
			terminated = torch.tensor(False).repeat(self.cfg.num_envs)
		elif not isinstance(terminated, torch.Tensor):
			terminated = torch.stack(terminated.tolist())
		if mu is None:
			mu = torch.zeros(self.cfg.num_envs, self.cfg.action_dim)
		if std is None:
			std = torch.ones(self.cfg.num_envs, self.cfg.action_dim)
		td = TensorDict(
			obs=obs,
			action=action,
			reward=reward,
			terminated=terminated,
			task=self._tasks,
			mu=mu,
			std=std,
			batch_size=(self.cfg.num_envs,))
		return td
		
	def train(self):
		"""Train a Newt agent."""
		
		# Load checkpoint
		checkpoint = self.cfg.get('checkpoint', None)
		if checkpoint:
			if not os.path.exists(checkpoint):
				raise FileNotFoundError(f'Checkpoint file not found: {checkpoint}')
			if self.cfg.get('partial_checkpoint', False):
				loaded, skipped = self.agent.load_partial(self.cfg.checkpoint)
				if self.cfg.rank == 0:
					print(colored(f'Partial checkpoint loaded from {self.cfg.checkpoint}.', 'blue', attrs=['bold']))
					print(colored(f'  Transferred {len(loaded)} tensors; re-initialised {len(skipped)} (shape mismatch):', 'yellow', attrs=['bold']))
					for k in skipped:
						print(f'    [reinit] {k}')
			else:
				self.agent.load(self.cfg.checkpoint)
				if self.cfg.rank == 0:
					print(colored(f'Loaded checkpoint from {self.cfg.checkpoint}.', 'blue', attrs=['bold']))
		else:
			checkpoint = None
			if self.cfg.rank == 0:
				print(colored(f'No checkpoint found, training from scratch.', 'yellow', attrs=['bold']))

		# Training loop
		if self.cfg.rank == 0:
			print(f'Training agent for {self.cfg.steps:,} steps...')
		train_metrics = defaultdict(list)
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_metrics = self.eval()
				eval_metrics.update(self.common_metrics())
				self.logger.log(eval_metrics, 'eval')

				# Save agent
				if self._step % self.cfg.save_freq == 0 and self._step > 0:
					self.logger.save_agent(self.agent, f'{self._step:,}'.replace(',', '_'))

				# Reset environment and metrics
				obs, info = self.env.reset()
				ep_reward = torch.zeros((self.cfg.num_envs,))
				ep_len = torch.zeros((self.cfg.num_envs,), dtype=torch.int32)
				done = torch.full((self.cfg.num_envs,), True, dtype=torch.bool)
				self._next_action = None
				self._tds[ep_len] = self.to_td(obs)

			# Collect experience
			if self._step >= self.cfg.seeding_coef * self._update_freq:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent(obs, t0=done, step=self._step, task=self._tasks, mpc=True)
				plan_mu = self.agent._plan_mu.cpu()
				plan_std = self.agent._plan_std.cpu()
			else:
				action = self.env.rand_act()
				plan_mu = torch.zeros_like(action)
				plan_std = torch.ones_like(action)

			obs, reward, terminated, truncated, info = self.env.step(action)
			assert not terminated.any(), \
				f'Unexpected termination signal received.'
			ep_reward += reward
			ep_len += 1
			done = terminated | truncated
			self._step += self.cfg.num_envs * self.cfg.world_size

			# Store experience
			_obs = obs.clone()
			if 'final_observation' in info:
				_obs[done] = info['final_observation']
			td = self.to_td(_obs, action, reward, terminated, mu=plan_mu, std=plan_std)
			self._tds[ep_len] = td
			if done.any():
				assert done.all(), 'Expected all environments to be done at the same time.'
				max_ep_len = ep_len.max()

				for i in range(self.cfg.num_envs):
					if done[i]:
						assert ep_len[i] == self.cfg.episode_length, \
							f'Episode length {ep_len[i]} does not match expected length {self.cfg.episode_length}.'

						# Add to buffer
						_td = self._tds[:ep_len[i]+1, i].unsqueeze(0)
						self.buffer.add(_td, self.cfg.world_size, self.cfg.rank)

						# Save metrics
						train_metrics['episode_reward'].append(ep_reward[i].item())
						train_metrics['episode_length'].append(ep_len[i].item())
						train_metrics['episode_terminated'].append(terminated[i].item())

						# Reset episode metrics
						ep_reward[i] = 0.0
						ep_len[i] = 0
				
				# Log and reset metrics if enough data is collected
				if max_ep_len >= self.cfg.episode_length:
					assert self._step % self._update_freq == 0, \
						f'Step {self._step} is not a multiple of update frequency {self._update_freq}.'
					self._ep_idx += self.cfg.num_envs
					for key in ['episode_reward', 'episode_length', 'episode_terminated']:  # 'episode_success' 'episode_score',
						train_metrics[key] = torch.tensor(train_metrics[key], dtype=torch.float32).nanmean().item()
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					train_metrics = defaultdict(list)
			
			# Update agent
			if self._step >= self.cfg.seeding_coef * self._update_freq:
				self._update_tokens += self.cfg.num_envs * self.cfg.world_size * self.cfg.utd
				if self._update_tokens >= 1.0:
					num_updates = int(self._update_tokens)
					for _ in range(num_updates):
						_train_metrics = self.agent.update(self.buffer)
					train_metrics.update(_train_metrics)
					self._update_tokens -= num_updates
		
		self.logger.finish()
