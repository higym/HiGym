# config.py
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any
import json
import hydra
from termcolor import colored
from omegaconf import OmegaConf


@dataclass
class Config:
	"""
	Config for experiments.
	"""

	# environment
	task: str = "finger-2-spin"							# see tdmpc2/common/__init__.py for task list
	num_envs: int = 20										# number of parallel environments
	episode_length: int = 500								# max episode length in environment steps
	action_repeat: int = 2									# number of environment steps per agent step
	env_mode: str = "async"									# environment mode, one of ["async", "sync"]

	# evaluation
	checkpoint: Optional[str] = None						# path to model checkpoint for evaluation / finetuning
	partial_checkpoint: bool = False						# if True, skip shape-mismatched layers (for cross-task transfer)
	eval_episodes: int = 2									# number of evaluation episodes per parallel environment

	# training
	steps: int = 50_000_000									# total environment steps to train for
	batch_size: int = 512									# effective batch size across all devices
	utd: float = 0.25										# update-to-data ratio, i.e., model updates per environment step
	reward_coef: float = 0.1								# coefficient for reward prediction loss
	value_coef: float = 0.1									# coefficient for value prediction loss
	consistency_coef: float = 20.0							# coefficient for latent consistency loss
	prior_coef: float = 1.0									# coefficient for policy prior regularization loss
	scale_threshold: float = 2.0							# min Q-scale before activating prior constraint
	rho: float = 0.5										# temporal weight coefficient for model losses
	lr: float = 3e-4										# global learning rate
	enc_lr_scale: float = 0.3								# encoder learning rate scale (wrt global lr)
	grad_clip_norm: float = 20.0							# gradient clipping norm
	tau: float = 0.01										# target value network update rate
	discount: float = 0.99									# discount factor for returns
	buffer_size: int = 1_000_000							# replay buffer capacity
	seeding_coef: int = 5									# number of random rollouts (per env) to seed the buffer with
	exp_name: str = "finger-2-spin-pclosstrue-s2"				# experiment name for logging

	# planning
	mpc: bool = True										# whether to use planning for action selection
	iterations: int = 6										# number of planning iterations
	num_samples: int = 512									# number of action sequences sampled per iteration
	num_elites: int = 64									# number of elite action sequences to refit distribution
	num_pi_trajs: int = 24									# number of action sequences to sample from policy prior
	horizon: int = 3										# planning horizon (also determines model training horizon)
	min_std: float = 0.05									# minimum action sampling stddev
	max_std: float = 2.0									# maximum action sampling stddev
	temperature: float = 0.5								# softmax temperature for mppi weighting
	bc_loss: bool = False

	# actor
	log_std_min: float = -10								# min log stddev for actor
	log_std_max: float = 2.0								# max log stddev for actor
	entropy_coef: float = 1e-4								# coefficient for actor entropy bonus

	# critic
	num_bins: int = 101										# number of bins for discrete regression
	vmin: float = -10.0										# min (log) value for discrete regression
	vmax: float = +10.0										# max (log) value for discrete regression

	# architecture
	num_enc_layers: int = 2									# number of encoder layers
	enc_dim: int = 256										# encoder mlp width
	mlp_dim: int = 512										# model mlp width
	latent_dim: int = 512									# model latent state dim
	num_q: int = 5											# number of Q-functions in ensemble
	simnorm_dim: int = 8									# number of dims per simplex in simplicial embedding layer

	# value overestimation bias logging (per TDM(PC)² Figure 1 methodology)
	value_bias_eval: bool = False							# whether to compute overestimation bias during eval
	value_bias_episodes: int = 100							# MC episodes to estimate true value (paper uses 100)

	# logging
	wandb_project: str = "finger_exp"							# wandb project name
	wandb_entity: str = "ADD_HERE"					# wandb entity (user) name
	enable_wandb: bool = True								# whether to enable wandb logging

	# misc
	multiproc: bool = False									# whether to use multiple GPUs (will use all visible GPUs)
	compile: bool = True									# whether to use torch.compile for model compilation (faster)
	render_size: int = 384									# render size for visualization
	save_video: bool = True									# whether to save evaluation videos
	save_agent: bool = True									# whether to save agent checkpoints
	seed: int = 2											# random seed

	# convenience (filled at runtime)
	work_dir: Optional[str] = None
	task_title: Optional[str] = None
	tasks: Any = None
	global_tasks: Any = None
	num_tasks: Optional[int] = None
	num_global_tasks: Optional[int] = None
	obs_shape: Any = None
	obs_shapes: Any = None
	eval_freq: Optional[int] = None
	save_freq: Optional[int] = None
	bin_size: Optional[float] = None
	rank: int = 0
	world_size: int = 1
	port: Optional[str] = None
	child_env: bool = False

	get = lambda self, val, default=None: getattr(self, val, default)


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def parse_cfg(cfg):
	"""
	Parses the experiment config dataclass. Mostly for convenience.
	"""
	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1)  # Bin size for discrete regression

	# Set defaults
	cfg.tasks = [cfg.task] * cfg.num_envs
	cfg.num_tasks = len(dict.fromkeys(cfg.tasks))  # Unique tasks
	cfg.global_tasks = deepcopy(cfg.tasks)
	cfg.num_global_tasks = cfg.num_tasks
	cfg.eval_freq = 10 * 500 * cfg.num_envs
	cfg.save_freq = 20 * cfg.eval_freq

	return OmegaConf.to_object(cfg)
