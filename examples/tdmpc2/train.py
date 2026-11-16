# train.py
import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import hydra
from hydra.core.config_store import ConfigStore
from termcolor import colored
from tensordict import TensorDict

from common import barrier, set_seed
from common.buffer import Buffer, EnsembleBuffer
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, split_by_rank, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from trainer import Trainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def setup(rank, world_size, port):
	os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
	os.environ["MASTER_PORT"] = port
	torch.distributed.init_process_group(
		backend="nccl",
		rank=rank,
		world_size=world_size
	)
	return port


class DDPWrapper(nn.Module):
	def __init__(self, module: nn.Module):
		super().__init__()
		self._module = module  # Can be plain or DDP-wrapped

	def forward(self, *args, **kwargs):
		return self._module(*args, **kwargs)

	def __getattr__(self, name):
		if name == '_module':
			return super().__getattr__(name)
		try:
			return getattr(self._module, name)
		except AttributeError:
			# Try to unwrap once if wrapped by DDP
			if hasattr(self._module, 'module'):
				return getattr(self._module.module, name)
			raise

	def __setattr__(self, name, value):
		if name == '_module':
			super().__setattr__(name, value)
		else:
			setattr(self._module, name, value)

	def state_dict(self, *args, **kwargs):
		return self._module.state_dict(*args, **kwargs)

	def load_state_dict(self, *args, **kwargs):
		return self._module.load_state_dict(*args, **kwargs)


def train(rank: int, cfg: dict, buffer: Buffer):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.
	See config.yaml for a full list of args.
	"""
	if cfg.world_size > 1:
		setup(rank, cfg.world_size, cfg.port)
		print(colored('Rank:', 'yellow', attrs=['bold']), rank)
	set_seed(cfg.seed + rank)
	cfg.rank = rank
	torch.cuda.set_device(rank)

	def make_agent(cfg):
		model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
		if cfg.world_size > 1:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.rank])
			model = DDPWrapper(model)
		agent = TDMPC2(model, cfg)
		return agent

	trainer = Trainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=make_agent(cfg),
		buffer=buffer,
		logger=Logger(cfg),
	)
	barrier()  # Ensure all processes are ready before starting training
	try:
		trainer.train()
		if cfg.rank == 0:
			print('\nTraining completed successfully')
	except Exception as e:
		print(colored(f'[Rank {cfg.rank}] Training crashed with exception: {repr(e)}', 'red', attrs=['bold']))
		raise  # Propagate the exception so it isn't swallowed
	except KeyboardInterrupt:
		print(colored(f'[Rank {cfg.rank}] Training interrupted by user (Ctrl+C)', 'red', attrs=['bold']))
		raise  # Optional: raise again if you want the shell to show KeyboardInterrupt
	finally:
		print(colored('Training interrupted', 'red', attrs=['bold']))
		if torch.distributed.is_initialized():
			torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	assert cfg.world_size == 1, 'Multiprocessing is experimental'
	assert cfg.steps > 0, 'Must train for at least 1 step'
	cfg = parse_cfg(cfg)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# Set batch size
	cfg.world_size = torch.cuda.device_count() if cfg.multiproc else 1
	if cfg.world_size > 1:
		print(colored(f'Using {cfg.world_size} GPUs', 'green', attrs=['bold']))
		assert cfg.batch_size % cfg.world_size == 0, \
			'Batch size must be divisible by number of GPUs.'
		print(colored('Effective batch size:', 'yellow', attrs=['bold']), cfg.batch_size)
		cfg.batch_size = cfg.batch_size // cfg.world_size
		print(colored('Per-GPU batch size:', 'yellow', attrs=['bold']), cfg.batch_size)

	# Create buffer
	buffer_args = {
		'capacity': cfg.buffer_size,
		'batch_size': cfg.batch_size,
		'horizon': cfg.horizon,
		'multiproc': cfg.multiproc,
		'compile': cfg.compile,
	}
	buffer = Buffer(**buffer_args)

	if cfg.world_size > 1:
		cfg.port = os.getenv("MASTER_PORT", str(12355 + int(os.getpid()) % 1000))
		torch.multiprocessing.spawn(
			train,
			args=(cfg, buffer),
			nprocs=cfg.world_size,
			join=True,
		)
	else:
		train(0, cfg, buffer)


if __name__ == '__main__':
	launch()
