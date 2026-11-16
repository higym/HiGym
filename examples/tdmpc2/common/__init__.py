import random

import numpy as np
import torch


def set_seed(seed):
	"""Set seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def barrier():
	if torch.distributed.is_initialized():
		try:
			torch.distributed.barrier()
		except Exception:
			raise
