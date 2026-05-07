# HiGym

A DOF-scaling benchmark suite for continuous control, with a TD-MPC2-based
training pipeline for evaluating learning algorithms across embodiments of
varying degrees of freedom.

## Environments

Each domain registers `{domain}-{size}_{task}` variants on top of `dm_control`,
where `size` is the number of actuated DOF.

| Domain        | Tasks                       | DOF sizes                        |
|---------------|-----------------------------|----------------------------------|
| `cheetah`     | `run`                       | 4, 8, 16, 32, 64, 128, 256, 512  |
| `walker`      | `stand`, `walk`, `run`      | 4, 8, 16, 32, 64, 128, 256       |
| `quadruped`   | `walk`, `run`, `escape`, `fetch` | 16, 32, 64, 128, 256, 512   |
| `finger`      | `spin`, `turn_easy`, `turn_hard` | 2, 4, 8, 16, 32, 64         |
| `reacher`     | `easy`, `hard`              | 2, 4, 8, 16, 32, 64              |
| `humanoid_CMU`| `stand`, `walk`, `run`      | 56 (baseline), 80 (scaled)       |

Example task strings: `cheetah-128_run`, `walker-32_walk`, `finger-8_spin`.

## Installation

A reference Docker image is provided:

```bash
docker build -t higym:latest docker/
```

Or install dependencies from `docker/environment.yaml` into a conda env:

```bash
conda env create -n higym -f docker/environment.yaml
conda activate higym
```

## Training

Single-task training is launched via Hydra:

```bash
cd tdmpc2
python train.py task=cheetah-128_run seed=1 exp_name=cheetah-128
```

Common overrides:

- `task=<domain>-<size>_<task>` — environment to train on
- `steps=50_000_000` — total environment steps
- `seed=<int>` — random seed
- `enable_wandb=False` — disable W&B logging
- `wandb_project=<str> wandb_entity=<str>` — W&B project/entity (set in `tdmpc2/config.py`)
- `checkpoint=<path>` — load a checkpoint for evaluation or fine-tuning
- `partial_checkpoint=True` — skip shape-mismatched layers (cross-DOF transfer)

Logs and checkpoints are written to `logs/<task>/<seed>/<exp_name>/`.

## Repository layout

```
tdmpc2/
  train.py           # Hydra entry point
  trainer.py         # training loop
  tdmpc2.py          # TD-MPC2 agent
  config.py          # Config dataclass
  common/            # buffer, world model, logger, math utils
  envs/
    dm_control.py    # gym-style wrapper
    tasks/           # DOF-scaled task registrations (cheetah, walker, ...)
    wrappers/        # Timeout, vectorized, render
docker/              # Dockerfile + conda environment
render_embodiments.py  # offscreen renders of each embodiment
```

## Acknowledgements

The training pipeline is built on TD-MPC2 and uses environments from DeepMind
Control Suite. Math utilities are adapted from DreamerV3 and the gym wrapper
follows DrQ-v2.

## License

MIT — see `LICENSE`.
