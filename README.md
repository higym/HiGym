# HiGym

Continuous-control environments whose embodiments scale to hundreds of
degrees of freedom (DoF): six MuJoCo families from the DeepMind Control
Suite, each generated at many sizes. HiGym installs as a Python package
and works like any [Gymnasium](https://gymnasium.farama.org) environment,
so you can train any algorithm or policy on it. Project page:
<https://higym.github.io>.

## Install

```bash
pip install git+https://github.com/higym/HiGym
```

Python 3.10–3.12. `dm-control` and `mujoco` are pinned to the versions the
paper used, and `numpy` is kept below 2, so every install generates the
same models.

## Quick start

```python
import gymnasium
import higym  # registers the higym/ ids

env = gymnasium.make("higym/cheetah-128_run-v0")  # or higym.make("cheetah-128_run")
obs, info = env.reset(seed=0)
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

`higym.list_tasks()` lists all task names. The environments work with
Gymnasium's vector environments and with any library that takes a
Gymnasium env. See `examples/random_policy.py` and `examples/vector_env.py`.

## Tasks

The 81 tasks of the paper's Table 1. Ids are
`higym/<domain>-<dof>_<task>-v0`, e.g. `higym/walker-32_walk-v0`, where
`<dof>` is the number of actuated degrees of freedom.

| Domain         | Tasks                                  | DoF                          |
|----------------|----------------------------------------|------------------------------|
| `quadruped`    | `walk`, `run`, `escape`, `fetch`       | 16, 32, 64, 128, 256, 512    |
| `cheetah`      | `run`                                  | 8, 16, 32, 64, 128, 256      |
| `walker`       | `stand`, `walk`, `run`                 | 8, 16, 32, 64, 128, 256      |
| `finger`       | `spin`, `turn_easy`, `turn_hard`       | 2, 4, 8, 16, 32, 64          |
| `reacher`      | `easy`, `hard`                         | 2, 4, 8, 16, 32, 64          |
| `humanoid_CMU` | `stand`, `walk`, `run`                 | 80                           |

## Protocol

The defaults reproduce the protocol the paper trained and evaluated with:

| Setting       | Default                                                                       |
|---------------|-------------------------------------------------------------------------------|
| Actions       | `Box(-1, 1)`, rescaled to each actuator's control range                       |
| Action repeat | 2; the rewards of the repeated steps are summed                               |
| Episode       | 1000 control steps (500 agent steps), then `truncated=True`; never `terminated` |
| Observations  | dm_control's observation dict, flattened to float32                           |
| Seeding       | `reset(seed=s)` seeds the task's initial states, targets and terrain          |
| Score         | `info["score"]` = episode return / 1000                                       |

Settings are keyword arguments:

```python
import higym

env = higym.make("walker-64_run", action_repeat=1)  # 1000 agent steps per episode
env = higym.make("walker-64_run", render_mode="rgb_array", width=320, height=240)
env.reset(seed=0)
frame = env.render()  # (240, 320, 3) uint8
```

`render_mode="rgb_array"` renders from camera 0 (camera 2 for quadruped);
pass `camera_id` to choose another. On a headless Linux machine set
`MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, needs `libosmesa6`).

Gymnasium ids exist once `higym` is imported. In a fresh process that has
not imported it, use `gymnasium.make("higym:higym/walker-64_run-v0")`,
which imports it first.

## dm_control access

```python
import higym

dm_env = higym.load("cheetah", "128_run", seed=0)  # a dm_control environment
timestep = dm_env.reset()

env = higym.make("cheetah-128_run")
env.unwrapped.dm_env  # the dm_control environment behind a Gymnasium env
```

Importing `higym` also adds the tasks to dm_control's own suites, so
`dm_control.suite.load("cheetah", "128_run")` works too.

## Ablations

The paper's gear-scaling ablations of Walker and Reacher are opt-in:

```python
import higym.ablations

higym.ablations.list_tasks()  # 24 names, e.g. "walker-256a25_run"
env = higym.make("walker-256a25_run")
```

`a<AA>` scales gear, stiffness and damping by `(baseline / n) ** (AA / 100)`,
`m` conserves limb mass, `p` restores DMControl's passive dynamics, and
`s<SSS>` sets Reacher's spring strength to `SSS / 1000` of max motor torque.
The two `s000` variants have no spring and are numerically unstable: their
episodes can stop with MuJoCo's `mjWARN_BADQACC`.

## Reproducing the paper

The paper's results were produced with python 3.10, dm-control 1.0.34,
mujoco 3.3.6, gymnasium 0.29.1 and numpy 1.24.4. To install exactly that
stack (python 3.10 or 3.11):

```bash
pip install "higym[paper] @ git+https://github.com/higym/HiGym"
```

MuJoCo simulations are deterministic on one platform but not bitwise
identical across CPU platforms, so seeded trajectories can differ between,
say, Linux x86-64 and macOS arm64.

## Training with TD-MPC2

`examples/tdmpc2/` holds a TD-MPC2 training pipeline built on this package.
Create its environment from the Docker image or the conda file, then install
HiGym into it from the repo root:

```bash
docker build -t higym:latest examples/tdmpc2/docker/
# or
conda env create -n higym -f examples/tdmpc2/docker/environment.yaml
conda activate higym

pip install -e .
cd examples/tdmpc2
python train.py task=cheetah-128_run seed=1 exp_name=cheetah-128
```

Common overrides:

- `task=<domain>-<dof>_<task>`: the environment to train on
- `steps=50_000_000`: total environment steps
- `seed=<int>`: random seed
- `enable_wandb=False`: disable W&B logging
- `wandb_project=<str> wandb_entity=<str>`: W&B project and entity (set in `examples/tdmpc2/config.py`)
- `checkpoint=<path>`: load a checkpoint for evaluation or fine-tuning
- `partial_checkpoint=True`: skip shape-mismatched layers (cross-DoF transfer)

Logs and checkpoints are written to `logs/<task>/<seed>/<exp_name>/`.

## Repository layout

```
src/higym/           # the package
  tasks/             # the task generators and base XMLs, one module per domain
  env.py             # HiGymEnv: the Gymnasium environment
  ablations.py       # opt-in ablation ids
tests/               # pytest suite (pytest -m "not slow" for the quick part)
examples/            # random_policy.py, vector_env.py, tdmpc2/
scripts/             # maintainer tools (renders, wheel check, reference rollouts)
```

## Citation

If you use HiGym, please cite the paper *HiGym: Learning to Act in
High-Dimensional Spaces* (<https://higym.github.io>).

## Acknowledgements

The environments build on the DeepMind Control Suite, and the Gymnasium
wrapper follows DrQ-v2's. The training example is built on TD-MPC2, with
math utilities adapted from DreamerV3.

## License

MIT; see `LICENSE`. The task modules and XMLs adapt the DeepMind Control
Suite (`dm_control`), which is licensed under Apache-2.0.
