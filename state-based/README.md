

# **Unsupervised Skill Discovery through Skill Regions Differentiation**

This is the official codebase for our state-based experiments . This codebase is built on top of the [Unsupervised Reinforcement Learning Benchmark (URLB) codebase](https://github.com/rll-research/url_benchmark). Our method `SD3` is implemented in `agents/deviation.py` and the config is specified in `agents/deviation.yaml`.

To pre-train SD3, run the following command:

``` sh
python pretrain.py agent=deviation domain=walker seed=1
```

This script will produce several agent snapshots after training for `100k`, `500k`, `1M`, and `2M` frames and snapshots will be stored in `./models/states/<domain>/<agent>/<seed>/ `.

To finetune SD3, run the following command:

```sh
python finetune.py task=walker_stand obs_type=states agent=deviation reward_free=false seed=1 domain=walker snapshot_ts=2000000
```

## Requirements

We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with
```sh
conda activate urlb
```

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |

### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in the form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```
