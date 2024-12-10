# **Unsupervised Skill Discovery through Skill Regions Differentiation**

This is the code for our pixel-based experiments.  This codebase is built on top of the [Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels codebase](https://github.com/mazpie/mastering-urlb/). Our method `SD3` is implemented in `agents/deviation.py` and the config is specified in `agents/deviation_dreamer.yaml`.

## Requirements
The environment assumes you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```sh
conda activate urlb
```

## Domains and tasks
We support the following domains and tasks.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |

## Instructions
### Pre-training
To run pre-training use the `dreamer_pretrain.py` script
```sh
python dreamer_pretrain.py configs=dmc_pixels agent=deviation_dreamer domain=walker seed=1
```
This script will produce several agent snapshots after training for `100k`, `500k`, `1M`, and `2M` frames. The snapshots will be stored under the following directory:
```sh
./pretrained_models/<obs_type>/<domain>/<agent>/<seed>
```
For example:
```sh
./pretrained_models/pixels/walker/deviation/
```

### Fine-tuning
Once you have pre-trained your method, you can use the saved snapshots to initialize the `Dreamer` agent and fine-tune it on a downstream task. For example, you can fine-tune on `walker_stand by running the following command:
```sh
python dreamer_finetune.py configs=dmc_pixels agent=deviation_dreamer task=walker_stand snapshot_ts=2000000 seed=1
```
This will load a snapshot stored in `./pretrained_models/pixels/walker/deviation_dreamer/1/snapshot_2000000.pt`, initialize `Dreamer` with it, and start training on `walker_stand using the extrinsic reward of the task.

### Monitoring

#### Console
The console output is also available in a form:
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

#### Tensorboard
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```

#### Weights and Bias (wandb)
You can also use Weights and Bias, by launching the experiments with `use_wandb=True`.
