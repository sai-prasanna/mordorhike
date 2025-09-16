# One Does Not Simply Estimate State: Comparing Model-based and Model-free Reinforcement Learning on the Partially Observable MordorHike Benchmark

This repository contains the implementation for the EWRL 2025 paper "One Does Not Simply Estimate State: Comparing Model-based and Model-free Reinforcement Learning on the Partially Observable MordorHike Benchmark".


If you use our work in you research or application, please cite us:

```bibtex
@article{prasanna-ewrl2024a,
  title = {One Does Not Simply Estimate State: Comparing Model-based and Model-free Reinforcement Learning on the Partially Observable MordorHike Benchmark
},
  author = {Prasanna, Sai and Rajan, Raghu and Biedenkapp, André},
  year = {2025},
  journal = {European Workshop for Reinforcment Learning}
}
```

## Setup

```bash
# Install dependencies using uv
uv sync

# Activate the environment (optional, uv run commands work without activation)
source .venv/bin/activate
```

## Example Commands

### Training Dreamer Easy with Tuned Hyperparameters

```bash
uv run train-dreamer --logdir experiments/dreamer_easy_tuned --configs mordorhike --seed 42 --task "gymnasium_mordor-hike-easy-v0" --batch_size 8 --dyn.rssm.deter 256 --dyn.rssm.hidden 85 --dyn.rssm.classes 11 --opt.lr 0.00023854151368141174 --run.train_ratio 262 --.*\\.units 165
```

### Rollout and Evaluation

```bash
# Generate rollouts from trained model
uv run rollout-dreamer --logdir experiments/dreamer_easy_tuned

# Analyze results and generate plots  
uv run analyze --logdir experiments/dreamer_easy_tuned
```

### Links to Scripts

For full training runs with multiple seeds, see the SLURM scripts in the [`scripts/`](scripts/) directory:
- [`scripts/dreamer_easy_tuned.sh`](scripts/dreamer_easy_tuned.sh) - Dreamer training with tuned hyperparameters

## All Training Commands 
Example training commands for medium difficulty and one seed.
```bash

# Dreamer Medium Tuned
uv run train-dreamer --logdir experiments/dreamer_medium_tuned --configs mordorhike --seed 42 --task "gymnasium_mordor-hike-medium-v0" --batch_size 8 --dyn.rssm.deter 256 --dyn.rssm.hidden 85 --dyn.rssm.classes 11 --opt.lr 0.00023854151368141174 --run.train_ratio 262 --.*\\.units 165

# R2I Medium Tuned
uv run train-r2i --logdir experiments/r2i_medium_tuned --configs mordorhike --seed 42 --task "gymnasium_mordor-hike-medium-v0" --batch_size 4 --rssm.deter 1024 --rssm.units 1024 --rssm.hidden 512 --ssm.n_layers 4 --model_opt.lr 0.00017958212993107736 --actor_opt.lr 0.0002341129438718781 --critic_opt.lr 0.0002341129438718781 --.*\\.units 1024 --.*\\.mlp_units 1024 --.*\\.mlp_layers 1 --run.train_ratio 457
# DRQN Medium Tuned
uv run train-drqn --logdir experiments/drqn_medium_tuned --configs defaults --seed 42 --env.name "mordor-hike-medium-v0" --train.batch_size 126 --train.epsilon 0.2389 --drqn.hidden_size 160 --train.learning_rate 0.00022516182798426598 --train.num_gradient_steps 13 --train.train_every 35 --train.target_period 280
```

**Note:** Hyperparameter tuning was performed on medium difficulty using random search and applied across all difficulty levels.

## Rollout and Evaluation
Example evaluation for the above training commands.
```bash
# Generate rollouts
uv run rollout-dreamer --logdir experiments/dreamer_easy_tuned
uv run rollout-r2i --logdir experiments/r2i_easy_tuned
uv run rollout-drqn --logdir experiments/drqn_easy_tuned

# Analyze results and generate plots
uv run analyze --logdir experiments/dreamer_easy_tuned
uv run analyze --logdir experiments/r2i_easy_tuned
uv run analyze --logdir experiments/drqn_easy_tuned
```

## SLURM Scripts

Pre-configured SLURM training scripts for all seeds and configs are available in the `scripts/` directory:

```bash
# Hyperparameter tuned training (5 seeds: 42, 1337, 13, 19, 94)
sbatch scripts/dreamer_easy_tuned.sh
sbatch scripts/dreamer_medium_tuned.sh
sbatch scripts/dreamer_hard_tuned.sh

sbatch scripts/r2i_easy_tuned.sh
sbatch scripts/r2i_medium_tuned.sh
sbatch scripts/r2i_hard_tuned.sh

sbatch scripts/drqn_easy_tuned.sh
sbatch scripts/drqn_medium_tuned.sh
sbatch scripts/drqn_hard_tuned.sh

# To replicate hyperparameter tuning
sbatch scripts/tune_dreamer.sh
sbatch scripts/tune_r2i.sh
sbatch scripts/tune_drqn.sh
```