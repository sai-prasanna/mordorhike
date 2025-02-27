import argparse
import logging
from functools import partial
from pathlib import Path

import neps
import numpy as np
import ruamel.yaml as yaml
import torch
from dreamerv3 import embodied

from powm.algorithms.rollout_drqn import collect_rollouts
from powm.algorithms.train_drqn import DRQN, build_env
from powm.algorithms.train_drqn import main as train_main
from powm.utils import set_seed

logging.basicConfig(level=logging.DEBUG)

def evaluate_pipeline(pipeline_directory: Path, learning_rate: float, batch_size: int, hidden_size: int, num_layers: int, env_steps: int, train_every: int, target_update_mult: int, epsilon: float, num_gradient_steps: int, training_args: list[str]) -> float:
    
    """Evaluate a configuration by training and evaluating DRQN with multiple seeds."""
    seeds = [1337, 42, 13]  # Use 3 different seeds
    scores = []
    
    # Use the pipeline_directory provided by NEPS which contains a unique config ID
    base_logdir = embodied.Path(str(pipeline_directory))
    
    # Get original command line arguments and append our hyperparameters
    hparam_args = [
        f"--train.steps={env_steps}",
        f"--train.learning_rate={learning_rate}",
        f"--train.batch_size={batch_size}",
        f"--train.epsilon={epsilon}",
        f"--train.train_every={train_every}",
        f"--train.target_period={train_every * target_update_mult}",
        f"--train.num_gradient_steps={num_gradient_steps}",
        "--train.save_every=999999999",  # Only save final checkpoint
        f"--drqn.hidden_size={hidden_size}",
        f"--drqn.num_layers={num_layers}",
        "--wandb.project=",  # Disable wandb logging
        "--write_logs=False",  # Disable full logging during tuning, keep only terminal output
    ]
    
    for seed in seeds:
        # Create unique logdir for this run
        logdir = base_logdir / f"seed_{seed}"
        logdir.mkdir()
        
        # Train DRQN with combined arguments
        args = training_args + hparam_args + [
            f"--seed={seed}",
            f"--logdir={logdir}"
        ]
        train_main(args)
        
        # Load config and evaluate the last checkpoint
        config_path = logdir / "config.yaml"
        config = embodied.Config.load(config_path)
        checkpoint_path = logdir / "checkpoint.ckpt"
        
        # Create environment to get action and observation sizes
        env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
        env = build_env(config.env.name, env_kwargs, config.seed)
        
        # Create agent once
        agent = DRQN(
            action_size=env.action_space.n,
            observation_size=env.observation_space.shape[0],
            hidden_size=config.drqn.hidden_size,
            num_layers=config.drqn.num_layers,
            device=config.device,
        )
        
        # Load checkpoint
        checkpoint = torch.load(str(checkpoint_path), weights_only=False)
        agent.load_state_dict(checkpoint["agent"])
        
        # Evaluate the trained agent
        rollout_data = collect_rollouts(
            agent=agent,
            config=config,
            num_episodes=100,
            epsilon=0.0,
            collect_only_rewards=True,
        )
        scores.append(np.mean([sum(ep["reward"]) for ep in rollout_data]))
        
        # Close environment and delete the logdir to save space
        env.close()
        logdir.rmtree()

    # Return negative mean return across seeds (NEPS minimizes)
    score = -float(np.mean(scores))
    return score

def main():
    """Run hyperparameter optimization using NEPS."""
    # Parse any command line arguments for the script itself
    parser = argparse.ArgumentParser()
    neps_group = parser.add_argument_group('neps')
    neps_group.add_argument("--neps_root_directory", type=str, required=True, help="Root directory for optimization results")
    neps_group.add_argument("--neps_max_evaluations_total", type=int, default=100, help="Number of configurations to evaluate")
    neps_group.add_argument("--neps_max_evaluations_per_run", type=int, default=1, help="Number of configurations to evaluate per run")
    # env steps budget
    neps_group.add_argument("--neps_env_steps_min", type=int, default=100000, help="Minimum number of environment steps to evaluate for a configuration")
    neps_group.add_argument("--neps_env_steps_max", type=int, default=500000, help="Maximum number of environment steps to evaluate for a configuration")
    neps_group.add_argument("--neps_eta", type=int, default=2, help="eta for priorband")
    args, training_args = parser.parse_known_args()  # Use known_args to ignore training script args
    set_seed(42)
    # Define search space    
    pipeline_space = dict(
        learning_rate=neps.Float(
            lower=1e-5,
            upper=1e-2,
            log=True,
            prior=1e-3, 
        ),
        batch_size=neps.Integer(
            lower=32,
            upper=256,
            prior=32,
            log=True,
        ),
        hidden_size=neps.Integer(
            lower=32, 
            upper=512,
            prior=32,
            log=True,
        ),
        num_layers=neps.Integer(
            lower=1,
            upper=3,
            prior=2,
        ),
        env_steps=neps.Integer(
            lower=args.neps_env_steps_min,
            upper=args.neps_env_steps_max,
            is_fidelity=True
        ),
        train_every=neps.Integer(
            lower=10,
            prior=10,
            upper=100,
        ),
        target_update_mult=neps.Integer(
            lower=5,
            upper=10,
            prior=10,
        ),
        epsilon=neps.Float(
            lower=0.1,
            upper=0.3,
            prior=0.2,
        ),
        num_gradient_steps=neps.Integer(
            lower=5,
            upper=30,
            prior=10,
        ),
    )
    # Run optimization
    neps.run(
        evaluate_pipeline=partial(evaluate_pipeline, training_args=training_args, env_steps=args.neps_env_steps_max),
        pipeline_space=pipeline_space,
        optimizer=("random_search", {"use_priors": True}),
        root_directory=args.neps_root_directory,
        max_evaluations_total=args.neps_max_evaluations_total,
        max_evaluations_per_run=args.neps_max_evaluations_per_run,
        overwrite_working_directory=False,
        post_run_summary=True,
        ignore_errors=True
    )
if __name__ == "__main__":
    main()