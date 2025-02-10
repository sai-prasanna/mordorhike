import argparse
import logging
import time
from functools import partial
from pathlib import Path

import neps
import numpy as np
from dreamerv3 import embodied

from powm.algorithms.rollout_dreamer import collect_rollouts
from powm.algorithms.train_dreamer import main as train_main
from powm.algorithms.train_dreamer import make_env
from powm.utils import set_seed

logging.basicConfig(level=logging.INFO)

def evaluate_pipeline(pipeline_directory: Path, learning_rate: float, deter_size: int, 
                     hidden_size: int, classes: int, units: int, env_steps: int, 
                     train_ratio: int, batch_size: int, training_args: list[str]) -> float:
    """Evaluate a configuration by training and evaluating Dreamer with multiple seeds."""
    seeds = [1337, 42, 13, 5, 94]  # Use 5 different seeds
    scores = []
    
    # Use the pipeline_directory provided by NEPS which contains a unique config ID
    base_logdir = embodied.Path(str(pipeline_directory))
    
    # Get original command line arguments and append our hyperparameters
    hparam_args = [
        f"--run.steps={env_steps}",
        f"--opt.lr={learning_rate}",
        f"--dyn.rssm.deter={deter_size}",
        f"--dyn.rssm.hidden={hidden_size}",
        f"--dyn.rssm.classes={classes}",
        f"--run.batch_size={batch_size}",
        f"--.*\\.units={units}",  # Set units for all layers
        f"--run.train_ratio={train_ratio}",
        "--run.save_each_ckpt=False",  # Disable intermediate checkpoints
        "--wandb.project=",  # Disable wandb logging
        "--write_logs=False",  # Disable full logging during tuning
        "--configs=mordorhike",  # Use mordorhike config
    ]
    
    for seed in seeds:
        # Create unique logdir for this run
        logdir = base_logdir / f"seed_{seed}"
        logdir.mkdir()
        
        # Train Dreamer with combined arguments
        args = training_args + hparam_args + [
            f"--seed={seed}",
            f"--logdir={logdir}"
        ]
        
        train_main(args)
        
        # Evaluate the trained agent
        config = embodied.Config.load(logdir / "config.yaml")
        env = make_env(config, 0)
        driver = embodied.Driver([partial(make_env, config, i) for i in range(config.run.num_envs)])
        
        rollout_data = collect_rollouts(
            checkpoint_path=logdir / "checkpoint.ckpt",
            config=config,
            driver=driver,
            num_episodes=100,
            epsilon=0.0,
            collect_only_rewards=True,
        )
        scores.append(np.mean([sum(ep["reward"]) for ep in rollout_data]))
        
        # Clean up
        driver.close()
        env.close()
        logdir.rmtree()

    # Return negative mean return across seeds (NEPS minimizes)
    return -np.mean(scores)

def main():
    """Run hyperparameter optimization using NEPS."""
    parser = argparse.ArgumentParser()
    neps_group = parser.add_argument_group('neps')
    neps_group.add_argument("--neps_root_directory", type=str, required=True)
    neps_group.add_argument("--neps_max_evaluations_total", type=int, default=50)
    neps_group.add_argument("--neps_max_evaluations_per_run", type=int, default=1)
    neps_group.add_argument("--neps_env_steps_min", type=int, default=100000)
    neps_group.add_argument("--neps_env_steps_max", type=int, default=500000)
    neps_group.add_argument("--neps_eta", type=int, default=2)
    args, training_args = parser.parse_known_args()
    
    set_seed(42)
    
    # Define search space    
    pipeline_space = dict(
        learning_rate=neps.Float(
            lower=1e-5,
            upper=1e-3,
            log=True,
            default=4e-5,
        ),
        batch_size=neps.Integer(
            lower=8,
            upper=128,
            default=16,
            log=True,
        ),
        deter_size=neps.Integer(
            lower=64,
            upper=1024,
            default=1024,
            log=True,
        ),
        hidden_size=neps.Integer(
            lower=32,
            upper=128,
            default=128,
            log=True,
        ),
        classes=neps.Integer(
            lower=8,
            upper=32,
            default=8,
            log=True,
        ),
        units=neps.Integer(
            lower=32,
            upper=256,
            default=128,
            log=True,
        ),
        env_steps=neps.Integer(
            lower=args.neps_env_steps_min,
            upper=args.neps_env_steps_max,
            is_fidelity=True
        ),
        train_ratio=neps.Integer(
            lower=128,
            upper=1024,
            default=512,
            log=True,
        ),
    )
    
    # Run optimization
    neps.run(
        run_pipeline=partial(evaluate_pipeline, training_args=training_args),
        pipeline_space=pipeline_space,
        searcher="priorband",
        root_directory=args.neps_root_directory,
        max_evaluations_total=args.neps_max_evaluations_total,
        max_evaluations_per_run=args.neps_max_evaluations_per_run,
        overwrite_working_directory=False,
        post_run_summary=True,
        eta=args.neps_eta,
    )

if __name__ == "__main__":
    main() 