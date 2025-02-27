import argparse
import logging
import time
from functools import partial
from pathlib import Path

import neps
import numpy as np
from recall2imagine import embodied
from recall2imagine.train import make_env, make_envs

from powm.algorithms.rollout_r2i import collect_rollouts
from powm.algorithms.train_r2i import main as train_main
from powm.utils import set_seed

logging.basicConfig(level=logging.INFO)

def evaluate_pipeline(pipeline_directory: Path, log_rssm_deter: int, 
                     log_rssm_units: int, log_units: int, mlp_layers: int, ssm_layers: int, env_steps: int, 
                     train_ratio: int, wm_lr: float, actor_critic_lr: float, batch_size: int, training_args: list[str]) -> float:
    """Evaluate a configuration by training and evaluating R2I with multiple seeds."""
    seeds = [1337, 42, 13]  # Use 3 different seeds
    scores = []
    
    # Use the pipeline_directory provided by NEPS which contains a unique config ID
    base_logdir = embodied.Path(str(pipeline_directory))
    
    rssm_deter = 2 ** ( 8 + log_rssm_deter)
    rssm_units = 2 ** ( 8 + log_rssm_units)
    units = 2 ** ( 8 + log_units)
    
    # Get original command line arguments and append our hyperparameters
    hparam_args = [
        f"--run.steps={env_steps}",
        f"--batch_size={batch_size}",
        f"--rssm.deter={rssm_deter}",
        f"--rssm.units={rssm_units}",
        f"--rssm.hidden={rssm_units//2}",
        f"--ssm.n_layers={ssm_layers}",
        f"--model_opt.lr={wm_lr}",
        f"--actor_opt.lr={actor_critic_lr}",
        f"--critic_opt.lr={actor_critic_lr}",
        f"--.*\\.units={units}",  # Set units for all layers
        f"--.*\\.mlp_units={units}",
        f"--.*\\.mlp_layers={mlp_layers}",
        f"--run.train_ratio={train_ratio}",
        "--run.save_each_ckpt=False",  # Disable intermediate checkpoints
        "--wandb.project=",  # Disable wandb logging
        "--write_logs=False",  # Disable full logging during tuning
        "--configs=mordorhike",  # Use mordorhike config
    ]
    
    for seed in seeds:
        # Create unique logdir for this run
        logdir = base_logdir / f"seed_{seed}"
        logdir.mkdirs()
        
        # Train R2I with combined arguments
        args = training_args + hparam_args + [
            f"--seed={seed}",
            f"--logdir={logdir}"
        ]
        
        train_main(args)
        
        # Evaluate the trained agent
        config = embodied.Config.load(logdir / "config.yaml")
        env = make_envs(config)
        
        rollout_data = collect_rollouts(
            checkpoint_path=logdir / "checkpoint.ckpt",
            config=config,
            env=env,
            num_episodes=100,
            epsilon=0.0,
            collect_only_rewards=True,
        )
        scores.append(np.mean([sum(ep["reward"]) for ep in rollout_data]))
        
        env.close()
        logdir.rmtree()

    # Return negative mean return across seeds (NEPS minimizes)
    return -np.mean(scores)

def main():
    """Run hyperparameter optimization using NEPS."""
    parser = argparse.ArgumentParser()
    neps_group = parser.add_argument_group('neps')
    neps_group.add_argument("--neps_root_directory", type=str, required=True)
    neps_group.add_argument("--neps_max_evaluations_total", type=int, default=100)
    neps_group.add_argument("--neps_max_evaluations_per_run", type=int, default=1)
    neps_group.add_argument("--neps_env_steps_min", type=int, default=100000)
    neps_group.add_argument("--neps_env_steps_max", type=int, default=500000)
    neps_group.add_argument("--neps_eta", type=int, default=2)
    args, training_args = parser.parse_known_args()
    
    set_seed(42)
    
    # Define search space    
    pipeline_space = dict(
        wm_lr=neps.Float(
            lower=1e-5,
            upper=1e-3,
            log=True,
            prior=1e-4,
        ),
        actor_critic_lr=neps.Float(
            lower=1e-5,
            upper=1e-3,
            log=True,
            prior=3e-5,
        ),
        batch_size=neps.Integer(
            lower=4,
            upper=32,
            prior=4,
            log=True,
        ),
        log_rssm_deter=neps.Integer(
            lower=0,
            upper=2,
            prior=2,
        ),
        log_rssm_units=neps.Integer(
            lower=0,
            upper=2,
            prior=2,
        ),
        log_units=neps.Integer(
            lower=0,
            upper=2,
            prior=2,
        ),
        mlp_layers=neps.Integer(
            lower=1,
            upper=3,
            prior=1,
        ),
        ssm_layers=neps.Integer(
            lower=2,
            upper=4,
            prior=3,
        ),
        env_steps=neps.Integer(
            lower=args.neps_env_steps_min,
            upper=args.neps_env_steps_max,
            is_fidelity=True
        ),
        train_ratio=neps.Integer(
            lower=128,
            upper=1024,
            prior=512,
            log=True,
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