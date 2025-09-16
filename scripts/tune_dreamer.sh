#!/bin/bash

#SBATCH --array=0-50%10
#SBATCH --job-name tune_dreamer
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 24GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";

start=`date +%s`
job_name=$SLURM_JOB_NAME
uv run tune-dreamer --task "gymnasium_mordor-hike-medium-v0" --neps_root_directory experiments/neps_dreamer_medium --neps_env_steps_max 300000
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";