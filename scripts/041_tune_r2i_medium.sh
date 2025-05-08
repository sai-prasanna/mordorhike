#!/bin/bash

#SBATCH --array=0-50%10
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 041_tune_r2i_medium
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1

echo "Workingdir: $PWD";
echo "Started at $(date)";
export XLA_PYTHON_CLIENT_PREALLOCATE=false


start=`date +%s`
job_name=$SLURM_JOB_NAME
uv run tune-r2i --task "gymnasium_mordor-hike-medium-v0" --neps_root_directory experiments/neps_r2i_medium --neps_env_steps_max 300000 --neps_max_evaluations_total 50
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";
