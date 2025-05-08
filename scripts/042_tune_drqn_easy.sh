#!/bin/bash

#SBATCH --array=0-50%10
#SBATCH --partition alldlc_gpu-rtx2080
#SBATCH --job-name 042_tune_drqn_easy
#SBATCH --output experiments/slurm/%x-%A-%a.out
#SBATCH --error experiments/slurm/%x-%A-%a.err
#SBATCH --mem 16GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-core=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=dlcgpu22,dlcgpu12,dlcgpu11,dlcgpu16,dlcgpu07,dlcgpu23,dlcgpu24,dlcgpu25

echo "Workingdir: $PWD";
echo "Started at $(date)";


start=`date +%s`
job_name=$SLURM_JOB_NAME
uv run tune-drqn --env.name "mordor-hike-easy-v0" --neps_root_directory experiments/neps_drqn_easy --neps_env_steps_max 300000
end=`date +%s`
echo "Finished at $(date)";
echo "Time taken: $((end-start)) seconds";
